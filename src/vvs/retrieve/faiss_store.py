"""On-disk FAISS ANN + SQLite payload. Vectors are mmap'd, not copied into NumPy."""

from __future__ import annotations

import sqlite3
import threading
from typing import Any

import numpy as np

from vvs.paths import video_dir
from vvs.retrieve.store import StoreHit, VectorPayload


# IVF-PQ needs ~39 training vectors per list. Below this, use HNSW / flat.
_IVF_MIN_N = 256


class FaissStore:
    def __init__(self, video_id: str, name: str, *, index_type: str = "hnsw") -> None:
        self.video_id = video_id
        self.name = name
        self.index_type = (index_type or "hnsw").strip().lower()
        self.dim = 0
        self._index = None
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.Lock()
        root = video_dir(video_id)
        root.mkdir(parents=True, exist_ok=True)
        self.index_path = root / f"{name}.faiss"
        self.sqlite_path = root / f"{name}.sqlite"

    def create(self, dim: int) -> None:
        self.dim = int(dim)
        self._open_db()
        with self._lock:
            self._exec(
                "INSERT OR REPLACE INTO kv(k, v) VALUES ('dim', ?)",
                (str(self.dim),),
            )
            assert self._conn is not None
            self._conn.commit()
        self._index = None
        if self.index_path.exists():
            self.index_path.unlink()

    def open(self, dim: int) -> None:
        import faiss

        self._open_db()
        stored = self._kv("dim")
        if stored:
            store_dim = int(stored)
            if int(dim) != store_dim:
                print(
                    f"[store] {self.name} dim mismatch: store={store_dim} requested={dim}; "
                    "using store"
                )
            self.dim = store_dim
        else:
            self.dim = int(dim)
        if not self.index_path.exists():
            self._index = None
            return
        # IO_FLAG_MMAP keeps the posting lists on disk; do not np.load a second copy.
        self._index = faiss.read_index(str(self.index_path), faiss.IO_FLAG_MMAP)

    def add(self, vectors: np.ndarray, payloads: list[VectorPayload]) -> None:
        import faiss

        if vectors.ndim != 2:
            raise ValueError("vectors must be (n, dim)")
        if len(payloads) != vectors.shape[0]:
            raise ValueError("payloads length must match vectors")
        if vectors.shape[0] == 0:
            return
        mat = np.ascontiguousarray(vectors.astype(np.float32))
        if self.dim == 0:
            self.dim = mat.shape[1]
        elif mat.shape[1] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {mat.shape[1]}")

        start_id = self._next_id()
        ids = np.arange(start_id, start_id + mat.shape[0], dtype=np.int64)
        for i, payload in enumerate(payloads):
            payload.vector_id = int(ids[i])

        if self._index is None:
            base = self._build_base(mat)
            self._index = faiss.IndexIDMap2(base)
        self._index.add_with_ids(mat, ids)
        self._write_payloads(payloads)
        self._kv_set("next_id", str(int(ids[-1]) + 1))
        self._persist_index()

    def search(
        self,
        query_vec: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[StoreHit]:
        if self._index is None or self._index.ntotal == 0 or top_k <= 0:
            return []
        q = np.ascontiguousarray(np.asarray(query_vec, dtype=np.float32).reshape(1, -1))
        if q.shape[1] != self._index.d:
            raise ValueError(f"query dim {q.shape[1]} != index dim {self._index.d}")
        over = min(self._index.ntotal, max(top_k * 4, top_k) if filters else top_k)
        self._tune_nprobe()
        scores, ids = self._index.search(q, int(over))
        hits: list[StoreHit] = []
        by_id = self._payloads([int(i) for i in ids[0] if int(i) >= 0])
        for score, raw_id in zip(scores[0], ids[0], strict=False):
            vid = int(raw_id)
            if vid < 0:
                continue
            payload = by_id.get(vid)
            if payload is None:
                continue
            if filters and not _payload_matches(payload, filters):
                continue
            hits.append(StoreHit(vector_id=vid, score=float(score), payload=payload))
            if len(hits) >= top_k:
                break
        return hits

    def ntotal(self) -> int:
        if self._index is None:
            return 0
        return int(self._index.ntotal)

    def all_payloads(self, modality: str | None = None) -> list[VectorPayload]:
        """Every payload in insertion order. Used to re-embed a store in place."""
        self._open_db()
        assert self._conn is not None
        sql = (
            "SELECT vector_id, asset_id, t_s, modality, path, text, frame_index, "
            "capture_ts, site_id, shot_id, motion_energy, flash_energy FROM payload"
        )
        params: tuple[Any, ...] = ()
        if modality:
            sql += " WHERE modality = ?"
            params = (modality,)
        sql += " ORDER BY vector_id"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [
            VectorPayload(
                vector_id=int(row[0]),
                asset_id=str(row[1]),
                t_s=float(row[2]),
                modality=str(row[3]),
                path=str(row[4] or ""),
                text=row[5],
                frame_index=row[6],
                capture_ts=row[7],
                site_id=row[8],
                shot_id=row[9],
                motion_energy=float(row[10] or 0.0),
                flash_energy=float(row[11] or 0.0),
            )
            for row in rows
        ]

    def close(self) -> None:
        self._index = None
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _build_base(self, train: np.ndarray):
        import faiss

        n, dim = train.shape
        want_ivf = self.index_type == "ivf_pq" and n >= _IVF_MIN_N
        if want_ivf:
            built = self._try_ivf_pq(train)
            if built is not None:
                return built
        # Small-N POC (and IVF fallback): HNSW over inner product on L2-normalized vecs.
        try:
            index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        except TypeError:
            index = faiss.IndexFlatIP(dim)
            return index
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 32
        return index

    def _try_ivf_pq(self, train: np.ndarray):
        import faiss

        n, dim = train.shape
        nlist = int(max(1, min(n // 39, 4 * int(np.sqrt(n)))))
        if nlist < 1 or n < nlist * 39:
            return None
        m = 8 if dim % 8 == 0 else 4
        if dim % m != 0:
            return None
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
        index.train(train)
        index.nprobe = min(16, nlist)
        return index

    def _tune_nprobe(self) -> None:
        inner = self._unwrap()
        if inner is not None and hasattr(inner, "nprobe"):
            nlist = int(getattr(inner, "nlist", 1) or 1)
            inner.nprobe = min(16, max(1, nlist))

    def _unwrap(self):
        index = self._index
        if index is None:
            return None
        return getattr(index, "index", index)

    def _persist_index(self) -> None:
        import faiss

        if self._index is None:
            return
        tmp = self.index_path.with_suffix(".faiss.tmp")
        faiss.write_index(self._index, str(tmp))
        tmp.replace(self.index_path)
        # Re-open mmap'd so subsequent search does not pin a write-copy.
        self._index = faiss.read_index(str(self.index_path), faiss.IO_FLAG_MMAP)

    def _open_db(self) -> None:
        if self._conn is not None:
            return
        # Gradio runs search on a worker thread; analyze/load may open on another.
        self._conn = sqlite3.connect(str(self.sqlite_path), check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS payload (
                vector_id INTEGER PRIMARY KEY,
                asset_id TEXT NOT NULL,
                t_s REAL NOT NULL,
                modality TEXT NOT NULL,
                path TEXT,
                text TEXT,
                frame_index INTEGER,
                capture_ts TEXT,
                site_id TEXT
            )
            """
        )
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(payload)")}
        if "shot_id" not in cols:
            self._conn.execute("ALTER TABLE payload ADD COLUMN shot_id INTEGER")
        if "motion_energy" not in cols:
            self._conn.execute("ALTER TABLE payload ADD COLUMN motion_energy REAL DEFAULT 0")
        if "flash_energy" not in cols:
            self._conn.execute("ALTER TABLE payload ADD COLUMN flash_energy REAL DEFAULT 0")
        self._conn.commit()

    def _next_id(self) -> int:
        value = self._kv("next_id")
        return int(value) if value else 1

    def _kv(self, key: str) -> str | None:
        assert self._conn is not None
        with self._lock:
            row = self._conn.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return None if row is None else str(row[0])

    def _kv_set(self, key: str, value: str) -> None:
        assert self._conn is not None
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO kv(k, v) VALUES (?, ?)", (key, value))
            self._conn.commit()

    def _exec(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        assert self._conn is not None
        self._conn.execute(sql, params)

    def _write_payloads(self, payloads: list[VectorPayload]) -> None:
        assert self._conn is not None
        with self._lock:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO payload
                (vector_id, asset_id, t_s, modality, path, text, frame_index, shot_id,
                 motion_energy, flash_energy, capture_ts, site_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        p.vector_id,
                        p.asset_id,
                        p.t_s,
                        p.modality,
                        p.path,
                        p.text,
                        p.frame_index,
                        p.shot_id,
                        p.motion_energy,
                        p.flash_energy,
                        p.capture_ts,
                        p.site_id,
                    )
                    for p in payloads
                ],
            )
            self._conn.commit()

    def _payloads(self, ids: list[int]) -> dict[int, VectorPayload]:
        if not ids or self._conn is None:
            return {}
        q = ",".join("?" * len(ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT vector_id, asset_id, t_s, modality, path, text, frame_index, "
                f"capture_ts, site_id, shot_id, motion_energy, flash_energy "
                f"FROM payload WHERE vector_id IN ({q})",
                ids,
            ).fetchall()
        out: dict[int, VectorPayload] = {}
        for row in rows:
            payload = VectorPayload(
                vector_id=int(row[0]),
                asset_id=str(row[1]),
                t_s=float(row[2]),
                modality=str(row[3]),
                path=str(row[4] or ""),
                text=row[5],
                frame_index=row[6],
                capture_ts=row[7],
                site_id=row[8],
                shot_id=row[9] if len(row) > 9 else None,
                motion_energy=float(row[10] or 0.0) if len(row) > 10 else 0.0,
                flash_energy=float(row[11] or 0.0) if len(row) > 11 else 0.0,
            )
            out[payload.vector_id] = payload
        return out


def _payload_matches(payload: VectorPayload, filters: dict[str, Any]) -> bool:
    for key, value in filters.items():
        if value is None:
            continue
        got = getattr(payload, key, None)
        if isinstance(value, (list, tuple, set)):
            if got not in value:
                return False
        elif got != value:
            return False
    return True
