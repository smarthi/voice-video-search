"""Visual language for the Gradio demo. Style only — no retrieval logic."""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from gradio.themes.utils.colors import Color
from gradio.themes.utils.fonts import Font

CSS_PATH = Path(__file__).with_name("theme.css")

# Accent red + ink neutrals for the dark theme.
ACCENT_RED = Color(
    name="accent_red",
    c50="#FFE8E6",
    c100="#FFC7C2",
    c200="#FF9A90",
    c300="#FF6A45",
    c400="#F5372E",
    c500="#E2231A",
    c600="#C41D16",
    c700="#9A1611",
    c800="#70110D",
    c900="#4A0B08",
    c950="#2A0604",
)

NEUTRAL_INK = Color(
    name="neutral_ink",
    c50="#EAEEF4",
    c100="#C5CDD8",
    c200="#8B98A9",
    c300="#5A6675",
    c400="#3A4656",
    c500="#26313F",
    c600="#1A2331",
    c700="#131B26",
    c800="#0E141D",
    c900="#0A0F17",
    c950="#05070C",
)

# Library preview behaviour. Registered from `head` rather than Blocks' `js=`
# hook: that hook evaluates its payload as statements, so passing a function
# expression there silently defines and discards it. Listeners are delegated on
# `document` because the grid is re-rendered as raw HTML whenever the archive
# changes, which would throw away per-element handlers.
_LIBRARY_JS = """
<script>
(() => {
  const tile = (event) => {
    const video = event.target;
    if (!(video instanceof HTMLVideoElement)) return null;
    return video.classList.contains('sl-lib-vid') ? video : null;
  };
  document.addEventListener('mouseover', (event) => {
    const video = tile(event);
    if (!video) return;
    video.muted = true;
    video.play().catch(() => {});
  }, true);
  document.addEventListener('mouseout', (event) => {
    const video = tile(event);
    if (!video) return;
    video.pause();
    video.currentTime = 0;
  }, true);
  // Touch devices never hover, so a tap toggles playback instead.
  document.addEventListener('click', (event) => {
    const video = tile(event);
    if (!video) return;
    video.muted = true;
    if (video.paused) video.play().catch(() => {});
    else video.pause();
  }, true);
})();
</script>
"""

HEAD = '<meta name="theme-color" content="#05070C">' + _LIBRARY_JS
FORCE_DARK_JS = "document.documentElement.classList.add('dark');"

TOPBAR_HTML = """
<div class="sl-top">
  <svg class="sl-arcs" viewBox="0 0 420 420" aria-hidden="true">
    <g fill="none" stroke="#E2231A" stroke-opacity=".14">
      <circle cx="210" cy="210" r="120"/>
      <circle cx="210" cy="210" r="160"/>
      <circle cx="210" cy="210" r="200"/>
      <circle cx="210" cy="210" r="240"/>
    </g>
  </svg>
  <div class="sl-brand">
    <svg class="sl-sphere" viewBox="0 0 24 24" fill="none" stroke="#E2231A" stroke-width="1.6" aria-hidden="true">
      <circle cx="12" cy="12" r="9"/>
      <ellipse cx="12" cy="12" rx="4" ry="9"/>
      <path d="M3.2 9h17.6M3.2 15h17.6"/>
    </svg>
    <div class="sl-mark"><b>Voice</b><span>Video Search</span></div>
  </div>
  <div class="sl-pills">
    <span class="sl-pill"><i class="sl-dot on"></i>Local</span>
    <span class="sl-pill"><i class="sl-dot ice"></i>Qwen-VL · BGE · OCR</span>
  </div>
</div>
"""

RESULTS_CHROME = """
<div class="sl-rhead">
  <span class="sl-lab">Moments</span>
  <span class="sl-corpus">Archive search · clips from every indexed video · still · start–end · confidence</span>
</div>
"""


def load_css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def build_theme() -> gr.themes.Base:
    return gr.themes.Base(
        primary_hue=ACCENT_RED,
        secondary_hue=ACCENT_RED,
        neutral_hue=NEUTRAL_INK,
        font=(Font("Helvetica Neue"), "Helvetica", "Arial", "ui-sans-serif", "sans-serif"),
        font_mono=("ui-monospace", "SF Mono", "Menlo", "Consolas", "Liberation Mono", "monospace"),
    ).set(
        body_background_fill="#05070C",
        body_background_fill_dark="#05070C",
        body_text_color="#EAEEF4",
        body_text_color_dark="#EAEEF4",
        body_text_color_subdued="#8B98A9",
        body_text_color_subdued_dark="#8B98A9",
        background_fill_primary="#0A0F17",
        background_fill_primary_dark="#0A0F17",
        background_fill_secondary="#0E141D",
        background_fill_secondary_dark="#0E141D",
        border_color_primary="#1A2331",
        border_color_primary_dark="#1A2331",
        border_color_accent="#E2231A",
        border_color_accent_dark="#E2231A",
        color_accent="#E2231A",
        color_accent_soft="rgba(226,35,26,.14)",
        color_accent_soft_dark="rgba(226,35,26,.14)",
        link_text_color="#7FD4FF",
        link_text_color_dark="#7FD4FF",
        link_text_color_hover="#EAEEF4",
        link_text_color_hover_dark="#EAEEF4",
        block_background_fill="#0A0F17",
        block_background_fill_dark="#0A0F17",
        block_border_color="#1A2331",
        block_border_color_dark="#1A2331",
        block_border_width="1px",
        block_radius="9px",
        block_shadow="none",
        block_shadow_dark="none",
        block_title_text_color="#EAEEF4",
        block_title_text_color_dark="#EAEEF4",
        block_title_text_weight="600",
        block_label_background_fill="#0E141D",
        block_label_background_fill_dark="#0E141D",
        block_label_text_color="#8B98A9",
        block_label_text_color_dark="#8B98A9",
        block_label_text_weight="500",
        block_info_text_color="#5A6675",
        block_info_text_color_dark="#5A6675",
        panel_background_fill="#0A0F17",
        panel_background_fill_dark="#0A0F17",
        panel_border_color="#1A2331",
        panel_border_color_dark="#1A2331",
        input_background_fill="#03050A",
        input_background_fill_dark="#03050A",
        input_border_color="#26313F",
        input_border_color_dark="#26313F",
        input_border_color_focus="#E2231A",
        input_border_color_focus_dark="#E2231A",
        input_placeholder_color="#6E7C8D",
        input_placeholder_color_dark="#6E7C8D",
        input_radius="7px",
        input_shadow="inset 0 1px 0 rgba(0,0,0,.6)",
        input_shadow_focus="0 0 0 4px rgba(226,35,26,.09)",
        input_shadow_focus_dark="0 0 0 4px rgba(226,35,26,.09)",
        button_large_radius="7px",
        button_medium_radius="7px",
        button_small_radius="7px",
        button_primary_background_fill="#E2231A",
        button_primary_background_fill_dark="#E2231A",
        button_primary_background_fill_hover="#F5372E",
        button_primary_background_fill_hover_dark="#F5372E",
        button_primary_border_color="#E2231A",
        button_primary_border_color_dark="#E2231A",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_primary_text_color_hover="#ffffff",
        button_primary_shadow="none",
        button_secondary_background_fill="#131B26",
        button_secondary_background_fill_dark="#131B26",
        button_secondary_background_fill_hover="#1A2331",
        button_secondary_background_fill_hover_dark="#1A2331",
        button_secondary_border_color="#26313F",
        button_secondary_border_color_dark="#26313F",
        button_secondary_text_color="#EAEEF4",
        button_secondary_text_color_dark="#EAEEF4",
        button_cancel_background_fill="#131B26",
        button_cancel_background_fill_dark="#131B26",
        button_cancel_border_color="#26313F",
        button_cancel_text_color="#8B98A9",
        slider_color="#E2231A",
        slider_color_dark="#E2231A",
        loader_color="#E2231A",
        loader_color_dark="#E2231A",
        shadow_drop="0 8px 26px rgba(0,0,0,.5)",
        shadow_drop_lg="0 14px 34px rgba(0,0,0,.5)",
        code_background_fill="#0E141D",
        code_background_fill_dark="#0E141D",
        error_background_fill="#131B26",
        error_background_fill_dark="#131B26",
        error_border_color="#E2231A",
        error_border_color_dark="#E2231A",
        error_text_color="#FFB3AD",
        error_text_color_dark="#FFB3AD",
    )
