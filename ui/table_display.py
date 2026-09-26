"""Shared dataframe rendering — pinch-zoom sheets on phone, normal tables on desktop."""

from __future__ import annotations

import html as html_lib
import re
from urllib.parse import unquote_plus

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

SPREADSHEET_TABLES_KEY = "spreadsheet_tables"
_CSS_INJECTED_KEY = "_spreadsheet_css_injected"

_MOBILE_UA_MARKERS = (
    "mobile",
    "iphone",
    "ipod",
    "android",
    "ipad",
    "webos",
    "blackberry",
    "opera mini",
)

_LINK_CELL_RE = re.compile(
    r"(<t[dh][^>]*>)([^<]*\?player=[^<]+)(</t[dh]>)",
    re.IGNORECASE,
)


def is_mobile_client() -> bool:
    """Best-effort phone/tablet detection from the request User-Agent."""
    try:
        headers = st.context.headers
    except Exception:
        return False

    ua = ""
    if headers is not None:
        ua = str(
            headers.get("User-Agent")
            or headers.get("user-agent")
            or ""
        ).lower()
    return any(marker in ua for marker in _MOBILE_UA_MARKERS)


def spreadsheet_mode_enabled() -> bool:
    return bool(st.session_state.get(SPREADSHEET_TABLES_KEY, True))


def ensure_spreadsheet_default() -> None:
    """Pinch-zoom sheets on by default (used on phone only)."""
    if SPREADSHEET_TABLES_KEY not in st.session_state:
        st.session_state[SPREADSHEET_TABLES_KEY] = True


def render_spreadsheet_toggle() -> None:
    """Sidebar control — phone only uses pinch-zoom sheets when on."""
    ensure_spreadsheet_default()
    st.sidebar.toggle(
        "Pinch-zoom tables (phone)",
        key=SPREADSHEET_TABLES_KEY,
        help=(
            "On phone: render tables like a Google Sheet — pinch to zoom "
            "the whole table in/out, drag to pan, Fit to see everything. "
            "Desktop always uses normal Streamlit tables. Turn off to use "
            "the regular table on phone too."
        ),
    )


def inject_spreadsheet_css() -> None:
    """Tighter mobile padding so sheets get more room."""
    if st.session_state.get(_CSS_INJECTED_KEY):
        return
    st.session_state[_CSS_INJECTED_KEY] = True

    st.html(
        """
<style>
  @media (max-width: 768px) {
    [data-testid="stMainBlockContainer"] {
      padding-left: 0.5rem !important;
      padding-right: 0.5rem !important;
      padding-top: 0.85rem !important;
    }
    [data-testid="stHeadingWithActionElements"] h1 {
      font-size: 1.45rem !important;
    }
  }
</style>
"""
    )


def _cfg_label(cfg) -> str | None:
    if cfg is None or not isinstance(cfg, dict):
        return None
    label = cfg.get("label")
    return str(label) if label else None


def _hidden_columns(column_config) -> set[str]:
    if not column_config:
        return set()
    return {col for col, cfg in column_config.items() if cfg is None}


def _link_display_and_href(raw: str) -> tuple[str, str]:
    text = html_lib.unescape(raw).strip()
    if "#" in text:
        href, fragment = text.split("#", 1)
        display = unquote_plus(fragment)
    else:
        href, display = text, text
    if href.startswith("/?"):
        # Stay on same origin inside the component iframe parent.
        href = href
    return display, href


def _replace_link_cells(table_html: str) -> str:
    def _repl(match: re.Match) -> str:
        open_tag, raw, close_tag = match.group(1), match.group(2), match.group(3)
        display, href = _link_display_and_href(raw)
        return (
            f'{open_tag}<a href="{html_lib.escape(href, quote=True)}" '
            f'target="_parent" rel="noopener">'
            f"{html_lib.escape(display)}</a>{close_tag}"
        )

    return _LINK_CELL_RE.sub(_repl, table_html)


def _relabel_header_cells(table_html: str, column_config) -> str:
    if not column_config:
        return table_html
    for col, cfg in column_config.items():
        label = _cfg_label(cfg)
        if not label:
            continue
        table_html = table_html.replace(
            f"<th>{html_lib.escape(str(col))}</th>",
            f"<th>{html_lib.escape(label)}</th>",
        )
        # pandas sometimes adds classes on th
        table_html = re.sub(
            rf"(<th[^>]*>){re.escape(html_lib.escape(str(col)))}(</th>)",
            rf"\1{html_lib.escape(label)}\2",
            table_html,
        )
    return table_html


def _build_table_html(data, column_config, hide_index: bool) -> str:
    hidden = _hidden_columns(column_config)

    if hasattr(data, "data") and hasattr(data, "to_html") and not isinstance(data, pd.DataFrame):
        drop = [c for c in hidden if c in data.data.columns]
        if drop:
            try:
                table_html = data.hide(axis="columns", subset=drop).to_html()
            except Exception:
                visible = [c for c in data.data.columns if c not in hidden]
                table_html = data.data[visible].to_html(
                    index=not hide_index, border=0, classes="sheet"
                )
        else:
            table_html = data.to_html()
    else:
        frame = data
        if isinstance(frame, pd.DataFrame) and hidden:
            frame = frame.drop(
                columns=[c for c in hidden if c in frame.columns],
                errors="ignore",
            )
        table_html = frame.to_html(index=not hide_index, border=0, classes="sheet")

    table_html = _replace_link_cells(table_html)
    table_html = _relabel_header_cells(table_html, column_config)
    return table_html


def _pinch_sheet_document(table_html: str, viewport_h: int) -> str:
    # Escape closing script tags in table content just in case.
    safe_table = table_html.replace("</script>", "<\\/script>")
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />
<style>
  html, body {{
    margin: 0;
    padding: 0;
    background: #f8f9fa;
    color: #202124;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    overflow: hidden;
    height: 100%;
  }}
  .bar {{
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.35rem 0.45rem;
    background: #fff;
    border-bottom: 1px solid #dadce0;
    font-size: 12px;
    user-select: none;
    -webkit-user-select: none;
  }}
  .bar button {{
    border: 1px solid #dadce0;
    background: #fff;
    color: #202124;
    border-radius: 6px;
    padding: 0.28rem 0.55rem;
    font-size: 13px;
    font-weight: 600;
    min-width: 2.1rem;
  }}
  .bar button:active {{ background: #e8f0fe; }}
  .bar .hint {{
    margin-left: auto;
    color: #5f6368;
    font-size: 11px;
  }}
  .bar #pct {{
    min-width: 3.2rem;
    text-align: center;
    color: #5f6368;
    font-variant-numeric: tabular-nums;
  }}
  .vp {{
    position: relative;
    height: {int(viewport_h)}px;
    overflow: hidden;
    touch-action: none;
    background: #f8f9fa;
    cursor: grab;
  }}
  .canvas {{
    transform-origin: 0 0;
    will-change: transform;
    display: inline-block;
  }}
  table {{
    border-collapse: collapse;
    font-size: 12px;
    white-space: nowrap;
    background: #fff;
  }}
  th, td {{
    border: 1px solid #e0e0e0;
    padding: 4px 8px;
    text-align: left;
  }}
  th {{
    background: #f3f3f3;
    position: sticky;
    top: 0;
    z-index: 1;
    font-weight: 650;
  }}
  a {{ color: #1a73e8; text-decoration: none; }}
  a:active {{ color: #174ea6; }}
</style>
</head>
<body>
  <div class="bar">
    <button type="button" id="fit" title="Fit whole table">Fit</button>
    <button type="button" id="out" title="Zoom out">−</button>
    <span id="pct">100%</span>
    <button type="button" id="in" title="Zoom in">+</button>
    <span class="hint">pinch · drag</span>
  </div>
  <div class="vp" id="vp">
    <div class="canvas" id="canvas">{safe_table}</div>
  </div>
<script>
(function () {{
  const vp = document.getElementById("vp");
  const canvas = document.getElementById("canvas");
  const pctEl = document.getElementById("pct");
  const MIN = 0.15;
  const MAX = 3.0;
  let scale = 1;
  let tx = 0;
  let ty = 0;
  let pointers = new Map();
  let pinchStartDist = 0;
  let pinchStartScale = 1;
  let pinchOriginTx = 0;
  let pinchOriginTy = 0;
  let pinchCx = 0;
  let pinchCy = 0;
  let panStartX = 0;
  let panStartY = 0;
  let panOriginX = 0;
  let panOriginY = 0;
  let lastTap = 0;

  function apply() {{
    canvas.style.transform = "translate(" + tx + "px," + ty + "px) scale(" + scale + ")";
    pctEl.textContent = Math.round(scale * 100) + "%";
  }}

  function fit() {{
    const vw = vp.clientWidth;
    const vh = vp.clientHeight;
    const tw = Math.max(canvas.scrollWidth, 1);
    const th = Math.max(canvas.scrollHeight, 1);
    const next = Math.max(MIN, Math.min(MAX, Math.min(vw / tw, vh / th) * 0.98));
    scale = next;
    tx = Math.max(0, (vw - tw * scale) / 2);
    ty = Math.max(0, (vh - th * scale) / 2);
    apply();
  }}

  function zoomAt(factor, cx, cy) {{
    const prev = scale;
    const next = Math.max(MIN, Math.min(MAX, prev * factor));
    if (next === prev) return;
    tx = cx - (cx - tx) * (next / prev);
    ty = cy - (cy - ty) * (next / prev);
    scale = next;
    apply();
  }}

  function dist(a, b) {{
    const dx = a.x - b.x;
    const dy = a.y - b.y;
    return Math.hypot(dx, dy);
  }}

  vp.addEventListener("pointerdown", (e) => {{
    vp.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, {{ x: e.clientX, y: e.clientY }});
    if (pointers.size === 1) {{
      panStartX = e.clientX;
      panStartY = e.clientY;
      panOriginX = tx;
      panOriginY = ty;
      const now = Date.now();
      if (now - lastTap < 280) {{
        fit();
        lastTap = 0;
      }} else {{
        lastTap = now;
      }}
    }} else if (pointers.size === 2) {{
      const pts = Array.from(pointers.values());
      const rect = vp.getBoundingClientRect();
      pinchStartDist = dist(pts[0], pts[1]) || 1;
      pinchStartScale = scale;
      pinchOriginTx = tx;
      pinchOriginTy = ty;
      pinchCx = (pts[0].x + pts[1].x) / 2 - rect.left;
      pinchCy = (pts[0].y + pts[1].y) / 2 - rect.top;
    }}
  }});

  vp.addEventListener("pointermove", (e) => {{
    if (!pointers.has(e.pointerId)) return;
    pointers.set(e.pointerId, {{ x: e.clientX, y: e.clientY }});
    if (pointers.size === 2) {{
      const pts = Array.from(pointers.values());
      const d = dist(pts[0], pts[1]) || 1;
      const next = Math.max(MIN, Math.min(MAX, pinchStartScale * (d / pinchStartDist)));
      const ratio = next / pinchStartScale;
      tx = pinchCx - (pinchCx - pinchOriginTx) * ratio;
      ty = pinchCy - (pinchCy - pinchOriginTy) * ratio;
      scale = next;
      apply();
    }} else if (pointers.size === 1) {{
      tx = panOriginX + (e.clientX - panStartX);
      ty = panOriginY + (e.clientY - panStartY);
      apply();
    }}
  }});

  function endPointer(e) {{
    pointers.delete(e.pointerId);
    if (pointers.size === 1) {{
      const pt = Array.from(pointers.values())[0];
      panStartX = pt.x;
      panStartY = pt.y;
      panOriginX = tx;
      panOriginY = ty;
    }}
  }}
  vp.addEventListener("pointerup", endPointer);
  vp.addEventListener("pointercancel", endPointer);

  document.getElementById("fit").onclick = fit;
  document.getElementById("in").onclick = () => zoomAt(1.2, vp.clientWidth / 2, vp.clientHeight / 2);
  document.getElementById("out").onclick = () => zoomAt(1 / 1.2, vp.clientWidth / 2, vp.clientHeight / 2);

  // Start fitted so the whole board is visible like Sheets zoomed out.
  requestAnimationFrame(() => requestAnimationFrame(fit));
  window.addEventListener("resize", fit);
}})();
</script>
</body>
</html>"""


def _show_pinch_sheet(data, *, height, column_config, hide_index: bool) -> None:
    viewport_h = int(height or 560)
    viewport_h = max(280, min(viewport_h, 720))
    table_html = _build_table_html(data, column_config, hide_index)
    components.html(
        _pinch_sheet_document(table_html, viewport_h),
        height=viewport_h + 42,
        scrolling=False,
    )


def show_dataframe(
    data,
    *,
    height=None,
    column_config=None,
    hide_index: bool = True,
    key: str | None = None,
    row_height: int | None = None,
    width=None,
):
    """
    Phone + toggle on: pinch-zoom Google Sheet-style HTML table.
    Desktop (or toggle off): normal ``st.dataframe``.
    """
    ensure_spreadsheet_default()
    mobile = is_mobile_client()
    sheet = spreadsheet_mode_enabled()

    if mobile and sheet:
        _show_pinch_sheet(
            data,
            height=height,
            column_config=column_config,
            hide_index=hide_index,
        )
        return None

    kwargs: dict = {
        "hide_index": hide_index,
        "width": width if width is not None else "stretch",
    }
    if column_config is not None:
        kwargs["column_config"] = column_config
    if key is not None:
        kwargs["key"] = key
    if height is not None:
        kwargs["height"] = height
    if row_height is not None:
        kwargs["row_height"] = row_height

    return st.dataframe(data, **kwargs)
