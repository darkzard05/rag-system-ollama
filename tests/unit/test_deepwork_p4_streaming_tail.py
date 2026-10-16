"""Deepwork P4 — fixed-behavior verification.

Verifies after fix:
 G1: content_buffer_size tail config unchanged (buffered tail is by design).
 G2: mobile 1024/768 scroller blocks now carry overflow-anchor:none (plain).
 G3: probe is observe-only (IntersectionObserver, single writer in ui.py).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAIN_CSS = ROOT / "src" / "ui" / "styles" / "main.css"
PROBE_JS = ROOT / "src" / "ui" / "components" / "chat_autoscroll.py"
CONFIG = ROOT / "config.yml"


def test_p4_content_buffer_config_intact():
    text = CONFIG.read_text(encoding="utf-8")
    assert "content_buffer_size: 4" in text


def test_p4_mobile_scrollers_opt_out_of_anchoring():
    css = MAIN_CSS.read_text(encoding="utf-8")
    norm = css.replace(" ", "")
    idx = norm.find("@media(max-width:1024px)")
    assert idx != -1
    block = norm[idx : idx + 2000]
    assert "overflow-y:auto" in block
    assert "overflow-anchor:none" in block


def test_p4_probe_is_observe_only_single_writer_intact():
    src = PROBE_JS.read_text(encoding="utf-8")
    assert "insertBefore" in src
    assert "IntersectionObserver" in src
    assert "MutationObserver" not in src
    ui_src = (ROOT / "src" / "ui" / "ui.py").read_text(encoding="utf-8")
    assert "MutationObserver" in ui_src
