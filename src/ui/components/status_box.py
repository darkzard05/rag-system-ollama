"""
시스템 상태 박스(Status Box) UI 컴포넌트
"""

import html
import re

from core.session import SessionManager


def render_status_box(container):
    """시스템 상태 로그 박스를 최신순(역순)으로 렌더링합니다."""
    if container is None:
        return

    # [최적화] 세션이 없어도 에러 없이 빈 목록 반환
    try:
        status_logs = SessionManager.get("status_logs", [])
    except Exception:
        status_logs = []

    if not status_logs:
        container.info("시스템 준비 중...")
        return

    # [스타일링: 최신순 출력 전용 테마]
    log_html = """
    <style>
    .status-outer-container {
        border: 1px solid rgba(49, 51, 63, 0.2);
        border-radius: 12px;
        padding: 10px;
        background-color: rgba(128, 128, 128, 0.05);
        margin-bottom: 15px;
        width: 100%;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }
    .status-container {
        font-family: 'Source Code Pro', 'Consolas', monospace;
        height: 140px;
        overflow-y: auto;
        overflow-x: hidden;
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .status-line {
        flex-shrink: 0;
        line-height: 1.5;
        margin: 0px !important;
        padding: 4px 8px !important;
        font-size: 0.8rem;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        color: #666;
        border-left: 2px solid transparent;
        transition: all 0.2s;
    }
    .status-newest {
        color: #0068c9;
        font-weight: 600;
        background-color: rgba(0, 104, 201, 0.1);
        border-radius: 6px;
        border-left: 3px solid #0068c9;
    }

    @media (prefers-color-scheme: dark) {
        .status-outer-container { background-color: rgba(255, 255, 255, 0.05); }
        .status-line { color: #aaa; }
        .status-newest { color: #4fa8ff; background-color: rgba(79, 168, 255, 0.15); border-left-color: #4fa8ff; }
    }

    .status-container::-webkit-scrollbar { width: 4px; }
    .status-container::-webkit-scrollbar-thumb { background: rgba(128, 128, 128, 0.3); border-radius: 10px; }
    </style>
    """

    log_content = ""
    reversed_logs = status_logs[::-1]

    for i, log in enumerate(reversed_logs):
        safe_log = html.escape(log)
        clean_log = re.sub(
            r"[^\x00-\x7F가-힣\s\(\)\[\]\/\:\.\-\>]", "", safe_log
        ).strip()
        if not clean_log and safe_log:
            clean_log = safe_log.strip()

        is_newest = i == 0
        cls = "status-newest" if is_newest else ""
        icon = "●" if is_newest else "○"

        log_content += f"<div class='status-line {cls}' title='{clean_log}'>{icon} {clean_log}</div>"

    full_html = f"{log_html}<div class='status-outer-container'><div class='status-container'>{log_content}</div></div>"
    container.markdown(full_html, unsafe_allow_html=True)
