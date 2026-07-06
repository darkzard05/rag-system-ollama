def get_components_css() -> str:
    return """
    /* 4. 사이드바 확장 버튼 가시성 확보 */
    [data-testid="stSidebarCollapseButton"] {
        z-index: 100000 !important;
        visibility: visible !important;
        opacity: 1 !important;
        background-color: color-mix(in srgb, var(--background-color), transparent 20%) !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1) !important;
        border-radius: 50% !important;
    }

    .stAppDeployButton {
        display: none !important;
    }

    /* 7. 인용 가독성 및 툴팁 */
    .citation-highlight {
        background-color: color-mix(in srgb, var(--primary-color) 15%, transparent);
        border-bottom: 2px dashed var(--primary-color);
        padding: 0 4px;
        border-radius: 3px;
        color: var(--primary-color);
        font-weight: 600;
        cursor: help;
        transition: background-color 0.2s, transform 0.1s;
        display: inline-block;
        position: relative;
    }
    .citation-highlight:hover::after {
        content: attr(title);
        position: absolute;
        bottom: 100%;
        left: 50%;
        transform: translateX(-50%);
        background-color: var(--text-color);
        color: var(--background-color);
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.85em;
        font-weight: normal;
        white-space: pre-wrap;
        width: max-content;
        max-width: 250px;
        z-index: 1000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        pointer-events: none;
    }

    /* 9. 성능 지표 HUD - 컴팩트 디자인 */
    .perf-details {
        margin-top: var(--spacing-xs) !important;
        padding-top: 0px !important;
    }
    .perf-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: var(--spacing-sm);
        margin-top: var(--spacing-sm);
    }
    .perf-card {
        background-color: var(--secondary-background-color);
        border: 1px solid color-mix(in srgb, var(--faded-text-color) 15%, transparent);
        border-radius: 6px;
        padding: var(--spacing-xs) var(--spacing-sm);
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
    }
    .perf-card-title {
        font-size: 10px;
        color: var(--faded-text-color);
        margin-bottom: var(--spacing-xs);
        font-weight: 500;
    }
    .perf-card-value {
        font-size: 13px;
        font-weight: 700;
        color: var(--text-color);
    }
    .perf-card-desc {
        font-size: 9px;
        color: var(--faded-text-color);
        margin-top: var(--spacing-xs);
    }
    """
