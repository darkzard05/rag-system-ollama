def get_layout_css() -> str:
    return """
    /* [최적화] 메인 페이지 상하좌우 여백 최소화 및 바디 스크롤 방지 */
    .stApp, [data-testid="stAppViewContainer"] {
        overflow: hidden !important;
    }

    .block-container {
        padding: var(--spacing-xs) 0 !important;
        max-width: 100%;
        overflow: hidden !important;
    }

    /* 5. 상단 헤더 Glassmorphism */
    header[data-testid="stHeader"] {
        background: color-mix(in srgb, var(--background-color), transparent 30%) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-bottom: 1px solid color-mix(in srgb, var(--faded-text-color), transparent 80%) !important;
        box-shadow: 0 2px 15px rgba(0,0,0,0.05) !important;
        z-index: 99999 !important;
        display: flex;
        visibility: visible;
    }

    /* 10. 참조 페이지 버튼 반응형 랩핑 시스템 */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: var(--spacing-sm) !important;
    }
    div[data-testid="column"] {
        min-width: 60px !important;
        flex: 0 1 auto !important;
        margin: 0 !important;
    }
    div[data-testid="column"] button {
        border-radius: 20px !important;
        padding: var(--spacing-xs) var(--spacing-md) !important;
        font-size: 12px !important;
        font-weight: 600 !important;
        background-color: var(--secondary-background-color) !important;
        color: var(--text-color) !important;
        border: 1px solid color-mix(in srgb, var(--faded-text-color) 20%, transparent) !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
    }
    div[data-testid="column"] button:hover {
        border-color: var(--primary-color) !important;
        color: var(--primary-color) !important;
        background-color: color-mix(in srgb, var(--primary-color) 10%, transparent) !important;
    }

    /* 12. Mobile Responsiveness */
    @media (max-width: 768px) {
        .stApp, [data-testid="stAppViewContainer"] {
            height: auto;
            overflow: visible;
        }
        /* Revert containers to auto height on mobile */
        div[data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
            height: auto;
            max-height: none !important;
        }
        [data-testid="stChatInputContainer"] {
            left: 0 !important;
            width: 100% !important;
            position: fixed !important;
        }
    }

    /* 12b. Chat Input Container — Desktop fixed positioning (right column) */
    [data-testid="stChatInputContainer"] {
        position: fixed !important;
        bottom: 0;
        right: 0;
        left: 50%;
        width: 50%;
        background: var(--background-color);
        border-top: 1px solid color-mix(in srgb, var(--faded-text-color), transparent 80%);
        z-index: 100 !important;
    }

    /* 13. Responsive Container Heights for PDF Viewer and Chat columns */
    div[data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
        height: auto !important;
        max-height: calc(100vh - var(--viewport-offset));
        max-height: calc(100dvh - var(--viewport-offset));
        overflow-y: auto !important;
    }
    """
