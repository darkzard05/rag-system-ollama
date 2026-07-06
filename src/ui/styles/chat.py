def get_chat_css() -> str:
    return """
    /* 11. 채팅 메시지 내부 수직 유격 압축 */
    [data-testid="stChatMessage"] div[data-testid="stVerticalBlock"] {
        gap: 0px !important; /* [수정] 내부 블록 간격 완벽 제거 */
    }

    /* [추가] 채팅 메시지 자체의 패딩 축소 */
    [data-testid="stChatMessage"] {
        padding-top: var(--spacing-sm) !important;
        padding-bottom: var(--spacing-sm) !important;
    }

    /* 6. 사고 과정(Thought Process) UI - Layout Shift 방지 및 여백 압축 */
    details.thought-expander {
        background-color: var(--secondary-background-color);
        border: 1px solid color-mix(in srgb, var(--faded-text-color) 30%, transparent);
        border-radius: 8px;
        /* [수정] 표준 스페이싱 적용 및 텍스트 가독성을 위한 행간 확보 */
        margin: 0px 0 var(--spacing-xs) 0;
        padding: var(--spacing-sm) var(--spacing-md);
        line-height: 1.6;
        transition: all 0.2s ease-in-out;
    }
    details.thought-expander summary {
        cursor: pointer;
        font-size: 0.85em;
        font-weight: 600;
        color: var(--text-color);
        opacity: 0.7;
        outline: none;
        list-style: none;
        display: flex;
        align-items: center;
        gap: var(--spacing-sm);
        user-select: none;
    }
    details.thought-expander summary::before {
        content: "▶";
        font-size: 0.8em;
        transition: transform 0.2s;
    }
    details[open].thought-expander summary::before {
        transform: rotate(90deg);
    }
    .thought-container {
        border-left: 3px solid color-mix(in srgb, var(--primary-color) 70%, transparent);
        padding: var(--spacing-sm) var(--spacing-md);
        margin-top: var(--spacing-sm); /* [수정] 표준 스페이싱 적용 */
        font-size: 0.85em;
        color: var(--text-color);
        opacity: 0.85;
        line-height: 1.6;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        white-space: pre-wrap;
        max-height: 300px;
        overflow-y: auto;
        background-color: color-mix(in srgb, var(--background-color) 50%, transparent);
        border-radius: 0 4px 4px 0;
    }

    /* 8. 스트리밍 애니메이션 (빌드 상태 표시에도 공용 사용) */
    .streaming-pulse {
        animation: pulse 1.5s infinite ease-in-out;
        min-height: 24px;
        font-size: 0.9em;
        color: var(--primary-color);
        font-weight: 500;
        /* [수정] 표준 스페이싱 적용 */
        margin-bottom: var(--spacing-xs);
    }
    @keyframes pulse {
        0% { opacity: 0.4; }
        50% { opacity: 1; }
        100% { opacity: 0.4; }
    }
    """
