# UI Height Optimization Design (Approach 1)

- **Date**: 2026-06-08
- **Topic**: Resolving UI overflow issues in chat and preview windows
- **Status**: Approved

## 1. Problem Statement
The current UI implementation in `src/ui/ui.py` uses CSS `calc` functions with fixed offsets (`12rem` for chat, `10rem` for viewer) and a `min-height` of `400px`. On smaller screens or when the chat input/header takes up more space, these components overflow the viewport, causing the bottom elements (like the chat input or navigation buttons) to be cut off.

## 2. Proposed Solution: Approach 1 (Offset Optimization)
We will increase the vertical offsets and lower the minimum height constraints to ensure all components fit within the dynamic viewport height (`dvh`).

### 2.1 CSS Changes in `src/ui/ui.py`
- **Chat Scroll Container**:
    - Current: `height: calc(100dvh - 12rem) !important;`
    - Target: `height: calc(100dvh - 15rem) !important;`
- **Viewer Scroll Container**:
    - Current: `height: calc(100dvh - 10rem) !important;`
    - Target: `height: calc(100dvh - 13rem) !important;`
- **Min-Height Constraints**:
    - Change `.chat-scroll-container` and `.viewer-scroll-container` from `min-height: 400px` to `min-height: 200px`.
- **Thought Container**:
    - Adjust `max-height` from `300px` to `250px` for better stability during streaming.

### 2.2 Global Constants (`src/common/constants.py`)
- Update `UIConstants.CONTAINER_HEIGHT` if necessary to maintain consistency (though CSS overrides it, keeping it in sync is good practice).
    - Current: `700`
    - Target: `600` (safer default)

## 3. Success Criteria
- The chat input field must be fully visible and accessible on a standard 1080p and 768p screen.
- No vertical scrollbars should appear on the main `stApp` container (overflow should be contained within components).
- The PDF viewer's navigation buttons must remain visible at the bottom of its container.

## 4. Implementation Plan (Summary)
1. Modify `src/ui/ui.py` with updated CSS.
2. Update `src/common/constants.py` for fallback consistency.
3. Verify by running the Streamlit app.
