# Design Spec: Adaptive Native Layout (Dynamic Height Containers)

**Date:** 2026-06-13
**Topic:** Native-first Independent Scrolling and Viewport Containment
**Strategy:** Approach 1 (Dynamic Height via JS Eval + st.container)

## 1. Goal
Provide a seamless, app-like experience where the PDF and Chat windows perfectly fit the browser window and scroll independently, using Streamlit's native container height features combined with dynamic viewport detection.

## 2. Technical Strategy

### 2.1 Dynamic Height Detection
- **Tool:** `streamlit-js-eval`
- **Action:** Fetch `window.innerHeight` at runtime.
- **Calculation:** `available_height = inner_height - HEADER_OFFSET` (approx. 120-150px to account for the top bar and margins).

### 2.2 Native Independent Scrolling
- **Implementation:**
    - Use `st.columns([1, 1])` for the main split.
    - Inside each column, wrap content in `st.container(height=available_height, border=False)`.
- **Benefit:** Streamlit's internal engine handles the scrollbars, including features like `autoscroll` for the chat interface.

### 2.3 Global Layout Constraints (Minimal CSS)
- **Viewport Lock:** Apply `overflow: hidden !important;` to `.stApp` and `.stAppViewContainer` to ensure only the containers scroll.
- **Header Stability:** Ensure `stHeader` stays at the top without affecting the container's height calculation.

### 2.4 Chat Input Scoping
- **Target:** `[data-testid="stChatInputContainer"]`.
- **Positioning:** Use fixed positioning with `left: 50%` and `width: 50%` to align it with the right column on desktop.

## 3. Implementation Plan Overview
1.  **Refine UI Structure:** Modify `src/main.py` and `src/ui/ui.py` to incorporate `st.container(height=...)`.
2.  **Integrate Height Detection:** Use `streamlit_js_eval` in the main loop to calculate the target height.
3.  **Update CSS Injection:** Focus the CSS on global containment rather than trying to force the flex chain manually.

## 4. Success Criteria
- [ ] Page-level scrollbar is completely removed.
- [ ] Both columns have independent, visible scrollbars.
- [ ] Content is not cut off at the bottom on different resolutions.
- [ ] Chat input is visible and correctly aligned.
