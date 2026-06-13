# Design Spec: Independent Column Scrolling and Scoped Chat Input

**Date:** 2026-06-13
**Topic:** Independent Scrolling and Viewport Containment for RAG UI
**Strategy:** Flex-Container Strategy (Modern)

## 1. Goal
Achieve a professional "App-like" feel by ensuring the PDF preview and Chat interface scroll independently within a fixed 100vh viewport. The chat input should feel scoped to the chat column.

## 2. Architecture & CSS Strategy

### 2.1 Viewport Locking
We must prevent the browser's default scrollbar to keep the header and layout stable.
- **Target:** `.stApp` and `.block-container`.
- **Action:** Set `height: 100vh` and `overflow: hidden`.

### 2.2 The Flex Chain
Streamlit wraps columns in several layers of `stVerticalBlock` and `stHorizontalBlock`. For the columns to fill the remaining height, every parent between `.block-container` and `[data-testid="stColumn"]` must be part of the flex layout.
- **Intermediate Blocks:** `[data-testid="stVerticalBlock"]` and `[data-testid="stHorizontalBlock"]`.
- **Key Properties:** `display: flex; flex-direction: column; flex-grow: 1; min-height: 0;`.
- **Why `min-height: 0`?** It allows flex children to shrink smaller than their content, which is necessary to trigger the internal scrollbar of the columns rather than expanding the parent.

### 2.3 Independent Columns
- **Target:** `[data-testid="stColumn"]`.
- **Properties:** `height: 100%; overflow-y: auto; overflow-x: hidden;`.

### 2.4 Scoped Chat Input
Since `st.chat_input` is globally positioned by Streamlit, we will "scope" it via CSS.
- **Target:** `[data-testid="stChatInputContainer"]`.
- **Adjustment:** 
    - `width: 50% !important;` (assuming 1:1 column ratio).
    - `left: 50% !important;` (align to the right column).
    - `background: transparent !important;` (to prevent it from looking like a global footer bar).
- **Padding:** Add `padding-bottom: 100px` to the chat column's inner container to ensure the last message is visible above the input.

## 3. Implementation Plan Overview
1. Update `src/ui/ui.py`'s `inject_custom_css` function with the refined CSS.
2. Test the 1:1 ratio responsiveness (ensure it doesn't break on narrow screens/mobile).
3. Verify that the PDF viewer's internal scroll (if any) doesn't conflict with the column scroll.

## 4. Risks & Mitigations
- **Streamlit Version Updates:** Streamlit often changes data-testids or DOM structure. We use specific selectors like `[data-testid="stHorizontalBlock"]` which are relatively stable in recent versions.
- **Mobile View:** On mobile, `st.columns` usually stacks. We will add a media query to reset `height: auto` and `overflow: visible` on mobile to maintain standard scrolling behavior where needed.

## 5. Success Criteria
- [ ] No browser-level vertical scrollbar visible.
- [ ] PDF column scrolls independently.
- [ ] Chat column scrolls independently.
- [ ] Chat input is positioned only under the chat column.
- [ ] The header stays sticky at the top without jittering.
