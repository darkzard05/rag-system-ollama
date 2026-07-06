# Design Spec: Sidebar & Toolbar UI Restoration (Modern Glassmorphism)

**Date:** 2026-06-07  
**Status:** Approved  
**Author:** UI/UX Expert Agent

## 1. Background
The current UI has an issue where the sidebar expand button becomes invisible when the sidebar is collapsed. Additionally, the top toolbar is completely transparent, making it difficult to distinguish the UI boundaries. This spec defines the restoration of these elements using a modern "Glassmorphism" aesthetic.

## 2. Design Goals
- **Restore Sidebar Visibility:** Ensure the expand/collapse toggle is always visible and interactive.
- **Modern Aesthetic:** Apply a semi-transparent, blurred header (Glassmorphism) instead of total transparency or solid colors.
- **Maintain Openness:** Preserve the "open" feel of the application while improving functional clarity.

## 3. Technical Implementation (CSS)

### 3.1 Top Header & Toolbar
Modify `header[data-testid="stHeader"]` in `src/ui/ui.py`:
- **Background:** `rgba(var(--background-color-rgb), 0.7)` or similar semi-transparent value.
- **Effect:** `backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);`
- **Border:** `border-bottom: 1px solid rgba(var(--faded-text-color-rgb), 0.1);`
- **Shadow:** `box-shadow: 0 2px 10px rgba(0,0,0,0.05);`

### 3.2 Sidebar Toggle Button
Modify `[data-testid="stSidebarCollapseButton"]`:
- **Visibility:** `visibility: visible !important; opacity: 1 !important;`
- **Positioning:** Ensure it is not clipped by `overflow: hidden` on parent containers.
- **Z-Index:** Set to `100000` to ensure it stays on top of the blurred header.

### 3.3 Layout Adjustments
- **Main Container Padding:** Adjust `.block-container` `padding-top` to `3.5rem` to avoid overlap with the restored toolbar.

## 4. Verification Plan
- **Collapsed State:** Verify the ">" expand button is clearly visible against the blurred background.
- **Expanded State:** Verify the "<" collapse button functions normally inside the sidebar.
- **Theme Support:** Ensure the semi-transparent colors adapt correctly to both Light and Dark themes (using Streamlit CSS variables).

## 5. Success Criteria
1. User can collapse and expand the sidebar without the button disappearing.
2. The top area of the app has a distinct, modern blurred boundary.
3. No UI elements (like chat input or PDF viewer) overlap awkwardly with the header.
