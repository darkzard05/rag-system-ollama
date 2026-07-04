def get_variables_css() -> str:
    return """
    :root {
        /* Viewport Offset: Accounts for Header (~60px), Tabs (~50px), Chat Input (~80px), and Buffer (~90px) */
        --viewport-offset: 280px;

        /* Design System - Spacing Scale (4px/8px Grid) */
        --spacing-xs: 4px;
        --spacing-sm: 8px;
        --spacing-md: 16px;
        --spacing-lg: 24px;
        --spacing-xl: 32px;
    }
    """
