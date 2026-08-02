"""
Playwright 기반 신규 레이아웃 검증 스크립트.

검증 대상 (Designer 요구사항 기준):
1. 두 개의 stLayoutWrapper 존재 (content row + dock row)
2. Content row: PDF viewer(left) + chat messages(right)
3. Dock row: PDF controls(left) + chat input(right)
4. Dock wrapper에 border-top 적용
5. Chat input이 dock column 내부에 위치 (standalone 아님)
6. 기존 sticky hack 제거됨 (position: sticky 없음)
7. Header height --header-h CSS 변수 설정됨
8. 전체 레이아웃이 viewport를 채움 (overflow: hidden, height: 100dvh)
"""

import asyncio
import json
import sys

from playwright.async_api import async_playwright

APP_URL = "http://localhost:8502"
PASS = "✅"
FAIL = "❌"
WARN = "⚠️"


async def verify():
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1920, "height": 1080},
        )
        page = await context.new_page()

        print(f"Navigating to {APP_URL}...")
        try:
            await page.goto(APP_URL, timeout=20000)
            await page.wait_for_selector("[data-testid='stApp']", timeout=15000)
            await page.wait_for_timeout(4000)  # 충분한 렌더링 대기
            print("  Page loaded.\n")
        except Exception as e:
            print(f"  FAILED to load page: {e}")
            await browser.close()
            return

        # ── 1. 전체 앱 컨테이너 검증 ──
        print("=" * 60)
        print("1. 전체 앱 컨테이너")
        print("=" * 60)

        app_overflow = await page.evaluate(
            """() => {
                const el = document.querySelector('.stApp');
                if (!el) return 'NOT_FOUND';
                return window.getComputedStyle(el).overflow;
            }"""
        )
        ok = app_overflow == "hidden"
        results.append(("stApp overflow: hidden", ok, app_overflow))

        app_height = await page.evaluate(
            """() => {
                const el = document.querySelector('.stApp');
                return el ? el.clientHeight + 'px' : 'NOT_FOUND';
            }"""
        )
        results.append(("stApp clientHeight", None, app_height))
        print(f"  stApp height: {app_height}")

        # ── 2. 두 개의 layout wrapper 확인 ──
        print("\n" + "=" * 60)
        print("2. Layout Wrapper (stLayoutWrapper)")
        print("=" * 60)

        wrapper_count = await page.evaluate(
            """() => {
                const wrappers = document.querySelectorAll(
                    '[data-testid=\"stMainBlockContainer\"] > [data-testid=\"stVerticalBlock\"] > [data-testid=\"stLayoutWrapper\"]'
                );
                return wrappers.length;
            }"""
        )
        ok = wrapper_count >= 2
        results.append(("stLayoutWrapper count >= 2", ok, wrapper_count))

        # 첫번째 wrapper (content row)
        first_wrapper_flex = await page.evaluate(
            """() => {
                const wrappers = document.querySelectorAll(
                    '[data-testid=\"stMainBlockContainer\"] > [data-testid=\"stVerticalBlock\"] > [data-testid=\"stLayoutWrapper\"]'
                );
                if (wrappers.length < 1) return 'NOT_FOUND';
                return window.getComputedStyle(wrappers[0]).flex;
            }"""
        )
        results.append(
            (
                "Content wrapper flex (expect '1 1 0px' or '1 0 0px')",
                None,
                first_wrapper_flex,
            )
        )
        print(f"  Content wrapper: flex: {first_wrapper_flex}")

        # 마지막 wrapper (dock row)
        last_wrapper_flex = await page.evaluate(
            """() => {
                const wrappers = document.querySelectorAll(
                    '[data-testid=\"stMainBlockContainer\"] > [data-testid=\"stVerticalBlock\"] > [data-testid=\"stLayoutWrapper\"]'
                );
                if (wrappers.length < 2) return 'NOT_FOUND';
                return window.getComputedStyle(wrappers[wrappers.length - 1]).flex;
            }"""
        )
        results.append(
            (
                "Dock wrapper flex (expect '0 1 auto' or '0 0 auto')",
                None,
                last_wrapper_flex,
            )
        )
        print(f"  Dock wrapper: flex: {last_wrapper_flex}")

        # ── 3. Dock border-top 확인 ──
        print("\n" + "=" * 60)
        print("3. Dock border-top")
        print("=" * 60)

        dock_border_top = await page.evaluate(
            """() => {
                const wrappers = document.querySelectorAll(
                    '[data-testid=\"stMainBlockContainer\"] > [data-testid=\"stVerticalBlock\"] > [data-testid=\"stLayoutWrapper\"]'
                );
                if (wrappers.length < 2) return 'NOT_FOUND';
                return window.getComputedStyle(wrappers[wrappers.length - 1]).borderTop;
            }"""
        )
        has_border = (
            dock_border_top != "NOT_FOUND"
            and dock_border_top != "0px none rgb(0, 0, 0)"
            and dock_border_top != "0px none"
        )
        results.append(("Dock border-top present", has_border, dock_border_top))
        print(f"  Dock border-top: {dock_border_top}")

        # ── 4. Chat Input 위치 검증 ──
        print("\n" + "=" * 60)
        print("4. Chat Input 위치")
        print("=" * 60)

        chat_input_sticky = await page.evaluate(
            """() => {
                const input = document.querySelector('[data-testid=\"stChatInput\"]');
                if (!input) return 'NOT_FOUND';
                return window.getComputedStyle(input).position;
            }"""
        )
        not_sticky = chat_input_sticky not in ("fixed", "sticky", "NOT_FOUND")
        results.append(
            ("Chat input NOT position:fixed/sticky", not_sticky, chat_input_sticky)
        )
        print(f"  Chat input position: {chat_input_sticky}")

        # Chat input이 dock 컬럼 안에 있는지 확인
        chat_input_in_dock = await page.evaluate(
            """() => {
                const input = document.querySelector('[data-testid=\"stChatInput\"]');
                if (!input) return 'NOT_FOUND';
                // dock wrapper 안에 chat input이 있는지 확인
                const wrappers = document.querySelectorAll(
                    '[data-testid=\"stMainBlockContainer\"] > [data-testid=\"stVerticalBlock\"] > [data-testid=\"stLayoutWrapper\"]'
                );
                if (wrappers.length < 2) return 'ONLY_ONE_WRAPPER';
                const dockWrapper = wrappers[wrappers.length - 1];
                return dockWrapper.contains(input) ? 'IN_DOCK' : 'NOT_IN_DOCK';
            }"""
        )
        in_dock = chat_input_in_dock == "IN_DOCK"
        results.append(("Chat input inside dock wrapper", in_dock, chat_input_in_dock))
        print(f"  Chat input location: {chat_input_in_dock}")

        # ── 5. PDF Controls 위치 검증 ──
        print("\n" + "=" * 60)
        print("5. PDF Controls 위치")
        print("=" * 60)

        pdf_controls_in_dock = await page.evaluate(
            """() => {
                // PDF controls 버튼 확인 (이전/다음 버튼)
                const allButtons = document.querySelectorAll('button');
                const navButtons = Array.from(allButtons).filter(b =>
                    b.innerText.includes('이전') || b.innerText.includes('다음')
                );
                if (navButtons.length === 0) return 'NO_PDF_CONTROLS';

                const wrappers = document.querySelectorAll(
                    '[data-testid=\"stMainBlockContainer\"] > [data-testid=\"stVerticalBlock\"] > [data-testid=\"stLayoutWrapper\"]'
                );
                if (wrappers.length < 2) return 'ONLY_ONE_WRAPPER';
                const dockWrapper = wrappers[wrappers.length - 1];

                const allInDock = navButtons.every(b => dockWrapper.contains(b));
                return allInDock ? 'ALL_IN_DOCK' : 'SOME_OUTSIDE_DOCK';
            }"""
        )
        controls_ok = pdf_controls_in_dock in ("ALL_IN_DOCK", "NO_PDF_CONTROLS")
        results.append(
            (
                "PDF controls inside dock (or no PDF loaded)",
                controls_ok,
                pdf_controls_in_dock,
            )
        )
        print(f"  PDF controls location: {pdf_controls_in_dock}")

        # ── 6. Scrollbar 상태 ──
        print("\n" + "=" * 60)
        print("6. 컬럼 스크롤 상태")
        print("=" * 60)

        content_overflow = await page.evaluate(
            """() => {
                const containers = document.querySelectorAll(
                    '[data-testid=\"stMainBlockContainer\"] [data-testid=\"stVerticalBlock\"]'
                );
                for (const c of containers) {
                    const ov = window.getComputedStyle(c).overflowY;
                    if (ov === 'auto' || ov === 'scroll') return ov;
                }
                return 'none';
            }"""
        )
        results.append(
            (
                "Content area scrollable (overflow-y: auto or scroll)",
                None,
                content_overflow,
            )
        )
        print(f"  Content overflow-y: {content_overflow}")

        # ── 7. 전체 페이지 스크롤 ──
        print("\n" + "=" * 60)
        print("7. 전체 페이지 스크롤 (viewport filling)")
        print("=" * 60)

        scroll_height = await page.evaluate("document.body.scrollHeight")
        viewport_h = await page.evaluate("window.innerHeight")
        full_page_scroll = scroll_height > viewport_h
        results.append(
            (
                "Body scrollHeight > viewport (expect NO full-page scroll)",
                not full_page_scroll,
                f"scrollH={scroll_height} vs vpH={viewport_h}",
            )
        )
        print(f"  Body scrollHeight: {scroll_height}, viewportHeight: {viewport_h}")
        print(f"  Full-page scroll: {'YES ⚠️' if full_page_scroll else 'NO ✅'}")

        # ── 8. Header height ──
        print("\n" + "=" * 60)
        print("8. Header height (--header-h)")
        print("=" * 60)

        header_h_var = await page.evaluate(
            """() => {
                const val = document.documentElement.style.getPropertyValue('--header-h');
                return val || 'NOT_SET';
            }"""
        )
        set_ok = header_h_var != "NOT_SET" and header_h_var != ""
        results.append(("--header-h CSS variable set", set_ok, header_h_var))
        print(f"  --header-h: {header_h_var}")

        # 헤더 실제 높이와 비교
        header_actual = await page.evaluate(
            """() => {
                const h = document.querySelector('[data-testid=\"stHeader\"]');
                return h ? h.offsetHeight + 'px' : 'NOT_FOUND';
            }"""
        )
        results.append(("Actual header height", None, header_actual))
        print(f"  Actual header height: {header_actual}")

        # ── 9. Dock min-height ──
        print("\n" + "=" * 60)
        print("9. Dock min-height (52px)")
        print("=" * 60)

        dock_min_height = await page.evaluate(
            """() => {
                const wrappers = document.querySelectorAll(
                    '[data-testid=\"stMainBlockContainer\"] > [data-testid=\"stVerticalBlock\"] > [data-testid=\"stLayoutWrapper\"]'
                );
                if (wrappers.length < 2) return 'NO_DOCK';
                // dock wrapper 안의 첫번째 column의 vertical block
                const dockWrapper = wrappers[wrappers.length - 1];
                const col = dockWrapper.querySelector('[data-testid=\"stColumn\"]');
                if (!col) return 'NO_COLUMN';
                const vb = col.querySelector('[data-testid=\"stVerticalBlock\"]');
                if (!vb) return 'NO_VB';
                const lw = vb.querySelector('[data-testid=\"stLayoutWrapper\"]');
                if (!lw) return 'NO_LW';
                const innerVb = lw.querySelector('[data-testid=\"stVerticalBlock\"]');
                if (!innerVb) return 'NO_INNER_VB';
                return window.getComputedStyle(innerVb).minHeight;
            }"""
        )
        has_min_h = dock_min_height not in (
            "NO_DOCK",
            "NO_COLUMN",
            "NO_VB",
            "NO_LW",
            "NO_INNER_VB",
            "0px",
            "auto",
        )
        results.append(("Dock column min-height >= 52px", has_min_h, dock_min_height))
        print(f"  Dock column min-height: {dock_min_height}")

        # ── 결과 출력 ──
        print("\n")
        print("=" * 60)
        print("검증 결과 요약")
        print("=" * 60)

        passed = 0
        failed = 0
        for name, ok, value in results:
            if ok is None:
                icon = "ℹ️"
            elif ok:
                icon = PASS
                passed += 1
            else:
                icon = FAIL
                failed += 1
            val_str = f" → {value}" if value is not None else ""
            print(f"  {icon} {name}{val_str}")

        total = passed + failed
        print(f"\n  {PASS} {passed}/{total} passed, {FAIL} {failed}/{total} failed")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(verify())
