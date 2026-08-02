"""
간단한 페이지 상태 덤프 스크립트.
"""

import asyncio
from playwright.async_api import async_playwright


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("http://localhost:8502", timeout=20000)
        await page.wait_for_timeout(5000)

        # 페이지에 에러 메시지가 있는지 확인
        body_text = await page.evaluate("document.body.innerText")
        print("=== BODY TEXT (first 2000 chars) ===")
        print(body_text[:2000])

        # 타이틀 확인
        title = await page.title()
        print(f"\n=== PAGE TITLE: {title} ===")

        # stApp 유무
        stapp = await page.evaluate('document.querySelector(".stApp") !== null')
        print(f"\nstApp exists: {stapp}")

        # stMainBlockContainer 유무
        main_block = await page.evaluate(
            "document.querySelector('[data-testid=\"stMainBlockContainer\"]') !== null"
        )
        print(f"stMainBlockContainer exists: {main_block}")

        if main_block:
            html = await page.evaluate(
                "document.querySelector('[data-testid=\"stMainBlockContainer\"]').outerHTML"
            )
            print(f"\n=== stMainBlockContainer (first 3000 chars) ===")
            print(html[:3000])
        else:
            # count stApp children
            app_children = await page.evaluate(
                'document.querySelector(".stApp") ? document.querySelector(".stApp").children.length : 0'
            )
            print(f"stApp children count: {app_children}")

            # what's inside stApp?
            app_html = await page.evaluate(
                'document.querySelector(".stApp") ? document.querySelector(".stApp").outerHTML.substring(0, 2000) : "NO_STAPP"'
            )
            print(f"\n=== stApp HTML (first 2000 chars) ===")
            print(app_html[:2000])

        await browser.close()


asyncio.run(main())
