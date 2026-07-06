import asyncio
from playwright.async_api import async_playwright

async def verify_layout():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()

        print("Navigating to http://localhost:8501...")
        try:
            await page.goto("http://localhost:8501", timeout=15000)
            # 핵심 요소가 나타날 때까지 대기
            await page.wait_for_selector('[data-testid="stApp"]', timeout=10000)
            await page.wait_for_selector('[data-testid="stChatInput"]', timeout=10000)
            await page.wait_for_timeout(2000)
        except Exception as e:
            print(f"Error: Page load or element wait failed: {e}")
            content = await page.content()
            print(f"Page content snippet: {content[:1000]}")
            await browser.close()
            return

        print("\n--- Layout Verification ---")

        body_scrollTop = await page.evaluate("window.scrollTo(0,0); document.body.scrollTop")
        body_overflow = await page.evaluate("window.getComputedStyle(document.body).overflow")
        print(f"Body scrollTop: {body_scrollTop} (Expected: 0)")
        print(f"Body overflow style: {body_overflow}")

        metrics = await page.evaluate("""
            () => {
                const findContainer = (predicate) => {
                    const allDivs = Array.from(document.querySelectorAll('div'));
                    return allDivs.find(predicate);
                };

                const pdfContainer = findContainer(el => 
                    (el.innerText.includes('PDF') || el.querySelector('iframe')) && el.clientHeight > 100
                ) || document.querySelector('[data-testid="stColumn"]:first-child [data-testid="stVerticalBlock"]');

                const chatContainer = findContainer(el => 
                    (el.innerText.includes('Chat') || el.querySelector('[data-testid="stChatInput"]')) && el.clientHeight > 100
                ) || document.querySelector('[data-testid="stColumn"]:last-child [data-testid="stVerticalBlock"]');

                const getMetrics = (el) => el ? {
                    scrollHeight: el.scrollHeight,
                    clientHeight: el.clientHeight,
                    ratio: el.scrollHeight / el.clientHeight
                } : null;

                return {
                    pdf: getMetrics(pdfContainer),
                    chat: getMetrics(chatContainer)
                };
            }
        """)

        if metrics['pdf']:
            p = metrics['pdf']
            print(f"PDF Container -> sh: {p['scrollHeight']}, ch: {p['clientHeight']}, ratio: {p['ratio']:.2f}")
            print(f"PDF Height Fill: {'✅' if p['ratio'] >= 0.9 else '❌'}")
        else:
            print("PDF Container: ❌ NOT FOUND")

        if metrics['chat']:
            c = metrics['chat']
            print(f"Chat Container -> sh: {c['scrollHeight']}, ch: {c['clientHeight']}, ratio: {c['ratio']:.2f}")
            print(f"Chat Height Fill: {'✅' if c['ratio'] >= 0.9 else '❌'}")
        else:
            print("Chat Container: ❌ NOT FOUND")

        input_metrics = await page.evaluate("""
            () => {
                const input = document.querySelector('[data-testid="stChatInput"]');
                if (!input) return null;
                const rect = input.getBoundingClientRect();
                return {
                    bottom: rect.bottom,
                    left: rect.left,
                    width: rect.width,
                    viewportWidth: window.innerWidth,
                    viewportHeight: window.innerHeight
                };
            }
        """)

        if input_metrics:
            im = input_metrics
            bottom_ok = abs(im['bottom'] - im['viewportHeight']) < 10
            expected_left = (im['viewportWidth'] - im['width']) / 2
            center_ok = abs(im['left'] - expected_left) < 50
            
            print(f"Chat Input -> bottom: {im['bottom']}, viewportHeight: {im['viewportHeight']}")
            print(f"Chat Input -> left: {im['left']}, expectedLeft: {expected_left:.2f}")
            print(f"Input Bottom Fixed: {'✅' if bottom_ok else '❌'}")
            print(f"Input Centered: {'✅' if center_ok else '❌'}")
        else:
            print("Chat Input: ❌ NOT FOUND ([data-testid='stChatInput'])")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(verify_layout())
