import asyncio
from playwright.async_api import async_playwright

async def verify_layout():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1280, "height": 720})
        page = await context.new_page()
        
        print("Navigating to app...")
        await page.goto("http://127.0.0.1:8501")
        await page.wait_for_timeout(5000) # Wait for Streamlit to render
        
        results = {}
        
        # 1. Body Overflow Check
        body_scrollTop = await page.evaluate("window.scrollTo(0,0); document.body.scrollTop")
        body_overflow = await page.evaluate("window.getComputedStyle(document.body).overflow")
        results["body"] = {"scrollTop": body_scrollTop, "overflow": body_overflow}
        
        # 2. Container Height Fill Check
        # We need to find the containers. Based on the fix, they are likely the main content areas.
        # Let's look for the elements that are supposed to be scrollable.
        
        # PDF Container: Usually the one with the PDF viewer or the left column
        # Chat Container: Usually the one with the chat history or the right column
        
        # We'll use a JS function to find the containers that have a height and are likely the main ones
        container_data = await page.evaluate("""
            () => {
                const findContainer = (keyword) => {
                    const allDivs = Array.from(document.querySelectorAll('div'));
                    return allDivs.find(div => 
                        (div.innerText && div.innerText.includes(keyword)) || 
                        (div.getAttribute('data-testid') && div.getAttribute('data-testid').includes(keyword))
                    );
                };
                
                // This is a heuristic. In a real app, we'd use specific IDs or data-testids.
                // For this verification, we look for the largest containers in the main area.
                const main = document.querySelector('.stApp');
                if (!main) return null;
                
                const cols = Array.from(main.querySelectorAll('div[data-testid="stColumn"]'));
                if (cols.length < 2) return { error: "Less than 2 columns found" };
                
                const pdfCol = cols[0];
                const chatCol = cols[1];
                
                const getMetrics = (el) => {
                    if (!el) return null;
                    return {
                        scrollHeight: el.scrollHeight,
                        clientHeight: el.clientHeight,
                        offsetHeight: el.offsetHeight,
                        style: window.getComputedStyle(el).overflowY
                    };
                };
                
                return {
                    pdf: getMetrics(pdfCol),
                    chat: getMetrics(chatCol)
                };
            }
        """)
        results["containers"] = container_data
        
        # 3. Chat Input Position Check
        input_metrics = await page.evaluate("""
            () => {
                const input = document.querySelector('.st-key-main_chat_input');
                if (!input) return { error: "Input not found" };
                const rect = input.getBoundingClientRect();
                const style = window.getComputedStyle(input);
                return {
                    bottom: rect.bottom,
                    top: rect.top,
                    left: rect.left,
                    width: rect.width,
                    position: style.position,
                    bottomStyle: style.bottom
                };
            }
        """)
        results["chat_input"] = input_metrics
        
        print("\n--- Verification Results ---")
        print(f"Body: {results['body']}")
        print(f"Containers: {results['containers']}")
        print(f"Chat Input: {results['chat_input']}")
        
        # Final Verdict
        success = True
        evidence = []
        
        if results["body"]["scrollTop"] != 0:
            success = False
            evidence.append(f"FAIL: Body scrollTop is {results['body']['scrollTop']}, expected 0")
        
        if not results["containers"] or "error" in results["containers"]:
            success = False
            evidence.append(f"FAIL: Containers not found: {results['containers']}")
        else:
            pdf = results["containers"]["pdf"]
            chat = results["containers"]["chat"]
            if pdf["scrollHeight"] < pdf["clientHeight"] * 0.9:
                success = False
                evidence.append(f"FAIL: PDF container not filling height (SH: {pdf['scrollHeight']}, CH: {pdf['clientHeight']})")
            if chat["scrollHeight"] < chat["clientHeight"] * 0.9:
                success = False
                evidence.append(f"FAIL: Chat container not filling height (SH: {chat['scrollHeight']}, CH: {chat['clientHeight']})")
        
        if not results["chat_input"] or "error" in results["chat_input"]:
            success = False
            evidence.append(f"FAIL: Chat input not found")
        else:
            ci = results["chat_input"]
            # Check if it's at the bottom of the viewport
            viewport_height = 720
            if ci["bottom"] < viewport_height - 100: # Allow some margin
                success = False
                evidence.append(f"FAIL: Chat input not at bottom (bottom: {ci['bottom']}, viewport: {viewport_height})")
        
        if success:
            evidence.append("SUCCESS: All layout height-fill and overflow criteria met.")
        
        await browser.close()
        return "\n".join(evidence), success

if __name__ == "__main__":
    report, is_success = asyncio.run(verify_layout())
    print("\n" + report)
    exit(0 if is_success else 1)
