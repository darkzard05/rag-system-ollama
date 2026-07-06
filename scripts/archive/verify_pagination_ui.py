import asyncio
from playwright.async_api import async_playwright
import pytest


async def test_pagination_layout():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Streamlit default port
        url = "http://localhost:8501"
        try:
            await page.goto(url, timeout=30000)
        except Exception as e:
            print(
                f"Error: Could not connect to Streamlit app at {url}. Is it running? {e}"
            )
            await browser.close()
            return False

        # 1. Upload a sample PDF file to activate the viewer
        try:
            uploader = await page.wait_for_selector("input[type='file']", timeout=10000)
            if uploader:
                await uploader.set_input_files("tests/data/2201.07520v1.pdf")
                print("Uploaded sample PDF...")
            else:
                print("Error: File uploader not found")
                await browser.close()
                return False

            await page.wait_for_selector("text=⬅️ 이전", timeout=30000)
            print("PDF Viewer loaded.")
        except Exception as e:
            print(f"Error during PDF upload or loading: {e}")
            await browser.close()
            return False

        # 2. Verify layout
        page_label = await page.query_selector("text=Page")
        of_label = await page.query_selector("text=of ")

        if not (page_label and of_label):
            print(
                f"Missing labels: label={bool(page_label)}, of_label={bool(of_label)}"
            )
            await browser.close()
            return False

        bbox_label = await page_label.bounding_box()
        bbox_of = await of_label.bounding_box()

        if not (bbox_label and bbox_of):
            print("Fail: Could not get bounding boxes for labels")
            await browser.close()
            return False

        print(f"BBox Label: {bbox_label}")
        print(f"BBox Of: {bbox_of}")

        # Debug: Print all inputs found on the page
        inputs = await page.query_selector_all("input")
        print(f"Found {len(inputs)} inputs on page")
        for i, inp in enumerate(inputs):
            bbox = await inp.bounding_box()
            attr_type = await inp.get_attribute("type")
            print(f"Input {i}: type={attr_type}, bbox={bbox}")

        # Find the number input
        page_input = None
        for inp in inputs:
            bbox = await inp.bounding_box()
            if bbox:
                # We search for any input that is horizontally near the labels
                # Instead of strict Label < Input < Of, we check if it's in the general area
                if (
                    abs(bbox["x"] - bbox_label["x"]) < 500
                    and abs(bbox["y"] - bbox_label["y"]) < 100
                ):
                    page_input = inp
                    break

        if not page_input:
            await page.screenshot(path="pagination_fail.png")
            print(
                "Fail: Could not find the number input near labels. Screenshot saved to pagination_fail.png"
            )
            await browser.close()
            return False

        bbox_input = await page_input.bounding_box()
        print(f"BBox Input: {bbox_input}")

        # FINAL VERIFICATION
        # 1. Check x-axis sequence: Label < Input < Of
        if not (
            bbox_label
            and bbox_input
            and bbox_of
            and bbox_label["x"] < bbox_input["x"] < bbox_of["x"]
        ):
            await page.screenshot(path="pagination_fail_order.png")
            print(
                f"Fail: Elements are not ordered correctly. L:{bbox_label['x']}, I:{bbox_input['x'] if bbox_input else 'N/A'}, O:{bbox_of['x']}"
            )
            await browser.close()
            return False

        # 2. Check y-axis alignment
        if bbox_label and bbox_input and bbox_of:
            y_coords = [bbox_label["y"], bbox_input["y"], bbox_of["y"]]
            if max(y_coords) - min(y_coords) > 30:
                await page.screenshot(path="pagination_fail_align.png")
                print(
                    f"Fail: Vertical misalignment. Diff: {max(y_coords) - min(y_coords)}px"
                )
                await browser.close()
                return False
        else:
            await browser.close()
            return False

        print("Success: Pagination layout looks correct!")
        await browser.close()
        return True

        bbox_label = await page_label.bounding_box()
        bbox_of = await of_label.bounding_box()

        if not (bbox_label and bbox_of):
            print("Fail: Could not get bounding boxes for labels")
            await browser.close()
            return False

        # Find the number input
        inputs = await page.query_selector_all("input")
        page_input = None
        for inp in inputs:
            bbox = await inp.bounding_box()
            if bbox:
                # Check if it's the number input (roughly between labels horizontally)
                if (
                    bbox_label["x"] < bbox["x"] < bbox_of["x"]
                    and abs(bbox["y"] - bbox_label["y"]) < 50
                ):
                    page_input = inp
                    break

        if not page_input:
            # Screenshot for debugging
            await page.screenshot(path="pagination_fail.png")
            print(
                "Fail: Could not find the number input between labels. Screenshot saved to pagination_fail.png"
            )
            await browser.close()
            return False

        bbox_input = await page_input.bounding_box()

        print(f"BBox Label: {bbox_label}")
        print(f"BBox Input: {bbox_input}")
        print(f"BBox Of: {bbox_of}")

        # Verification:
        # 1. Check x-axis sequence: Label < Input < Of
        if not (
            bbox_label
            and bbox_input
            and bbox_of
            and bbox_label["x"] < bbox_input["x"] < bbox_of["x"]
        ):
            await page.screenshot(path="pagination_fail_order.png")
            print(
                "Fail: Elements are not ordered correctly (Label < Input < Of). Screenshot saved."
            )
            await browser.close()
            return False

        # 2. Check y-axis alignment: All should be roughly on the same line
        if bbox_label and bbox_input and bbox_of:
            y_coords = [bbox_label["y"], bbox_input["y"], bbox_of["y"]]
            if max(y_coords) - min(y_coords) > 30:
                await page.screenshot(path="pagination_fail_align.png")
                print(
                    f"Fail: Vertical misalignment detected. Diff: {max(y_coords) - min(y_coords)}px. Screenshot saved."
                )
                await browser.close()
                return False
        else:
            await browser.close()
            return False

        # 2. Check y-axis alignment: All should be roughly on the same line
        y_coords = [bbox_label["y"], bbox_input["y"], bbox_of["y"]]
        if max(y_coords) - min(y_coords) > 30:
            await page.screenshot(path="pagination_fail_align.png")
            print(
                f"Fail: Vertical misalignment detected. Diff: {max(y_coords) - min(y_coords)}px. Screenshot saved."
            )
            await browser.close()
            return False

        print("Success: Pagination layout looks correct!")
        await browser.close()
        return True

        print(f"BBox Of: {bbox_of}")

        # Verification:
        # 1. Check for overlap
        if not (bbox_label["x"] < bbox_input["x"] < bbox_of["x"]):
            print("Fail: Elements are not ordered correctly (Label < Input < Of)")
            await browser.close()
            return False

        # 2. Check if any element is too small
        if any(
            b["width"] < 5 or b["height"] < 5 for b in [bbox_label, bbox_input, bbox_of]
        ):
            print(
                "Fail: One of the elements has an suspiciously small size (possibly cut off)"
            )
            await browser.close()
            return False

        # 3. Check vertical alignment
        y_coords = [bbox_label["y"], bbox_input["y"], bbox_of["y"]]
        if max(y_coords) - min(y_coords) > 20:
            print(
                f"Fail: Vertical misalignment detected. Diff: {max(y_coords) - min(y_coords)}px"
            )
            await browser.close()
            return False

        print("Success: Pagination layout looks correct!")
        await browser.close()
        return True


if __name__ == "__main__":
    result = asyncio.run(test_pagination_layout())
    if not result:
        exit(1)
    exit(0)
