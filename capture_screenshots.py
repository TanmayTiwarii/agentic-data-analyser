import asyncio
from playwright.async_api import async_playwright
import os

async def capture_screenshots():
    print("Ensuring assets folder exists...")
    os.makedirs("assets", exist_ok=True)
    
    async with async_playwright() as p:
        print("Launching browser...")
        browser = await p.chromium.launch(headless=True)
        # Use dark mode for the classy look we just created
        context = await browser.new_context(
            color_scheme='dark',
            viewport={'width': 1280, 'height': 800}
        )
        page = await context.new_page()
        
        print("Navigating to Streamlit app...")
        await page.goto("http://localhost:8502")
        
        # Wait for the app to load
        await page.wait_for_timeout(3000)
        
        # 1. Screenshot of the landing page
        print("Capturing landing page...")
        await page.screenshot(path="assets/screenshot_1_landing.png")
        
        # 2. Upload file
        print("Uploading dataset...")
        # Streamlit's file uploader uses a hidden input
        file_input = page.locator("input[type='file']")
        await file_input.set_input_files("data/sample_sales.csv")
        
        # Wait for upload to complete
        await page.wait_for_timeout(2000)
        
        # 3. Screenshot of dataset preview
        print("Capturing dataset preview...")
        await page.screenshot(path="assets/screenshot_2_preview.png")
        
        # 4. Run Analysis
        print("Running AI analysis...")
        await page.get_by_role("button", name="🚀 Run AI Analysis").click()
        
        # Wait for analysis to complete (up to 45 seconds)
        print("Waiting for agents to finish...")
        try:
            # We look for the "Analysis complete!" success message
            await page.get_by_text("Analysis complete!", exact=False).wait_for(timeout=60000)
        except Exception:
            print("Wait timed out, proceeding anyway...")
        
        await page.wait_for_timeout(2000)
        
        # 5. Screenshot of agent workflow tab
        print("Capturing workflow tab...")
        await page.get_by_role("tab", name="Agent Workflow").click()
        await page.wait_for_timeout(1000)
        await page.screenshot(path="assets/screenshot_3_workflow.png", full_page=True)
        
        # 6. Screenshot of visualizations tab
        print("Capturing visualizations tab...")
        await page.get_by_role("tab", name="Visualizations").click()
        await page.wait_for_timeout(1000)
        await page.screenshot(path="assets/screenshot_4_visualizations.png", full_page=True)
        
        # 7. Screenshot of insights tab
        print("Capturing insights tab...")
        await page.get_by_role("tab", name="Insights").click()
        await page.wait_for_timeout(1000)
        await page.screenshot(path="assets/screenshot_5_insights.png", full_page=True)
        
        # 8. Chat interaction
        print("Capturing chat tab...")
        await page.get_by_role("tab", name="Ask AI").click()
        await page.wait_for_timeout(1000)
        # Ask a question
        await page.get_by_placeholder("e.g., What is the average value").fill("What are the main trends in this dataset?")
        await page.get_by_role("button", name="Send").click()
        
        # Wait for AI to respond
        await page.wait_for_timeout(8000)
        await page.screenshot(path="assets/screenshot_6_chat.png", full_page=True)
        
        print("All screenshots captured successfully!")
        await browser.close()

if __name__ == "__main__":
    asyncio.run(capture_screenshots())
