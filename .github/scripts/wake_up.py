import time
from playwright.sync_api import sync_playwright

URL = "https://rag-pdf-sithes.streamlit.app/"

def main():
    print(f"🚀 Launching headless browser to check {URL}...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        try:
            print(f"Navigating to {URL} ...")
            page.goto(URL, timeout=60000, wait_until="domcontentloaded")
            time.sleep(5)

            # Look for the wake-up button on Streamlit's hibernation page
            wake_btn = page.locator("button:has-text('get this app back up')")

            if wake_btn.count() > 0 and wake_btn.first.is_visible():
                print("😴 App is in sleep mode! Found 'Yes, get this app back up!' button.")
                wake_btn.first.click()
                print("Waking up container... waiting 15 seconds.")
                page.wait_for_timeout(15000)
                print("✅ Successfully clicked wake-up button!")
            else:
                print("⚡ App is awake and running. Active session confirmed.")

            # Keep page open briefly so WebSocket connection registers on Streamlit servers
            page.wait_for_timeout(8000)
            print("Keep-alive run finished successfully.")

        except Exception as e:
            print(f"Notice during keep-alive check: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    main()
