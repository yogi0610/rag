import time
from playwright.sync_api import sync_playwright

URL = "https://rag-pdf-sithes.streamlit.app/"

# Selectors that indicate the Streamlit app has fully booted
APP_LOADED_SELECTORS = [
    "text=PDF Assistant",        # App title
    "[data-testid='stAppViewContainer']",  # Streamlit main container
    "[data-testid='stChatInput']",         # Chat input box
]

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
            page.goto(URL, timeout=90000, wait_until="domcontentloaded")
            time.sleep(5)

            # Look for the wake-up button on Streamlit's hibernation page
            wake_btn = page.locator("button:has-text('get this app back up')")

            if wake_btn.count() > 0 and wake_btn.first.is_visible():
                print("😴 App is in sleep mode! Found 'Yes, get this app back up!' button.")
                wake_btn.first.click()
                print("Clicked wake-up button. Waiting for container to fully boot (up to 120s)...")

                # Wait for the Streamlit app to actually render — not just a fixed timer
                app_loaded = False
                for selector in APP_LOADED_SELECTORS:
                    try:
                        page.wait_for_selector(selector, timeout=120000)
                        print(f"✅ App loaded! Detected selector: {selector}")
                        app_loaded = True
                        break
                    except Exception:
                        print(f"⏳ Selector '{selector}' not found yet, trying next...")

                if not app_loaded:
                    # Fallback: wait a generous fixed time if no selector matched
                    print("⚠️ No app selector matched. Waiting 90s as fallback...")
                    page.wait_for_timeout(90000)

                # Keep browser open so WebSocket stays registered with Streamlit Cloud
                print("🔗 Keeping session alive for 30s to lock in WebSocket connection...")
                page.wait_for_timeout(30000)
                print("✅ Wake-up complete! App should now stay active.")
            else:
                print("⚡ App is already awake and running. No action needed.")
                # Still keep session open briefly to count as activity
                page.wait_for_timeout(15000)

            print("Keep-alive run finished successfully.")

        except Exception as e:
            print(f"Notice during keep-alive check: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    main()
