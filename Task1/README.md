# Global News Scraper

A robust, multi-threaded Python web scraper designed to collect news articles from ten distinct global news publishers. The script extracts article metadata and text, managing them in a grouped JSON database that can easily be loaded into dictionaries or dataframes for natural language processing and machine learning tasks. 

## 🚀 Features

*   **Broad Source Coverage:** Scrapes articles across 10 distinct websites: AAP News, Al Jazeera, Arab News, Bloomberg, CBS News, Express Tribune, Reuters, Straits Times, The New Republic, and UN News.
*   **Interactive CAPTCHA Guard:** Actively monitors page content for bot-detection walls (e.g., Cloudflare, reCAPTCHA, or "Verify you are human" prompts). If triggered, the script pauses execution and provides a 120-second window for manual CAPTCHA resolution in the browser window before resuming.
*   **Concurrent Execution:** Utilizes `ThreadPoolExecutor` to run multiple site scrapers in parallel, significantly reducing total runtime.
*   **Persistent Checkpointing:** Automatically saves the master dictionary to the JSON database every 10 articles (`SAVE_INTERVAL`) to prevent data loss in the event of an interruption. 
*   **Thread-Safe Duplicate Prevention:** Maintains a global, lock-protected cache of previously scraped URLs, ensuring no duplicate articles are appended to the dataset across sessions.

## 🛠️ Prerequisites

Ensure you have Python installed, along with the required third-party libraries. You can install the dependencies via pip:

```bash
pip install requests beautifulsoup4 seleniumbase playwright
