# Multi-Source News Scraper & Aggregator

A robust, multi-threaded news scraping system built with **Python**, **SeleniumBase (CDP Mode)**, **Playwright**, and **PostgreSQL**. It collects top news stories across 10 major global and regional publications, automatically bypasses CAPTCHAs and anti-bot walls, and securely persists the collected data into both a localized JSON file and a PostgreSQL database.

---

## Key Features

* **Multi-Threaded Execution:** Utilizes a `ThreadPoolExecutor` to run scrapers concurrently across multiple sources with configurable thread limits.
* **CDP-Powered Anti-Bot & CAPTCHA Bypass:** Leverages SeleniumBase Chrome CDP mode alongside SeleniumBase's native auto-solver (`sb.solve_captcha()`) and an interactive pause-and-resume fallback prompt for manual challenges.
* **Incremental Scraping & Deduplication:** Automatically loads existing records from JSON caches and PostgreSQL upon startup, registering URLs into a global memory set to skip already-fetched links and avoid redundant work.
* **Dual Persistence:** Stores and syncs collected articles atomically into a structured grouped JSON file (`scraped_articles_grouped.json`) and mirrors them into a relational PostgreSQL database via a thread-safe connection pool.
* **Robust Fallbacks:** Employs flexible CSS selectors and fallback loops to reliably extract publication dates, titles, and article lead paragraphs across diverse web layouts.

---

## Prerequisites & Installation

1. **Python 3.8+** installed on your system.
2. **PostgreSQL** installed and running locally or on a remote server.
3. Install the required Python dependencies:

```bash
pip install seleniumbase playwright pandas beautifulsoup4 python-dotenv psycopg2-binary lxml
playwright install chromium
