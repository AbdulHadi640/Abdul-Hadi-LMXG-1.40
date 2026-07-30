import os
import json
import time
import uuid
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
from seleniumbase import sb_cdp
from playwright.sync_api import sync_playwright
from dotenv import load_dotenv
import psycopg2
from psycopg2 import pool as pg_pool

# =============================================================
# ENV / GLOBAL CONFIGURATION & CACHE
# =============================================================
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

JSON_FILE = os.getenv("JSON_FILE", "scraped_articles_grouped.json")
TARGET_COUNT = int(os.getenv("TARGET_COUNT", "150"))          # hard per-site article cap
SAVE_INTERVAL = int(os.getenv("SAVE_INTERVAL", "10"))
MAX_PARALLEL_SCRAPERS = int(os.getenv("MAX_PARALLEL_SCRAPERS", "2"))

# Safety valves so a scraper never spins forever once a site runs out of fresh content
MAX_PAGES_SAFETY = int(os.getenv("MAX_PAGES_SAFETY", "60"))
MAX_STUCK_ATTEMPTS = int(os.getenv("MAX_STUCK_ATTEMPTS", "6"))

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_DATABASE = os.getenv("PG_DATABASE", "news_scraper")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "")

file_lock = threading.Lock()
master_articles_dict = {}
scraped_urls_set = set()
checkpoint_counter = 0

db_pool = None

# =============================================================
# POSTGRES SETUP
# =============================================================
def init_db():
    """Creates a threaded connection pool and ensures the articles table exists"""
    global db_pool
    masked_pw = "(empty)" if not PG_PASSWORD else f"({len(PG_PASSWORD)} chars set)"
    print(f"🔧 Postgres config -> host={PG_HOST} port={PG_PORT} db={PG_DATABASE} user={PG_USER} password={masked_pw}")

    try:
        db_pool = pg_pool.ThreadedConnectionPool(
            1, MAX_PARALLEL_SCRAPERS + 4,
            host=PG_HOST, port=PG_PORT, dbname=PG_DATABASE,
            user=PG_USER, password=PG_PASSWORD
        )
        setup_key = uuid.uuid4().hex
        conn = db_pool.getconn(key=setup_key)
        try:
            with conn.cursor() as cur:
                cur.execute("""
                   CREATE TABLE IF NOT EXISTS articles (
                     id SERIAL PRIMARY KEY,
                     source_name TEXT NOT NULL,
                     title TEXT,
                     url TEXT UNIQUE NOT NULL,
                     published_date TEXT,
                     first_paragraph TEXT,
                     scraped_at TIMESTAMPTZ,
                     created_at TIMESTAMPTZ DEFAULT NOW()
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source_name);")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_articles_url ON articles(url);")
            conn.commit()
            print(f"🗄️  Connected to Postgres '{PG_DATABASE}' — 'articles' table ready.")
        finally:
            db_pool.putconn(conn, key=setup_key)
    except Exception as e:
        print(f"❌ Could not initialize Postgres connection: {e}")
        print("⚠️ Continuing with JSON-only storage.")
        db_pool = None

def insert_article_db(source_name, article_dict):
    """Thread-safe insert into Postgres. Skips (no-ops) on duplicate URL.

    IMPORTANT: psycopg2's ThreadedConnectionPool, when called with no explicit
    key, ties a checked-out connection to the *OS thread identity* of the
    caller. Because our ThreadPoolExecutor reuses a small, fixed number of
    worker threads across many different scraper functions, that automatic
    keying can get out of sync under load and corrupt the pool's internal
    bookkeeping (surfacing as a cryptic 'tuple index out of range' error).
    Passing our own unique key per borrow/return pair sidesteps that entirely
    — each getconn()/putconn() call is now fully independent.
    """
    if db_pool is None:
        return False

    pool_key = uuid.uuid4().hex
    conn = None
    broken = False
    try:
        conn = db_pool.getconn(key=pool_key)
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO articles (source_name, title, url, published_date, first_paragraph, scraped_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (url) DO NOTHING;
            """, (
                source_name,
                article_dict.get("title"),
                article_dict.get("url"),
                article_dict.get("published_date"),
                article_dict.get("first_paragraph"),
                article_dict.get("scraped_at"),
            ))
            inserted = cur.rowcount > 0
        conn.commit()
        return inserted
    except Exception as e:
        broken = True
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                pass
        print(f"❌ DB insert error for {article_dict.get('url')}: {e}")
        return False
    finally:
        if conn is not None:
            try:
                # Discard (don't recycle) a connection that hit an error mid-transaction,
                # so a single bad connection can't poison future inserts.
                db_pool.putconn(conn, key=pool_key, close=broken)
            except Exception:
                pass

def close_db():
    if db_pool is not None:
        db_pool.closeall()
        print("🗄️  Postgres connection pool closed.")

def load_existing_data():
    """Loads existing JSON file structured by news web names to populate memory caches and syncs them into database."""
    global master_articles_dict, scraped_urls_set
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, "r", encoding="utf-8") as f:
                master_articles_dict = json.load(f)
                total_loaded = 0
                synced_to_db = 0
                for source_name, articles in master_articles_dict.items():
                    if isinstance(articles, list):
                        for item in articles:
                            if "url" in item:
                                scraped_urls_set.add(item["url"])
                                total_loaded += 1
                                if db_pool is not None:
                                    if insert_article_db(source_name, item):
                                        synced_to_db += 1
            print(f"📦 Loaded {total_loaded} existing articles across {len(master_articles_dict)} sources from {JSON_FILE}.")
            if db_pool is not None and synced_to_db > 0:
                print(f"🔄 Synced {synced_to_db} missing items from JSON into PostgreSQL.")
        except Exception as e:
            print(f"⚠️ Error reading JSON cache: {e}")
            master_articles_dict = {}

def save_checkpoint_json():
    """Atomically commits the grouped master database to the JSON file."""
    with file_lock:
        try:
            temp_file = f"{JSON_FILE}.tmp"
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(master_articles_dict, f, indent=4, ensure_ascii=False)
            os.replace(temp_file, JSON_FILE)
            total_items = sum(len(v) for v in master_articles_dict.values() if isinstance(v, list))
            print(f"💾 [SAVED] Total Database: {total_items} items stored under source keys in {JSON_FILE}")
        except Exception as e:
            print(f"❌ Error saving JSON file: {e}")

def is_new_url(url):
    """Thread-safe peek check — does NOT mark the URL as scraped.
    Use this while building candidate-link lists so scrapers can stop
    collecting the instant they have enough genuinely new articles,
    without wasting requests on ones already stored."""
    if not url:
        return False
    with file_lock:
        return url not in scraped_urls_set

def register_article(source_name, article_dict):
    """Thread-safe registration grouping items under their respective news web name key,
    and mirrored into Postgres. Returns True only if the article was newly added."""
    global checkpoint_counter
    url = article_dict.get("url")

    with file_lock:
        if url in scraped_urls_set:
            return False

        scraped_urls_set.add(url)

        if source_name not in master_articles_dict:
            master_articles_dict[source_name] = []

        master_articles_dict[source_name].append(article_dict)
        checkpoint_counter += 1
        should_save = (checkpoint_counter >= SAVE_INTERVAL)
        if should_save:
            checkpoint_counter = 0

    # Mirror into Postgres (safe no-op if init_db() wasn't called or failed)
    insert_article_db(source_name, article_dict)

    if should_save:
        save_checkpoint_json()
    return True

# =============================================================
# INTERACTIVE CAPTCHA / BOT WALL HANDLER (Pause & Resume)
# =============================================================
def check_and_handle_captcha(page, site_name):
    """Detects security challenges, pauses execution, and prompts the user to solve them safely."""
    try:
        page_content = page.content().lower()
        captcha_indicators = [
            "cf-chl-bypass", "g-recaptcha", "h-captcha",
            "challenge-running", "verify you are human", "access denied",
            "press & hold", "unusual activity", "please enable cookies", "checking your browser"
        ]

        is_blocked = any(ind in page_content for ind in captcha_indicators)

        if is_blocked:
            print(f"\n🚨 [BLOCK DETECTED] CAPTCHA or Anti-Bot wall triggered on [{site_name}]!")
            print(f"🛑 PAUSED: Please go to the browser window and solve the CAPTCHA manually.")
            print(f"⏳ Waiting up to 120 seconds for you to complete it...\n")

            for _ in range(120):
                time.sleep(1)
                try:
                    current_content = page.content().lower()
                    still_blocked = any(ind in current_content for ind in captcha_indicators)
                    if not still_blocked:
                        print(f"✅ [RESUMED] CAPTCHA solved successfully for [{site_name}]!\n")
                        return True
                except Exception:
                    pass

            print(f"⚠️ [TIMEOUT] Manual window expired for [{site_name}]. Attempting to proceed...\n")
    except Exception as e:
        print(f"⚠️ CAPTCHA check warning on {site_name}: {e}")
    return False

# =============================================================
# 10 INDIVIDUAL SITE SCRAPER MODULES
# =============================================================

def scrape_aapnews():
    site_name = "AAP News"
    base_url = "https://publications.aap.org/aapnews/search-results?sort=Date+-+Newest+First&fl_SiteID=1000011&page="
    print(f"🚀 Starting {site_name} scraper...")
    sb = None
    try:
        sb = sb_cdp.Chrome(locale="en")
        sb.goto(f"{base_url}1")
        time.sleep(3)
        endpoint_url = sb.get_endpoint_url()

        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(endpoint_url)
            page = browser.contexts[0].pages[0]

            collected = 0
            current_page = 1

            while collected < TARGET_COUNT and current_page <= MAX_PAGES_SAFETY:
                page.goto(f"{base_url}{current_page}")
                check_and_handle_captcha(page, site_name)

                try:
                    page.wait_for_selector(".sr-list_wrap.new-results .sr-list.al-article-box", timeout=8000)
                except Exception:
                    break

                boxes = page.locator(".sr-list_wrap.new-results .sr-list.al-article-box")
                count = boxes.count()
                if count == 0:
                    break

                for i in range(count):
                    if collected >= TARGET_COUNT:
                        break
                    box = boxes.nth(i)
                    title_a = box.locator(".sri-title.al-title h4 a")
                    title = title_a.inner_text().strip() if title_a.count() > 0 else ""
                    link = title_a.get_attribute("href") if title_a.count() > 0 else ""

                    if link and link.startswith("/"):
                        link = f"https://publications.aap.org{link}"

                    if not is_new_url(link):
                        continue

                    date_el = box.locator(".sf-facet-display")
                    pub_date = date_el.inner_text().replace("Published:", "").strip() if date_el.count() > 0 else "N/A"

                    abstract_el = box.locator(".abstract-response-placeholder")
                    abstract = abstract_el.inner_text().strip() if abstract_el.count() > 0 else "N/A"

                    item = {
                        "title": title, "url": link,
                        "published_date": pub_date,
                        "first_paragraph": abstract, "scraped_at": datetime.now(timezone.utc).isoformat()
                    }
                    if register_article(site_name, item):
                        collected += 1
                        if collected >= TARGET_COUNT:
                            break

                current_page += 1
                time.sleep(1)
            print(f"🏁 {site_name}: collected {collected}/{TARGET_COUNT} new articles.")
            browser.close()
    except Exception as e:
        print(f"⚠️ Exception in {site_name}: {e}")
    finally:
        if sb: sb.quit()

def scrape_aljazeera():
    site_name = "Al Jazeera"
    base_url = "https://www.aljazeera.com"
    news_url = "https://www.aljazeera.com/news/"
    print(f"🚀 Starting {site_name} scraper...")
    sb = None
    try:
        sb = sb_cdp.Chrome(locale="en")
        sb.goto(news_url)
        time.sleep(3)
        endpoint_url = sb.get_endpoint_url()

        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(endpoint_url)
            page = browser.contexts[0].pages[0]

            check_and_handle_captcha(page, site_name)
            article_selector = "article.article-card"
            try: page.wait_for_selector(article_selector, timeout=10000)
            except Exception: pass

            # Incrementally collect only NEW unique links, stopping the instant we hit TARGET_COUNT.
            links = []
            seen_in_batch = set()
            stuck = 0

            while len(links) < TARGET_COUNT and stuck < MAX_STUCK_ATTEMPTS:
                cards = page.locator(article_selector)
                count = cards.count()

                for i in range(count):
                    if len(links) >= TARGET_COUNT:
                        break
                    card = cards.nth(i)
                    link_el = card.locator("a.article-card__link, h2.article-card__title a").first
                    if link_el.count() == 0:
                        continue
                    href = link_el.get_attribute("href")
                    if not href:
                        continue
                    url = href if href.startswith("http") else f"{base_url}{href}"
                    if url in seen_in_batch or not is_new_url(url):
                        continue
                    title = link_el.inner_text().strip()
                    links.append({"title": title, "url": url})
                    seen_in_batch.add(url)

                if len(links) >= TARGET_COUNT:
                    break

                before = len(links)
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(1)
                btn = page.locator("button.show-more-button").first
                if btn.count() > 0 and btn.is_visible():
                    try:
                        btn.click(force=True)
                        time.sleep(2)
                    except Exception:
                        pass
                else:
                    time.sleep(1)

                stuck = stuck + 1 if len(links) == before else 0

            for item_info in links:
                try:
                    page.goto(item_info['url'], wait_until="domcontentloaded", timeout=20000)
                    check_and_handle_captcha(page, site_name)

                    pub_date = "N/A"
                    time_tag = page.locator("time").first
                    if time_tag.count() > 0:
                        pub_date = time_tag.inner_text().strip()
                    else:
                        date_el = page.locator(".article-dates, .date-simple, [data-testid='date']").first
                        if date_el.count() > 0:
                            pub_date = date_el.inner_text().strip()

                    content_p = page.locator(".wysiwyg p, article p, main p").first
                    first_p = content_p.inner_text().strip() if content_p.count() > 0 else "N/A"

                    register_article(site_name, {
                        "title": item_info["title"], "url": item_info["url"],
                        "published_date": pub_date, "first_paragraph": first_p,
                        "scraped_at": datetime.now(timezone.utc).isoformat()
                    })
                except Exception as e:
                    print(f"⚠️ Al Jazeera subpage exception: {e}")
                    continue
            print(f"🏁 {site_name}: collected {len(links)}/{TARGET_COUNT} new articles.")
            browser.close()
    except Exception as e:
        print(f"⚠️ Exception in {site_name}: {e}")
    finally:
        if sb: sb.quit()

def scrape_arabnews():
    site_name = "Arab News"
    base_url = "https://www.arabnews.com"
    news_url = "https://www.arabnews.com/world"
    print(f"🚀 Starting {site_name} scraper...")
    sb = None
    try:
        sb = sb_cdp.Chrome(locale="en")
        sb.goto(news_url)
        time.sleep(3)
        endpoint_url = sb.get_endpoint_url()

        articles_data = []
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(endpoint_url)
            page = browser.contexts[0].pages[0]

            check_and_handle_captcha(page, site_name)
            card_selector = ".view-content .article-item"
            page_num = 0

            while len(articles_data) < TARGET_COUNT and page_num <= MAX_PAGES_SAFETY:
                current_url = f"{news_url}?page={page_num}" if page_num > 0 else news_url
                try:
                    page.goto(current_url, wait_until="domcontentloaded", timeout=20000)
                    page.wait_for_selector(card_selector, timeout=10000)
                except Exception as e:
                    print(f"⚠️ Pagination load warning on {site_name}: {e}")
                    break

                cards = page.locator(card_selector)
                count = cards.count()
                if count == 0:
                    break

                before = len(articles_data)
                for i in range(count):
                    if len(articles_data) >= TARGET_COUNT:
                        break
                    card = cards.nth(i)
                    title_link = card.locator(".article-item-title a, h2 a, h3 a").first
                    if title_link.count() > 0:
                        title = title_link.inner_text().strip()
                        href = title_link.get_attribute("href")
                        if href:
                            full_url = href if href.startswith("http") else f"{base_url}{href}"
                            if is_new_url(full_url) and not any(item["url"] == full_url for item in articles_data):
                                articles_data.append({"title": title if title else "N/A", "url": full_url})

                if len(articles_data) >= TARGET_COUNT:
                    break
                # If a full page yielded nothing new, the site has likely run out of fresh content.
                if len(articles_data) == before:
                    break
                page_num += 1

            for item_info in articles_data:
                try:
                    page.goto(item_info["url"], wait_until="domcontentloaded", timeout=20000)
                    check_and_handle_captcha(page, site_name)

                    time_el = page.locator(".entry-date time, .article-item-meta time, time").first
                    pub_date = time_el.inner_text().strip() if time_el.count() > 0 else "N/A"

                    current_title = item_info["title"]
                    if not current_title or current_title == "N/A":
                        h1_el = page.locator("h1.article-title, h1").first
                        if h1_el.count() > 0:
                            current_title = h1_el.inner_text().strip()

                    first_p = "N/A"
                    paragraphs = page.locator("article p, .main-content p, p")
                    for p_idx in range(min(paragraphs.count(), 5)):
                        text = paragraphs.nth(p_idx).inner_text().strip()
                        if len(text) > 40:
                            first_p = text
                            break

                    register_article(site_name, {
                        "title": current_title, "url": item_info["url"],
                        "published_date": pub_date, "first_paragraph": first_p,
                        "scraped_at": datetime.now(timezone.utc).isoformat()
                    })
                except Exception as e:
                    print(f"⚠️ Arab News subpage exception: {e}")
                    continue
            print(f"🏁 {site_name}: collected {len(articles_data)}/{TARGET_COUNT} new articles.")
            try:
                browser.close()
            except:
                pass
    except Exception as e:
        print(f"⚠️ Exception in {site_name}: {e}")
    finally:
        if sb:
            try:
                sb.quit()
            except:
                pass

def scrape_bloomberg():
    site_name = "Bloomberg"
    print(f"🚀 Starting {site_name} scraper...")
    sb = None
    try:
        sb = sb_cdp.Chrome(locale="en")
        sb.goto("https://www.bloomberg.com/latest")
        time.sleep(4)
        endpoint_url = sb.get_endpoint_url()

        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(endpoint_url)
            page = browser.contexts[0].pages[0]

            check_and_handle_captcha(page, site_name)
            card_selector = "div.Latest_storyPadding__GBJUE"
            try: page.wait_for_selector(card_selector, timeout=15000)
            except Exception: pass

            # Incrementally collect only NEW unique links, stopping the instant we hit TARGET_COUNT.
            news_links = []
            seen_in_batch = set()
            stuck_count = 0

            while len(news_links) < TARGET_COUNT and stuck_count < MAX_STUCK_ATTEMPTS:
                cards = page.locator(card_selector)
                count = cards.count()

                for i in range(count):
                    if len(news_links) >= TARGET_COUNT:
                        break
                    card = cards.nth(i)
                    try:
                        headline_el = card.locator('[data-testid="headline"]').first
                        link_el = card.locator("a.Latest_storyLink__80QVD, a[href*='/news/']").first
                        if headline_el.count() == 0 or link_el.count() == 0:
                            continue
                        url = link_el.get_attribute("href")
                        if not url:
                            continue
                        if url.startswith("/"):
                            url = "https://www.bloomberg.com" + url
                        if url in seen_in_batch or not is_new_url(url):
                            continue
                        title = headline_el.inner_text().strip()
                        news_links.append({"title": title, "url": url})
                        seen_in_batch.add(url)
                    except Exception:
                        continue

                if len(news_links) >= TARGET_COUNT:
                    break

                before = len(news_links)
                page.evaluate("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(1.5)

                click_success = page.evaluate("""() => {
                    const container = document.querySelector('[data-testid="load-more"]');
                    if (!container) return false;
                    const button = container.querySelector('button') || container;
                    if (button) {
                        button.scrollIntoView({block: 'center'});
                        button.dispatchEvent(new MouseEvent('mousedown', {bubbles: true, cancelable: true}));
                        button.dispatchEvent(new MouseEvent('mouseup', {bubbles: true, cancelable: true}));
                        button.dispatchEvent(new MouseEvent('click', {bubbles: true, cancelable: true}));
                        return true;
                    }
                    return false;
                }""")

                check_and_handle_captcha(page, site_name)

                if click_success:
                    time.sleep(3)
                    page.evaluate("window.scrollBy(0, 300);")
                else:
                    time.sleep(2)

                stuck_count = stuck_count + 1 if len(news_links) == before else 0

            for article in news_links:
                try:
                    page.goto(article["url"], wait_until="domcontentloaded", timeout=15000)
                    check_and_handle_captcha(page, site_name)
                    page.wait_for_timeout(1000)

                    try: title = page.locator("h1").first.inner_text().strip()
                    except Exception: title = article["title"]

                    try: first_p = page.locator('p[data-component="paragraph"], article p').first.inner_text().strip()
                    except Exception: first_p = ""

                    try: pub_date = page.locator('div[data-component="timestamp"] time, time').first.get_attribute("datetime")
                    except Exception: pub_date = ""

                    register_article(site_name, {
                        "title": title, "url": article["url"],
                        "published_date": pub_date, "first_paragraph": first_p,
                        "scraped_at": datetime.now(timezone.utc).isoformat()
                    })
                except Exception:
                    continue

            print(f"🏁 {site_name}: collected {len(news_links)}/{TARGET_COUNT} new articles.")
            browser.close()
    except Exception as e:
        print(f"⚠️ Exception in {site_name}: {e}")
    finally:
        if sb: sb.quit()

def scrape_cbsnews():
    site_name = "CBS News"
    cbs_url = "https://www.cbsnews.com/"
    print(f"🚀 Starting {site_name} scraper...")

    try:
        resp = requests.get(cbs_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'lxml')
            articles = soup.find_all('article', class_="item")
            collected = 0

            for article in articles:
                if collected >= TARGET_COUNT:
                    break
                try:
                    anchor = article.find('a', class_="item__anchor")
                    if not anchor or not anchor.has_attr("href"):
                        continue
                    link = anchor["href"]

                    if link and link.startswith("http") and is_new_url(link):
                        sub_resp = requests.get(link, headers={"User-Agent": "Mozilla/5.0"}, timeout=8)
                        if sub_resp.status_code == 200:
                            soup2 = BeautifulSoup(sub_resp.text, 'lxml')
                            section = soup2.find("section", class_="content__body")
                            if section and section.find("p"):
                                meta_p = soup2.find("p", class_="content__meta")
                                raw_time = meta_p.text.strip() if meta_p else "N/A"
                                first_p = section.find("p").text.strip()
                                hed = article.find('h4', class_="item__hed")
                                title = hed.text.strip() if hed else "N/A"

                                item = {
                                    "title": title, "url": link,
                                    "published_date": raw_time, "first_paragraph": first_p,
                                    "scraped_at": datetime.now(timezone.utc).isoformat()
                                }
                                if register_article(site_name, item):
                                    collected += 1
                except Exception:
                    continue
            print(f"🏁 {site_name}: collected {collected}/{TARGET_COUNT} new articles (site's homepage list is naturally limited).")
    except Exception as e:
        print(f"⚠️ Exception in {site_name}: {e}")

def scrape_tribune():
    site_name = "Express Tribune"
    base_url = "https://tribune.com.pk"
    news_url = "https://tribune.com.pk/latest"
    print(f"🚀 Starting {site_name} scraper...")
    sb = None
    try:
        sb = sb_cdp.Chrome(locale="en")
        sb.goto(news_url)
        time.sleep(3)
        endpoint_url = sb.get_endpoint_url()

        articles_data = []
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(endpoint_url)
            page = browser.contexts[0].pages[0]

            check_and_handle_captcha(page, site_name)
            card_selector = "#all ul.tedit-shortnews > li"
            page_num = 1

            while len(articles_data) < TARGET_COUNT and page_num <= MAX_PAGES_SAFETY:
                current_url = f"{news_url}?page={page_num}"
                try:
                    page.goto(current_url, wait_until="domcontentloaded", timeout=12000)
                    page.wait_for_selector(card_selector, timeout=6000)
                except Exception:
                    break

                cards = page.locator(card_selector)
                card_count = cards.count()
                if card_count == 0:
                    break

                before = len(articles_data)
                for i in range(card_count):
                    if len(articles_data) >= TARGET_COUNT:
                        break
                    card = cards.nth(i)
                    link_element = card.locator(".horiz-news3-caption a").first

                    if link_element.count() > 0:
                        title_heading = link_element.locator("h2.title-heading").first
                        title = title_heading.inner_text().strip() if title_heading.count() > 0 else link_element.inner_text().strip()
                        href = link_element.get_attribute("href")
                        if href:
                            full_url = href if href.startswith("http") else f"{base_url}{href}"
                            if is_new_url(full_url) and not any(item["url"] == full_url for item in articles_data):
                                articles_data.append({"title": title, "url": full_url})

                if len(articles_data) >= TARGET_COUNT:
                    break
                if len(articles_data) == before:
                    break
                page_num += 1

            for item_info in articles_data:
                try:
                    page.goto(item_info["url"], wait_until="domcontentloaded", timeout=12000)
                    check_and_handle_captcha(page, site_name)
                    pub_date = "N/A"
                    date_spans = page.locator(".left-authorbox span")
                    for d_idx in range(date_spans.count()):
                        text = date_spans.nth(d_idx).inner_text().strip()
                        if any(month in text for month in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]):
                            pub_date = text
                            break

                    first_p = "N/A"
                    paragraphs = page.locator(".story-text p")
                    for p_idx in range(min(paragraphs.count(), 5)):
                        text = paragraphs.nth(p_idx).inner_text().strip()
                        if len(text) > 30:
                            first_p = text
                            break

                    register_article(site_name, {
                        "title": item_info["title"], "url": item_info["url"],
                        "published_date": pub_date, "first_paragraph": first_p,
                        "scraped_at": datetime.now(timezone.utc).isoformat()
                    })
                except Exception:
                    continue
            print(f"🏁 {site_name}: collected {len(articles_data)}/{TARGET_COUNT} new articles.")
            browser.close()
    except Exception as e:
        print(f"⚠️ Exception in {site_name}: {e}")
    finally:
        if sb: sb.quit()

def scrape_reuters():
    site_name = "Reuters"
    base_url = "https://www.reuters.com"
    print(f"🚀 Starting {site_name} scraper...")
    sb = None
    try:
        sb = sb_cdp.Chrome(locale="en")
        sb.goto("https://www.reuters.com/world/")
        time.sleep(3)

        # Incrementally collect only NEW unique links, stopping the instant we hit TARGET_COUNT.
        article_links = []
        seen_in_batch = set()
        stuck_counter = 0

        while len(article_links) < TARGET_COUNT and stuck_counter < MAX_STUCK_ATTEMPTS:
            sb.scroll_to_bottom()
            time.sleep(1)
            list_elements = sb.find_elements('ul[data-testid="FeedList"] > li')

            before = len(article_links)
            for element in list_elements:
                if len(article_links) >= TARGET_COUNT:
                    break
                if not element:
                    continue
                try:
                    title_a = element.querySelector('div[data-testid="Title"] a')
                    if not title_a:
                        continue
                    text = title_a.text.strip()
                    href = title_a.get_attribute("href")
                    if not href:
                        continue
                    full_link = href if href.startswith("http") else f"{base_url}{href}"
                    if full_link in seen_in_batch or not is_new_url(full_link):
                        continue
                    article_links.append({"title": text, "url": full_link})
                    seen_in_batch.add(full_link)
                except Exception:
                    continue

            if len(article_links) >= TARGET_COUNT:
                break

            if len(article_links) == before:
                stuck_counter += 1
            else:
                stuck_counter = 0

            try:
                load_more_btn = sb.find_element('button[data-testid="FeedContentLoadMore"]', timeout=2)
                if load_more_btn:
                    sb.scroll_into_view('button[data-testid="FeedContentLoadMore"]')
                    time.sleep(0.5)
                    load_more_btn.click()
                    time.sleep(2)
            except Exception:
                break

        for article in article_links:
            try:
                sb.goto(article["url"])
                time.sleep(1)

                try:
                    date_el = sb.find_element('[data-testid="DateLine"]', timeout=2)
                    raw_date = date_el.text.strip() if date_el else "N/A"
                except Exception:
                    raw_date = "N/A"

                try:
                    p_el = sb.find_element('[data-testid="paragraph-0"]', timeout=2)
                    if not p_el:
                        p_el = sb.find_element("article p", timeout=2)
                    first_p = p_el.text.strip() if p_el else "N/A"
                except Exception:
                    first_p = "N/A"

                register_article(site_name, {
                    "title": article["title"], "url": article["url"],
                    "published_date": raw_date, "first_paragraph": first_p,
                    "scraped_at": datetime.now(timezone.utc).isoformat()
                })
            except Exception:
                continue
        print(f"🏁 {site_name}: collected {len(article_links)}/{TARGET_COUNT} new articles.")
    except Exception as e:
        print(f"⚠️ Exception in {site_name}: {e}")
    finally:
        if sb: sb.quit()

def scrape_straitstimes():
    site_name = "Straits Times"
    base_url = "https://www.straitstimes.com"
    news_url = "https://www.straitstimes.com/world"
    print(f"🚀 Starting {site_name} scraper...")
    sb = None
    try:
        sb = sb_cdp.Chrome(locale="en")
        sb.goto(news_url)
        time.sleep(3)
        endpoint_url = sb.get_endpoint_url()

        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(endpoint_url)
            page = browser.contexts[0].pages[0]

            card_selector = "div[id^='section-latest-headline-card-']"
            try: page.wait_for_selector(card_selector, timeout=12000)
            except Exception: pass

            # Incrementally collect only NEW unique links, stopping the instant we hit TARGET_COUNT.
            articles_data = []
            seen_in_batch = set()
            retry_count = 0

            while len(articles_data) < TARGET_COUNT and retry_count < MAX_STUCK_ATTEMPTS:
                cards = page.locator(card_selector)
                count = cards.count()

                for i in range(count):
                    if len(articles_data) >= TARGET_COUNT:
                        break
                    card = cards.nth(i)
                    link_el = card.locator("a[data-testid='custom-link']").first
                    heading_el = card.locator("h4[data-testid='heading-test-id']").first
                    if link_el.count() == 0:
                        continue
                    href = link_el.get_attribute("href")
                    if not href:
                        continue
                    full_url = href if href.startswith("http") else f"{base_url}{href}"
                    if full_url in seen_in_batch or not is_new_url(full_url):
                        continue
                    title = heading_el.inner_text().strip() if heading_el.count() > 0 else "N/A"
                    articles_data.append({"title": title, "url": full_url})
                    seen_in_batch.add(full_url)

                if len(articles_data) >= TARGET_COUNT:
                    break

                before = len(articles_data)
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(1)
                load_btn = page.locator("button:has-text('Load more')").first
                if load_btn.count() > 0 and load_btn.is_visible():
                    try:
                        load_btn.click(force=True)
                        time.sleep(2)
                    except Exception:
                        pass
                else:
                    time.sleep(1)

                retry_count = retry_count + 1 if len(articles_data) == before else 0

            for item_info in articles_data:
                try:
                    page.goto(item_info["url"], wait_until="domcontentloaded", timeout=12000)
                    check_and_handle_captcha(page, site_name)
                    ts_el = page.locator("[data-testid='timestamp-test-id'] p").first
                    pub_date = ts_el.inner_text().replace("Published", "").strip() if ts_el.count() > 0 else "N/A"

                    para_el = page.locator("p[data-testid='article-paragraph-annotation-test-id']").first
                    first_p = para_el.inner_text().strip() if para_el.count() > 0 else "N/A"

                    register_article(site_name, {
                        "title": item_info["title"], "url": item_info["url"],
                        "published_date": pub_date, "first_paragraph": first_p,
                        "scraped_at": datetime.now(timezone.utc).isoformat()
                    })
                except Exception:
                    continue
            print(f"🏁 {site_name}: collected {len(articles_data)}/{TARGET_COUNT} new articles.")
            browser.close()
    except Exception as e:
        print(f"⚠️ Exception in {site_name}: {e}")
    finally:
        if sb: sb.quit()

def scrape_newrepublic():
    site_name = "The New Republic"
    base_url = "https://newrepublic.com"
    news_url = "https://newrepublic.com/latest"
    print(f"🚀 Starting {site_name} scraper...")
    sb = None
    try:
        sb = sb_cdp.Chrome(locale="en")
        sb.goto(news_url)
        time.sleep(3)
        endpoint_url = sb.get_endpoint_url()

        articles_data = []
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(endpoint_url)
            page = browser.contexts[0].pages[0]

            card_selector = ".articleResults__results .articleResults__result"
            page_num = 1

            while len(articles_data) < TARGET_COUNT and page_num <= MAX_PAGES_SAFETY:
                current_url = f"{news_url}?page={page_num}"
                try:
                    page.goto(current_url, wait_until="domcontentloaded", timeout=12000)
                    page.wait_for_selector(card_selector, timeout=6000)
                except Exception:
                    break

                cards = page.locator(card_selector)
                count = cards.count()
                if count == 0:
                    break

                before = len(articles_data)
                for i in range(count):
                    if len(articles_data) >= TARGET_COUNT:
                        break
                    card = cards.nth(i)
                    link_el = card.locator("a.Hed").first
                    if link_el.count() > 0:
                        title = link_el.inner_text().strip()
                        href = link_el.get_attribute("href")
                        if href:
                            full_url = href if href.startswith("http") else f"{base_url}{href}"
                            if is_new_url(full_url) and not any(item["url"] == full_url for item in articles_data):
                                articles_data.append({"title": title, "url": full_url})

                if len(articles_data) >= TARGET_COUNT:
                    break
                if len(articles_data) == before:
                    break
                page_num += 1

            for item_info in articles_data:
                try:
                    page.goto(item_info["url"], wait_until="domcontentloaded", timeout=12000)
                    check_and_handle_captcha(page, site_name)
                    time_el = page.locator(".date-mobile time, time").first
                    pub_date = time_el.inner_text().strip() if time_el.count() > 0 else "N/A"

                    first_p = "N/A"
                    paragraphs = page.locator(".article-text-grid p, article p")
                    for p_idx in range(min(paragraphs.count(), 5)):
                        text = paragraphs.nth(p_idx).inner_text().strip()
                        if len(text) > 30:
                            first_p = text
                            break

                    register_article(site_name, {
                        "title": item_info["title"], "url": item_info["url"],
                        "published_date": pub_date, "first_paragraph": first_p,
                        "scraped_at": datetime.now(timezone.utc).isoformat()
                    })
                except Exception:
                    continue
            print(f"🏁 {site_name}: collected {len(articles_data)}/{TARGET_COUNT} new articles.")
            browser.close()
    except Exception as e:
        print(f"⚠️ Exception in {site_name}: {e}")
    finally:
        if sb: sb.quit()

def scrape_unnews():
    site_name = "UN News"
    base_url = "https://news.un.org"
    news_url = "https://news.un.org/en/news"
    print(f"🚀 Starting {site_name} scraper...")
    sb = None
    try:
        sb = sb_cdp.Chrome(locale="en")
        sb.goto(news_url)
        time.sleep(3)
        endpoint_url = sb.get_endpoint_url()

        articles_data = []
        with sync_playwright() as p:
            browser = p.chromium.connect_over_cdp(endpoint_url)
            page = browser.contexts[0].pages[0]

            card_selector = "article.node--type-news-story"
            page_num = 0

            while len(articles_data) < TARGET_COUNT and page_num <= MAX_PAGES_SAFETY:
                current_url = f"{news_url}?page={page_num}"
                try:
                    page.goto(current_url, wait_until="domcontentloaded", timeout=12000)
                    page.wait_for_selector(card_selector, timeout=6000)
                except Exception:
                    break

                cards = page.locator(card_selector)
                count = cards.count()
                if count == 0:
                    break

                before = len(articles_data)
                for i in range(count):
                    if len(articles_data) >= TARGET_COUNT:
                        break
                    card = cards.nth(i)
                    link_el = card.locator(".node__title a").first
                    if link_el.count() > 0:
                        title = link_el.inner_text().strip()
                        href = link_el.get_attribute("href")
                        if href:
                            full_url = href if href.startswith("http") else f"{base_url}{href}"
                            if is_new_url(full_url) and not any(item["url"] == full_url for item in articles_data):
                                articles_data.append({"title": title, "url": full_url})

                if len(articles_data) >= TARGET_COUNT:
                    break
                if len(articles_data) == before:
                    break
                page_num += 1

            for item_info in articles_data:
                try:
                    page.goto(item_info["url"], wait_until="domcontentloaded", timeout=12000)
                    check_and_handle_captcha(page, site_name)
                    time_el = page.locator(".field--name-field-news-date time, time.datetime, time").first
                    pub_date = time_el.inner_text().strip() if time_el.count() > 0 else "N/A"

                    first_p = "N/A"
                    body_paras = page.locator(".field--name-field-text-column p, article p, main p")
                    for p_idx in range(min(body_paras.count(), 5)):
                        text = body_paras.nth(p_idx).inner_text().strip()
                        if len(text) > 30:
                            first_p = text
                            break

                    register_article(site_name, {
                        "title": item_info["title"], "url": item_info["url"],
                        "published_date": pub_date, "first_paragraph": first_p,
                        "scraped_at": datetime.now(timezone.utc).isoformat()
                    })
                except Exception:
                    continue
            print(f"🏁 {site_name}: collected {len(articles_data)}/{TARGET_COUNT} new articles.")
            browser.close()
    except Exception as e:
        print(f"⚠️ Exception in {site_name}: {e}")
    finally:
        if sb: sb.quit()

# =============================================================
# MAIN CONTROLLER
# =============================================================
if __name__ == "__main__":
    print("=====================================================")
    print("🌐 Launching Grouped JSON + Postgres Scraper with Interactive CAPTCHA Guard")
    print(f"🎯 Per-site article limit: {TARGET_COUNT}")
    print("=====================================================\n")

    init_db()
    load_existing_data()

    scrapers = [
        scrape_aapnews, scrape_aljazeera, scrape_arabnews, scrape_bloomberg,
        scrape_cbsnews, scrape_tribune, scrape_reuters, scrape_straitstimes,
        scrape_newrepublic, scrape_unnews
    ]

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_SCRAPERS) as executor:
        futures = {executor.submit(func): func.__name__ for func in scrapers}
        for future in as_completed(futures):
            func_name = futures[future]
            try:
                future.result()
                print(f"✅ {func_name} completed.")
            except Exception as exc:
                print(f"❌ {func_name} failed: {exc}")

    save_checkpoint_json()
    close_db()
    duration = round(time.time() - start_time, 2)

    total_items_saved = sum(len(v) for v in master_articles_dict.values() if isinstance(v, list))

    print("\n=====================================================")
    print(f"🎉 Process completed in {duration} seconds!")
    print(f"📁 Total unique items stored: {total_items_saved}")
    print(f"📄 Grouped JSON output saved to: {os.path.abspath(JSON_FILE)}")
    if db_pool is not None:
        print(f"🗄️  Articles also mirrored into Postgres table 'articles' (db: {PG_DATABASE})")
    else:
        print(f"⚠️  Postgres was unavailable this run — articles were saved to JSON only.")
    print("=====================================================")