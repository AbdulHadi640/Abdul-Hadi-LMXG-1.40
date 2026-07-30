import os
import json
import time
import threading
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from bs4 import BeautifulSoup
from seleniumbase import sb_cdp
from playwright.sync_api import sync_playwright

# =============================================================
# GLOBAL CONFIGURATION & CACHE
# =============================================================
JSON_FILE = "scraped_articles_grouped.json"
TARGET_COUNT = 150
SAVE_INTERVAL = 10        
MAX_PARALLEL_SCRAPERS = 2 

file_lock = threading.Lock()
master_articles_dict = {}
scraped_urls_set = set()
checkpoint_counter = 0

def load_existing_data():
    """Loads existing JSON file structured by news web names to populate memory caches."""
    global master_articles_dict, scraped_urls_set
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, "r", encoding="utf-8") as f:
                master_articles_dict = json.load(f)
                total_loaded = 0
                for source_name, articles in master_articles_dict.items():
                    if isinstance(articles, list):
                        for item in articles:
                            if "url" in item:
                                scraped_urls_set.add(item["url"])
                                total_loaded += 1
            print(f"📦 Loaded {total_loaded} existing articles across {len(master_articles_dict)} sources from {JSON_FILE}.")
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

def register_article(source_name, article_dict):
    """Thread-safe registration grouping items under their respective news web name key."""
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

            while collected < TARGET_COUNT:
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

                    if not link or link in scraped_urls_set:
                        continue

                    author_el = box.locator(".sri-authors.al-authors-list")
                    author = author_el.inner_text().strip() if author_el.count() > 0 else "N/A"

                    date_el = box.locator(".sf-facet-display")
                    pub_date = date_el.inner_text().replace("Published:", "").strip() if date_el.count() > 0 else "N/A"

                    abstract_el = box.locator(".abstract-response-placeholder")
                    abstract = abstract_el.inner_text().strip() if abstract_el.count() > 0 else "N/A"

                    item = {
                        "title": title, "url": link,
                        "author": author, "published_date": pub_date,
                        "first_paragraph": abstract, "scraped_at": datetime.now(timezone.utc).isoformat()
                    }
                    if register_article(site_name, item):
                        collected += 1

                current_page += 1
                time.sleep(1)
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

            retry_count = 0
            while page.locator(article_selector).count() < TARGET_COUNT:
                if retry_count > 5:
                    break
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(1)
                btn = page.locator("button.show-more-button").first
                if btn.count() > 0 and btn.is_visible():
                    try:
                        btn.click(force=True)
                        time.sleep(2)
                        retry_count = 0
                    except Exception:
                        retry_count += 1
                else:
                    retry_count += 1

            cards = page.locator(article_selector)
            limit = min(cards.count(), TARGET_COUNT)
            links = []

            for i in range(limit):
                card = cards.nth(i)
                link_el = card.locator("a.article-card__link, h2.article-card__title a").first
                if link_el.count() > 0:
                    title = link_el.inner_text().strip()
                    href = link_el.get_attribute("href")
                    if href:
                        url = href if href.startswith("http") else f"{base_url}{href}"
                        if url not in scraped_urls_set:
                            links.append({"title": title, "url": url})

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

            while len(articles_data) < TARGET_COUNT:
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
                            if full_url not in scraped_urls_set and not any(item["url"] == full_url for item in articles_data):
                                articles_data.append({"title": title if title else "N/A", "url": full_url})
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

            # FIX: Check captcha immediately upon loading Bloomberg
            check_and_handle_captcha(page, site_name)
            card_selector = "div.Latest_storyPadding__GBJUE"
            try: page.wait_for_selector(card_selector, timeout=15000)
            except Exception: pass

            stuck_count = 0
            last_card_count = 0

            while page.locator(card_selector).count() < TARGET_COUNT:
                current_cards = page.locator(card_selector).count()
                
                if current_cards > last_card_count:
                    last_card_count = current_cards
                    stuck_count = 0
                else:
                    stuck_count += 1
                    if stuck_count >= 5:
                        break
                    time.sleep(3)

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
                
                # FIX: Actively check for CAPTCHAs during infinite scrolling/loading loops
                check_and_handle_captcha(page, site_name)

                if click_success:
                    time.sleep(3)
                    page.evaluate("window.scrollBy(0, 300);")
                else:
                    time.sleep(2)

            cards = page.locator(card_selector)
            limit = min(cards.count(), TARGET_COUNT)
            news_links = []

            for i in range(limit):
                card = cards.nth(i)
                try:
                    headline_el = card.locator('[data-testid="headline"]').first
                    link_el = card.locator("a.Latest_storyLink__80QVD, a[href*='/news/']").first
                    if headline_el.count() > 0 and link_el.count() > 0:
                        title = headline_el.inner_text().strip()
                        url = link_el.get_attribute("href")
                        if url:
                            if url.startswith("/"):
                                url = "https://www.bloomberg.com" + url
                            if url not in scraped_urls_set and not any(item["url"] == url for item in news_links):
                                news_links.append({"title": title, "url": url})
                except Exception:
                    continue

            for article in news_links:
                try:
                    page.goto(article["url"], wait_until="domcontentloaded", timeout=15000)
                    # FIX: Check captcha on individual article sub-pages
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

                    if link and link.startswith("http") and link not in scraped_urls_set:
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

            while len(articles_data) < TARGET_COUNT:
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
                            if full_url not in scraped_urls_set and not any(item["url"] == full_url for item in articles_data):
                                articles_data.append({"title": title, "url": full_url})
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

        stuck_counter = 0
        previous_count = 0

        while True:
            sb.scroll_to_bottom()
            time.sleep(1)
            list_elements = sb.find_elements('ul[data-testid="FeedList"] > li')
            current_count = len(list_elements)

            if current_count >= TARGET_COUNT:
                break

            if current_count == previous_count:
                stuck_counter += 1
                if stuck_counter >= 3:
                    break
            else:
                stuck_counter = 0
                previous_count = current_count

            try:
                load_more_btn = sb.find_element('button[data-testid="FeedContentLoadMore"]', timeout=2)
                if load_more_btn:
                    sb.scroll_into_view('button[data-testid="FeedContentLoadMore"]')
                    time.sleep(0.5)
                    load_more_btn.click()
                    time.sleep(2)
            except Exception:
                break

        list_elements = sb.find_elements('ul[data-testid="FeedList"] > li')
        article_links = []
        for element in list_elements[:TARGET_COUNT]:
            if element:
                try:
                    title_a = element.querySelector('div[data-testid="Title"] a')
                    if title_a:
                        text = title_a.text.strip()
                        href = title_a.get_attribute("href")
                        full_link = href if href.startswith("http") else f"{base_url}{href}"
                        if full_link not in scraped_urls_set:
                            article_links.append({"title": text, "url": full_link})
                except Exception:
                    continue

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

            retry_count = 0
            while page.locator(card_selector).count() < TARGET_COUNT:
                if retry_count >= 5:
                    break
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(1)
                load_btn = page.locator("button:has-text('Load more')").first
                if load_btn.count() > 0 and load_btn.is_visible():
                    try:
                        load_btn.click(force=True)
                        time.sleep(2)
                        retry_count = 0
                    except Exception:
                        retry_count += 1
                else:
                    retry_count += 1

            cards = page.locator(card_selector)
            limit = min(cards.count(), TARGET_COUNT)
            articles_data = []

            for i in range(limit):
                card = cards.nth(i)
                link_el = card.locator("a[data-testid='custom-link']").first
                heading_el = card.locator("h4[data-testid='heading-test-id']").first

                if link_el.count() > 0:
                    title = heading_el.inner_text().strip() if heading_el.count() > 0 else "N/A"
                    href = link_el.get_attribute("href")
                    if href:
                        full_url = href if href.startswith("http") else f"{base_url}{href}"
                        if full_url not in scraped_urls_set:
                            articles_data.append({"title": title, "url": full_url})

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

            while len(articles_data) < TARGET_COUNT:
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
                            if full_url not in scraped_urls_set and not any(item["url"] == full_url for item in articles_data):
                                articles_data.append({"title": title, "url": full_url})
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

            while len(articles_data) < TARGET_COUNT:
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
                            if full_url not in scraped_urls_set and not any(item["url"] == full_url for item in articles_data):
                                articles_data.append({"title": title, "url": full_url})
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
    print("🌐 Launching Grouped JSON Scraper with Interactive CAPTCHA Guard")
    print("=====================================================\n")

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
    duration = round(time.time() - start_time, 2)

    total_items_saved = sum(len(v) for v in master_articles_dict.values() if isinstance(v, list))

    print("\n=====================================================")
    print(f"🎉 Process completed in {duration} seconds!")
    print(f"📁 Total unique items stored: {total_items_saved}")
    print(f"📄 Grouped JSON output saved to: {os.path.abspath(JSON_FILE)}")
    print("=====================================================")