from bs4 import BeautifulSoup
from typing import Optional, List, Dict
import logging
import config

def scrape_page(url: str, timeout_seconds: int = config.REQUEST_TIMEOUT) -> Optional[str]:
    """
    Fetches the HTML of a URL using Playwright (headless) and extracts clean text.
    Strips script, style, nav, footer, header, aside, form, iframe, noscript.
    Falls back to requests + bs4 if Playwright is unavailable or times out.
    """
    results = scrape_urls([url], timeout_seconds=timeout_seconds)
    return results.get(url)

def scrape_urls(urls: List[str], timeout_seconds: int = config.REQUEST_TIMEOUT) -> Dict[str, Optional[str]]:
    """
    Scrapes multiple URLs sharing a single Chromium browser process for efficiency.
    Falls back to requests for any URL where Playwright fails.
    """
    results: Dict[str, Optional[str]] = {}
    timeout_ms = timeout_seconds * 1000

    playwright_available = False
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                context = browser.new_context(
                    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
                for url in urls:
                    try:
                        page = context.new_page()
                        try:
                            response = page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
                            if response:
                                content_type = response.headers.get("content-type", "").lower()
                                if content_type and not any(t in content_type for t in ["html", "text", "xml"]):
                                    logging.warning(f"[Scraper] Skipping non-HTML content-type '{content_type}' for {url}")
                                    results[url] = None
                                    continue
                            content = page.content()
                            results[url] = _clean_html_text(content)
                        finally:
                            page.close()
                    except Exception as e:
                        logging.warning(f"[Scraper] Playwright scrape failed for {url}: {e}")
                        results[url] = _fallback_requests_scrape(url, timeout_seconds)
                playwright_available = True
            finally:
                browser.close()
    except Exception as e:
        logging.warning(f"[Scraper] Playwright unavailable or batch failed ({e}). Using requests fallback...")

    if not playwright_available:
        for url in urls:
            if url not in results:
                results[url] = _fallback_requests_scrape(url, timeout_seconds)

    return results

def _fallback_requests_scrape(url: str, timeout_seconds: int) -> Optional[str]:
    try:
        import requests
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=timeout_seconds)
        if response.status_code == 200:
            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and not any(t in content_type for t in ["html", "text", "xml"]):
                logging.warning(f"[Scraper] Skipping non-HTML fallback content-type '{content_type}' for {url}")
                return None
            return _clean_html_text(response.text)
    except Exception as fallback_e:
        logging.warning(f"[Scraper] Fallback requests scrape failed for {url}: {fallback_e}")
    return None

def _clean_html_text(html_content: str) -> Optional[str]:
    """
    Cleans HTML and extracts structured text up to 15000 characters.
    """
    if not html_content:
        return None
    soup = BeautifulSoup(html_content, "html.parser")
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]):
        element.decompose()

    text = soup.get_text(separator="\n")
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    clean_text = "\n".join(chunk for chunk in chunks if chunk)
    return clean_text[:15000] if clean_text else None
