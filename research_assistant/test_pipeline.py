import sys
import os
import re

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from searcher import search_web
from scraper import scrape_page
from summarizer import summarize_text
from reporter import generate_report, save_pdf, markdown_to_html
from main import slugify

def test_pipeline():
    print("Testing fixed pipeline components...")

    # Test 1: Slugify Edge Case
    print("\n--- Test 1: Slugify Edge Case ('???') ---")
    slug = slugify("???")
    print(f"Slugify('???') -> {repr(slug)}")
    assert slug == "arastirma-raporu", f"Expected 'arastirma-raporu', got {repr(slug)}"
    print("Slugify edge case passed!")

    # Test 2: Markdown to HTML Ordered List Fix
    print("\n--- Test 2: Markdown to HTML Ordered List Fix ---")
    md = "1. Birinci Madde\n2. İkinci Madde"
    html_out = markdown_to_html(md)
    print("Markdown input:\n", repr(md))
    print("HTML output:\n", repr(html_out))
    assert "<ol>" in html_out and "<li>Birinci Madde</li>" in html_out, "Ordered list HTML conversion failed!"
    print("Markdown ordered list conversion passed!")

    # Test 3: Search Web (with query expansion / interleave)
    print("\n--- Test 3: DuckDuckGo Search ---")
    query = "Python programming"
    results = search_web(client=None, query=query, max_results=2)
    print(f"Search results found: {len(results)}")
    for r in results:
        print(f"Title: {r['title']}")
        print(f"URL: {r['url']}")

    # Test 4: Scrape
    print("\n--- Test 4: Scraper ---")
    if results:
        target_url = results[0]['url']
        print(f"Scraping: {target_url}")
        content = scrape_page(target_url, timeout_seconds=10)
        if content:
            print(f"Scraped content length: {len(content)} chars")
            print(f"Preview: {content[:150]}...")
        else:
            print("Failed to scrape target URL")
    else:
        print("No search results to scrape, testing example.com...")
        content = scrape_page("https://example.com", timeout_seconds=10)
        if content:
            print(f"Scraped example.com content length: {len(content)} chars")

    # Test 5: PDF Generation with Turkish Characters
    print("\n--- Test 5: PDF Generation ---")
    dummy_markdown = """# Türkçe Test Raporu
## Giriş
Bu bir Türkçe test raporudur. Ğ, Ş, İ, Ö, Ü, Ç harfleri test edilmektedir [1].

## Sıralı Adımlar
1. İlk adım ve analiz [1]
2. İkinci adım ve doğrulama [2]

## Kaynakça
[1] Python Resmi Web Sitesi - https://python.org
[2] Django Web Framework - https://djangoproject.com
"""
    pdf_path = "test_report_output.pdf"
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    success = save_pdf(dummy_markdown, pdf_path)
    if success and os.path.exists(pdf_path):
        print(f"PDF successfully generated at {pdf_path}")
        os.remove(pdf_path)
    else:
        print("PDF generation FAILED")
        sys.exit(1)

    print("\nPipeline components test completed successfully!")

if __name__ == "__main__":
    test_pipeline()
