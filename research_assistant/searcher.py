import logging
import time
from typing import List, Dict, Optional
import ssl
from google import genai
import config
from summarizer import call_gemini_with_backoff, call_local_llm

# Runtime patch to bypass LibreSSL ValueError on macOS when forcing TLS 1.3
try:
    original_minimum_version_setter = ssl.SSLContext.minimum_version.__set__
    def patched_minimum_version_setter(self, value):
        try:
            original_minimum_version_setter(self, value)
        except ValueError as e:
            if "0x304" in str(e) or "TLSv1_3" in str(e):
                pass
            else:
                raise e
    ssl.SSLContext.minimum_version = property(
        ssl.SSLContext.minimum_version.fget,
        patched_minimum_version_setter,
        ssl.SSLContext.minimum_version.fdel
    )
except Exception:
    pass

try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS

def expand_query(client: Optional[genai.Client], query: str, model: str = config.DEFAULT_MODEL) -> str:
    """
    Translates and expands a Turkish search query into English search keywords using Gemini with retry/backoff.
    """
    prompt = f"""
Translate and expand the following Turkish search query into 2-3 relevant English search keywords or a short search phrase.
Provide ONLY the English search keywords/phrase, with no extra text or explanation.

Turkish query: {query}
"""
    try:
        expanded = call_gemini_with_backoff(client, model, prompt, delay_before=0.0)
        if expanded:
            expanded = expanded.strip().replace('"', '').replace("'", "")
            return expanded
    except Exception as e:
        logging.warning(f"Query expansion via Gemini failed ({e}), trying local fallback or keeping original.")
        if config.USE_LOCAL_FALLBACK:
            expanded = call_local_llm(prompt)
            if expanded:
                return expanded.strip().replace('"', '').replace("'", "")
    return query

def search_web(client: Optional[genai.Client] = None, query: str = "", model: str = config.DEFAULT_MODEL, max_results: int = config.MAX_SEARCH_RESULTS) -> List[Dict[str, str]]:
    """
    Searches the web using DuckDuckGo for both Turkish and English queries.
    Interleaves and deduplicates results from both queries up to max_results.
    """
    english_query = expand_query(client, query, model)
    logging.info(f"Expanded query: {english_query}")

    queries = [query]
    if english_query and english_query.lower() != query.lower():
        queries.append(english_query)

    query_results_list = []
    for q in queries:
        q_results = []
        try:
            time.sleep(1.0)
            with DDGS() as ddgs:
                ddgs_generator = ddgs.text(q, max_results=max_results)
                if ddgs_generator:
                    for r in ddgs_generator:
                        url = r.get("href", "")
                        if url:
                            q_results.append({
                                "title": r.get("title", ""),
                                "url": url,
                                "snippet": r.get("body", "")
                            })
        except Exception as e:
            logging.error(f"Search failed for query '{q}': {e}")
        query_results_list.append(q_results)

    # Interleave results round-robin across queries and deduplicate by URL
    combined_results = []
    seen_urls = set()
    max_len = max((len(qr) for qr in query_results_list), default=0)
    for i in range(max_len):
        for q_results in query_results_list:
            if i < len(q_results):
                item = q_results[i]
                url = item["url"]
                if url not in seen_urls:
                    seen_urls.add(url)
                    combined_results.append(item)
                    if len(combined_results) >= max_results:
                        return combined_results

    return combined_results[:max_results]
