import time
import random
import logging
import requests
from typing import Optional
from google import genai
import config

def call_local_llm(prompt: str) -> Optional[str]:
    """
    Calls local Ollama instance as a fallback.
    Returns None if Ollama is not accessible or model is missing.
    """
    try:
        logging.info(f"Attempting local LLM fallback using model: {config.OLLAMA_MODEL}")
        response = requests.post(
            config.OLLAMA_API_URL,
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=45
        )
        if response.status_code == 200:
            return response.json().get("response")
        else:
            logging.warning(f"Local LLM fallback failed with status code {response.status_code}: {response.text}")
            return None
    except Exception as e:
        logging.warning(f"Local LLM fallback unreachable or failed: {e}")
        return None

def call_gemini_with_backoff(
    client: Optional[genai.Client],
    model: str,
    contents: str,
    max_retries: int = config.MAX_RETRIES,
    backoff_factor: float = config.BACKOFF_FACTOR,
    delay_before: float = 0.0
) -> str:
    """
    Calls the Gemini API with optional pre-call delay and exponential backoff retry logic for 429/transient errors.
    Falls back to Ollama if client is None or if retries are exhausted.
    Raises RuntimeError if both Gemini and local fallback fail.
    """
    if not client:
        if config.USE_LOCAL_FALLBACK:
            local_res = call_local_llm(contents)
            if local_res:
                return local_res
        raise RuntimeError("Gemini client is not initialized and local fallback failed.")

    if delay_before > 0:
        time.sleep(delay_before)

    retries = 0
    while retries < max_retries:
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents
            )
            if response and response.text:
                return response.text
            raise ValueError("Empty response received from Gemini API.")
        except Exception as e:
            is_retryable = False
            err_msg = str(e).lower()
            if any(x in err_msg for x in ["429", "503", "500", "504", "resource_exhausted", "rate limit", "unavailable", "demand", "temporary", "quota"]):
                is_retryable = True

            if is_retryable and retries < max_retries - 1:
                sleep_time = (backoff_factor ** retries) + random.uniform(0, 1)
                logging.warning(f"Gemini API experiencing transient error/demand. Retrying in {sleep_time:.2f} seconds... (Attempt {retries + 1}/{max_retries})")
                time.sleep(sleep_time)
                retries += 1
            else:
                logging.warning(f"Gemini API call failed: {e}")
                if config.USE_LOCAL_FALLBACK:
                    local_res = call_local_llm(contents)
                    if local_res:
                        logging.info("Successfully recovered using Local LLM fallback.")
                        return local_res
                raise RuntimeError(f"Gemini API failed and local fallback unavailable: {e}") from e

    raise RuntimeError("Max retries exceeded for Gemini API call")

def summarize_text(client: Optional[genai.Client], text: str, query: str, model: str = config.DEFAULT_MODEL) -> Optional[str]:
    """
    Summarizes the scraped page text with respect to the user's research query.
    Returns clean summary string, or None if summarization fails.
    """
    if not text or not text.strip():
        return None

    prompt = f"""
You are an expert research analyst.
Below is the raw text scraped from a webpage.
Please summarize the key points in this text that are relevant to the query: "{query}".

Guidelines:
- Focus only on factual details relevant to the query.
- Keep the summary concise but informative (max 300 words).
- If the text is completely irrelevant to the query, state: "No relevant information found."

Webpage Content:
{text}
"""
    try:
        return call_gemini_with_backoff(client, model, prompt, delay_before=0.5)
    except Exception as e:
        logging.error(f"Failed to summarize text: {e}")
        return None
