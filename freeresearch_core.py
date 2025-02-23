#!/usr/bin/env python3

import warnings
import logging
import sys
import io
import os
import datetime
import json
import asyncio
import aiohttp
import numpy as np
import requests
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from dotenv import load_dotenv
import autogen
from openai import OpenAI
import faiss
import networkx as nx
from concurrent.futures import ThreadPoolExecutor
from ratelimit import limits, sleep_and_retry, RateLimitException
from cachetools import TTLCache, LRUCache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
from threading import Lock
from crawl4ai import WebCrawler, AsyncWebCrawler
from crawl4ai.models import CrawlResult
from PyPDF2 import PdfReader
import re
import urllib.parse
from io import BytesIO
from contextlib import asynccontextmanager

# ----------------------- Attempt local NER (spacy) -----------------------
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    _HAVE_SPACY = True
except ImportError:
    _HAVE_SPACY = False

# ----------------------- YouTube Transcript Integration -----------------------
import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi

def is_youtube_url(url: str) -> bool:
    return "youtube.com/watch" in url or "youtu.be/" in url

def fetch_youtube_transcript(video_url: str) -> str:
    """
    Fetches the transcript for a YouTube video using the YouTube Transcript API.
    Returns a single string containing the transcript text.
    """
    try:
        video_id = None
        if "youtube.com/watch" in video_url:
            video_id = video_url.split("v=")[-1].split("&")[0]
        elif "youtu.be/" in video_url:
            video_id = video_url.split("youtu.be/")[-1].split("?")[0]
        if not video_id:
            raise ValueError("Invalid YouTube URL format")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript_list])
        return transcript_text
    except Exception as e:
        logging.error(f"Error fetching transcript for video {video_url}: {e}")
        return ""

# -------------------------------------------------------------------
# Global Token Tracking and Utility Function
# -------------------------------------------------------------------
total_tokens_fetched = 0

def count_tokens(text: str) -> int:
    """
    Approximate token count by assuming one token per 2.7 characters.
    """
    return int(len(text) / 2.7)

# -------------------------------------------------------------------
# Custom Exception for Token Limit Exceeded
# -------------------------------------------------------------------
class TokenLimitExceededException(Exception):
    pass

# -------------------------------------------------------------------
# Monkey patch to implement retries and avoid gemini server side errors
# -------------------------------------------------------------------
from autogen.oai import gemini
_original_gemini_create = gemini.GeminiClient.create

def gemini_create_with_retry(self, *args, **kwargs):
    max_retries = kwargs.get("max_retries", 3)
    retry_wait_time = kwargs.get("retry_wait_time", 10)
    attempts = 0
    wait_time = retry_wait_time
    while True:
        try:
            response = _original_gemini_create(self, *args, **kwargs)
            return response
        except Exception as e:
            err_str = str(e).lower()
            if (("500" in err_str or "503" in err_str or "504" in err_str or "overloaded" in err_str or
                 "remote end closed connection" in err_str or "remote disconnected" in err_str)
                and attempts < max_retries):
                attempts += 1
                logging.warning(
                    f"Gemini API call failed with {e}. Retrying in {wait_time} seconds (attempt {attempts}/{max_retries})..."
                )
                time.sleep(wait_time)
                wait_time *= 2
                continue
            else:
                raise Exception(f"Gemini API call failed after {attempts} attempts: {e}")

gemini.GeminiClient.create = gemini_create_with_retry
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# RAG + LanceDB Imports (Using SentenceTransformers as fallback embeddings)
# -------------------------------------------------------------------
try:
    import lancedb
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Please install lancedb and sentence_transformers before running this script.")
    sys.exit(1)

_fallback_embedding_model = SentenceTransformer('all-mpnet-base-v2')

# ----------------------- Jina Integration Flags -----------------------
USE_JINA_SERVICES = True  # If True, attempts to call Jina endpoints, no API key used.

# -------------------------------------------------------------------
# Environment Setup and Configuration
# -------------------------------------------------------------------
load_dotenv()
#openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    logging.error("GEMINI_API_KEY not set in environment variables.")
    sys.exit(1)

import google.generativeai as genai
genai.configure(api_key=gemini_api_key, transport="rest")
warnings.filterwarnings(
    "ignore",
    message="temperature is not supported with o1-mini model and will be ignored.",
    category=UserWarning,
    module="autogen.oai.client"
)

# -------------------------------------------------------------------
# Core Utility Functions and Classes
# -------------------------------------------------------------------
def get_current_utc_timestamp() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

class FilteredStdOut(io.StringIO):
    def write(self, s: str) -> None:
        if not isinstance(s, str):
            return
        if 'params=' in s:
            return
        sys.__stdout__.write(s)

sys.stdout = FilteredStdOut()
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logging.getLogger("autogen").setLevel(logging.ERROR)

@dataclass
class ResourceStats:
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    api_calls: int = 0
    active_tasks: int = 0

# -------------------------------------------------------------------
# Resource Manager with Lazy Semaphore Creation
# -------------------------------------------------------------------
class ResourceManager:
    def __init__(self, max_concurrent_tasks: int = 5):
        self.stats = ResourceStats()
        self.max_concurrent_tasks = max_concurrent_tasks
        self._lock = Lock()
        self.semaphore = None  # Will be created on first use

    @asynccontextmanager
    async def allocate_resource(self, timeout: int = 30):
        current_loop = asyncio.get_running_loop()
        if self.semaphore is None or getattr(self.semaphore, "_loop", None) is not current_loop:
            self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
        try:
            async with asyncio.timeout(timeout):
                async with self.semaphore:
                    with self._lock:
                        self.stats.active_tasks += 1
                    try:
                        yield
                    finally:
                        with self._lock:
                            self.stats.active_tasks -= 1
        except asyncio.TimeoutError:
            logging.error(f"Resource allocation timeout after {timeout} seconds")
            raise
        except Exception as e:
            logging.error(f"Error in resource allocation: {e}")
            raise

class CacheSystem:
    def __init__(self):
        self.memory_cache = TTLCache(maxsize=100, ttl=3600)
        self.persistent_cache = LRUCache(maxsize=1000)
        self.vector_cache = {}
        self.index = faiss.IndexFlatL2(768)
        self._lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        if not isinstance(key, str):
            raise ValueError("Cache key must be a string")
        with self._lock:
            return self.memory_cache.get(key) or self.persistent_cache.get(key)
    
    def set(self, key: str, value: Any, persistent: bool = False) -> None:
        if not isinstance(key, str):
            raise ValueError("Cache key must be a string")
        with self._lock:
            self.memory_cache[key] = value
            if persistent:
                self.persistent_cache[key] = value

# -------------------------------------------------------------------
# Named Entity Extraction (spaCy)
# -------------------------------------------------------------------
def extract_named_entities(text: str) -> Dict[str, int]:
    if not _HAVE_SPACY or not text:
        return {}
    doc = nlp(text)
    entity_counts = {}
    for ent in doc.ents:
        label = ent.label_
        entity_counts[label] = entity_counts.get(label, 0) + 1
    return entity_counts

class ContentProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self._lock = Lock()
        self._fitted = False
    
    def sanitize_content(self, content: str) -> str:
        if not isinstance(content, str):
            raise ValueError("Content must be a string")
        if not content:
            return ""
        return " ".join(content.split())
    
    def extract_key_info(self, content: str) -> Dict[str, Any]:
        if not isinstance(content, str):
            raise ValueError("Content must be a string")
        if not content:
            return {
                'summary': "",
                'length': 0,
                'word_count': 0,
                'sentence_count': 0,
                'entities': {}
            }
        try:
            sentences = content.split('.')
            words = content.split()
            entity_counts = extract_named_entities(content)
            return {
                'summary': ' '.join(sentences[:3]) if sentences else "",
                'length': len(content),
                'word_count': len(words),
                'sentence_count': len(sentences),
                'entities': entity_counts
            }
        except Exception as e:
            logging.error(f"Error extracting key info: {e}")
            return {'summary': "", 'length': 0, 'word_count': 0, 'sentence_count': 0, 'entities': {}}
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        A simple TF-IDF based similarity measure.
        Returns a float in [0, 1].
        """
        if not text1 or not text2:
            return 0.0
        try:
            with self._lock:
                # Reset vectorizer for each pair to ensure no leftover
                self.vectorizer = TfidfVectorizer()
                self.vectorizer.fit([text1, text2])
                vectors = self.vectorizer.transform([text1, text2])
                sim = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
                return float(sim)
        except Exception as e:
            logging.error(f"Error computing similarity: {e}")
            return 0.0

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_embeddings = {}
    
    def add_node(self, node_id: str, attributes: Dict[str, Any]):
        self.graph.add_node(node_id, **attributes)
    
    def add_relation(self, source: str, target: str, relation_type: str):
        self.graph.add_edge(source, target, relation=relation_type)

class LoggedList(list):
    color_mapping = {
        "DeepResearchAgent": "\033[94m",
        "AssessmentCommander": "\033[95m",
        "Summarizer": "\033[92m",
        "unknown": "\033[90m",
    }
    reset_color = "\033[0m"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = Lock()
        time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"conversation_log_{time_str}.json"
        try:
            self.log_file = open(self.log_filename, "a", encoding='utf-8')
        except Exception as e:
            logging.error(f"Failed to open log file: {e}")
            self.log_file = None
    
    def append(self, item: dict) -> None:
        if not isinstance(item, dict):
            logging.error("Invalid item type for LoggedList")
            return
        with self._lock:
            item['timestamp'] = datetime.datetime.now().isoformat()
            if 'sender' not in item:
                item['sender'] = item.get('name', 'unknown')
            if self.log_file:
                try:
                    json.dump(item, self.log_file)
                    self.log_file.write("\n")
                    self.log_file.flush()
                except Exception as e:
                    logging.error(f"Error logging item: {e}")
            super().append(item)
            try:
                sender = item.get('sender', 'unknown')
                color = self.color_mapping.get(sender, "\033[90m")
                separator = "=" * 80
                sys.__stdout__.write(f"\n{separator}\n")
                sys.__stdout__.write(f"{color}{sender} at {item['timestamp']}:\n")
                sys.__stdout__.write(f"{item.get('content','')}\n")
                sys.__stdout__.write(f"{self.reset_color}{separator}\n")
                sys.__stdout__.flush()
            except Exception as e:
                logging.error(f"Error writing to stdout: {e}")
    
    def __del__(self):
        if hasattr(self, 'log_file') and self.log_file:
            try:
                self.log_file.close()
            except Exception as e:
                logging.error(f"Error closing log file: {e}")

# Global conversation log for interactive chat
logs_data = LoggedList()

# Global fetched links data for tracking websites retrieved during research
fetched_links_data = []

# Global deep_research_flow session id
GLOBAL_SESSION_ID = None

# Global knowledge graph data for front-end consumption (session-specific)
knowledge_graph_data = {}

# Global persistent knowledge graph variables (aggregated across sessions)
persistent_knowledge_graph = KnowledgeGraph()
persistent_knowledge_graph_data = {}

# Store the last query node from the previous runs, and the subnodes from previous runs
last_query_node_global: Optional[str] = None
last_run_subnodes_global: List[str] = []

# Global search results filename (created once in run_full_research)
SEARCH_RESULTS_FILENAME = ""

def reset_globals():
    global logs_data, fetched_links_data, total_tokens_fetched
    global knowledge_graph_data, persistent_knowledge_graph_data
    global last_query_node_global, last_run_subnodes_global, SEARCH_RESULTS_FILENAME
    logs_data.clear()
    fetched_links_data.clear()  # clear the list in place
    total_tokens_fetched = 0
    knowledge_graph_data.clear()  # if this is a dict, use clear()
    persistent_knowledge_graph_data.clear()
    last_query_node_global = None
    last_run_subnodes_global.clear()  # assuming it's a list
    SEARCH_RESULTS_FILENAME = ""

# -------------------------------------------------------------------
# Rate Limit Constants for NO-API-KEY usage
# -------------------------------------------------------------------
JINA_EMBEDDING_CALLS = 500  # 500 RPM
JINA_RERANK_CALLS = 500     # 500 RPM
JINA_READER_CALLS = 20      # 20 RPM
JINA_DEEPSEARCH_CALLS = 2   # 2 RPM
JINA_SEGMENT_CALLS = 20     # 20 RPM

# -------------------------------------------------------------------
# Attempted Jina-based Web Reader (optional) - No API key
# -------------------------------------------------------------------
@sleep_and_retry
@limits(calls=JINA_READER_CALLS, period=60)
async def jina_read_url(url: str) -> str:
    """
    Use Jina's Reader endpoint to fetch main content from a URL in Markdown.
    Called WITHOUT an API key => 20 RPM limit.
    """
    if not USE_JINA_SERVICES:
        return ""
    target = f"https://r.jina.ai/{urllib.parse.quote(url, safe='')}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/markdown"
    }
    async with aiohttp.ClientSession() as session:
        retries = 3
        for attempt in range(1, retries + 1):
            try:
                async with session.get(target, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    else:
                        logging.warning(f"Jina Reader returned status {resp.status} for {url}.")
                        # If the status is not 200, retry if possible.
                        if attempt < retries:
                            await asyncio.sleep(min(2 ** attempt, 5))
                        else:
                            return ""
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logging.warning(f"Attempt {attempt}/{retries} for {url} failed with error: {e}")
                if attempt < retries:
                    await asyncio.sleep(min(2 ** attempt, 5))
                else:
                    return ""
            except Exception as e:
                logging.error(f"Unhandled error reading {url} via Jina Reader: {e}")
                return ""
    return ""

# -------------------------------------------------------------------
# Enhanced Web Crawler
# -------------------------------------------------------------------
class EnhancedWebCrawler:
    def __init__(self):
        self._sync_crawler: Optional[WebCrawler] = None
        self._async_crawler: Optional[AsyncWebCrawler] = None
        self.headers = {'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)'}
    
    async def _process_pdf(self, content: bytes, url: str) -> str:
        def parse_pdf() -> str:
            try:
                with BytesIO(content) as pdf_file:
                    reader = PdfReader(pdf_file)
                    texts = [page.extract_text() for page in reader.pages if page.extract_text()]
                    return "\n".join(texts) if texts else f"No text extracted from PDF: {url}"
            except Exception as e:
                logging.error(f"PDF processing error for {url}: {e}")
                return f"PDF processing error: {str(e)}"
        return await asyncio.to_thread(parse_pdf)

    def fetch_sync(self, url: str) -> str:
        global total_tokens_fetched
        logging.info(f"Fetching content (sync): {url}")
        try:
            if is_youtube_url(url):
                content = fetch_youtube_transcript(url)
                if not content:
                    content = f"Failed to fetch transcript for {url}"
            elif url.lower().endswith(".pdf"):
                response = requests.get(url, timeout=10, headers=self.headers)
                response.raise_for_status()
                content = asyncio.run(self._process_pdf(response.content, url))
            else:
                if USE_JINA_SERVICES:
                    # Attempt Jina Reader first
                    content = asyncio.run(jina_read_url(url))
                    if content:
                        tokens = count_tokens(content)
                        total_tokens_fetched += tokens
                        logging.info(f"[TOKENS] Fetched {tokens} tokens from {url} using Jina Reader. Total so far: {total_tokens_fetched}")
                        if total_tokens_fetched >= 920000:
                            termination_msg = (
                                "AssessmentCommander We need to terminate the Chat "
                                "Because Token Limit has been exceeded TERMINATE_CHAT"
                            )
                            logs_data.append({"sender": "System", "content": termination_msg})
                            raise TokenLimitExceededException(termination_msg)
                        return content
                # fallback local crawler
                if not self._sync_crawler:
                    self._sync_crawler = WebCrawler()
                result = self._sync_crawler.run(
                    url=url,
                    bypass_cache=True,
                    output="markdown",
                    extract_rules=["main_content"],
                    extractor="readability",
                    timeout=15
                )
                content = self._process_crawl_result(result, url)

            tokens = count_tokens(content)
            total_tokens_fetched += tokens
            logging.info(f"[TOKENS] Fetched {tokens} tokens from {url}. Total so far: {total_tokens_fetched}")
            if total_tokens_fetched >= 920000:
                termination_msg = (
                    "AssessmentCommander We need to terminate the Chat Because "
                    "Token Limit has been exceeded TERMINATE_CHAT"
                )
                logs_data.append({"sender": "System", "content": termination_msg})
                raise TokenLimitExceededException(termination_msg)
            return content
        except Exception as e:
            error_msg = f"Sync crawling error for {url}: {str(e)}"
            logging.error(error_msg)
            return error_msg

    async def fetch_async(self, session: aiohttp.ClientSession, url: str, retries: int = 3) -> str:
        global total_tokens_fetched
        logging.info(f"Async fetching: {url}")
        for attempt in range(1, retries + 1):
            try:
                if is_youtube_url(url):
                    loop = asyncio.get_running_loop()
                    content = await loop.run_in_executor(None, fetch_youtube_transcript, url)
                    if not content:
                        content = f"Failed to fetch transcript for {url}"
                elif url.lower().endswith(".pdf"):
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15), headers=self.headers) as response:
                        response.raise_for_status()
                        content_bytes = await response.read()
                        content = await self._process_pdf(content_bytes, url)
                else:
                    if USE_JINA_SERVICES:
                        content = await jina_read_url(url)
                        if content:
                            tokens = count_tokens(content)
                            total_tokens_fetched += tokens
                            logging.info(f"[TOKENS] Fetched {tokens} tokens from {url} using Jina Reader (async). Total so far: {total_tokens_fetched}")
                            if total_tokens_fetched >= 920000:
                                termination_msg = (
                                    "AssessmentCommander We need to terminate the Chat "
                                    "Because Token Limit has been exceeded TERMINATE_CHAT"
                                )
                                logs_data.append({"sender": "System", "content": termination_msg})
                                raise TokenLimitExceededException(termination_msg)
                            return content
                    async with AsyncWebCrawler() as crawler:
                        result = await crawler.arun(
                            url=url,
                            screenshot=False,
                            output="markdown",
                            extract_rules=["main_content"],
                            extractor="readability",
                            timeout=20
                        )
                        content = self._process_crawl_result(result, url)

                tokens = count_tokens(content)
                total_tokens_fetched += tokens
                logging.info(f"[TOKENS] Fetched {tokens} tokens from {url}. Total so far: {total_tokens_fetched}")
                if total_tokens_fetched >= 920000:
                    termination_msg = (
                        "AssessmentCommander We need to terminate the Chat Because "
                        "Token Limit has been exceeded TERMINATE_CHAT"
                    )
                    logs_data.append({"sender": "System", "content": termination_msg})
                    raise TokenLimitExceededException(termination_msg)
                return content
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logging.warning(f"Attempt {attempt}/{retries} failed for {url}: {e}")
                if attempt < retries:
                    await asyncio.sleep(min(2 ** attempt, 5))
            except Exception as e:
                error_msg = f"Critical async error for {url}: {str(e)}"
                logging.error(error_msg)
                break
        return f"Failed after {retries} attempts: {url}"

    def _process_crawl_result(self, result: CrawlResult, url: str) -> str:
        try:
            if hasattr(result, 'markdown') and result.markdown:
                return result.markdown
            for attr in ['text', 'content', 'cleaned_text']:
                if hasattr(result, attr):
                    content = getattr(result, attr)
                    if content:
                        return content
            return f"No content extracted from {url}"
        except Exception as e:
            logging.error(f"Error processing crawl result for {url}: {e}")
            return f"Error processing content: {str(e)}"

# -------------------------------------------------------------------
# Jina Rerank Support (anonymous => 500 RPM)
# -------------------------------------------------------------------
@sleep_and_retry
@limits(calls=JINA_RERANK_CALLS, period=60)
async def rerank_results(query: str, original_max_results: int, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Use Jina's /rerank endpoint to reorder results by relevance to the query.
    Called without an API key => 500 RPM.
    """
    if not USE_JINA_SERVICES:
        return results

    endpoint = "https://api.jina.ai/v1/rerank"
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "model": "jina-reranker-v2-base-multilingual",
        "query": query,
        "top_n": original_max_results,
        "documents": [r.get("snippet", "") for r in results]
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers, json=payload, timeout=30) as resp:
                if resp.status != 200:
                    logging.warning(f"Rerank API returned {resp.status}, skipping rerank.")
                    return results
                data = await resp.json()
                reranked_output = data.get("results", data.get("data", []))
                if not reranked_output or len(reranked_output) != len(results):
                    return results

                snippet_score_map = {}
                for entry in reranked_output:
                    document = entry.get("document", {})
                    snippet = document.get("text", "")
                    score = entry.get("relevance_score", entry.get("score", 0))
                    snippet_score_map[snippet] = score

                for res in results:
                    snip = res.get("snippet", "")
                    res["_jina_score"] = snippet_score_map.get(snip, 0)
                sorted_results = sorted(results, key=lambda r: r["_jina_score"], reverse=True)
                return sorted_results[:original_max_results]
    except Exception as e:
        logging.error(f"Error reranking results with Jina: {e}")
        return results

# -------------------------------------------------------------------
# Search Helpers
# -------------------------------------------------------------------
@sleep_and_retry
@limits(calls=1, period=2)
async def async_fetch_page_content(session: aiohttp.ClientSession, url: str, retries: int = 3) -> str:
    crawler = EnhancedWebCrawler()
    for attempt in range(1, retries + 1):
        try:
            return await crawler.fetch_async(session, url, retries)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logging.warning(f"Attempt {attempt}/{retries} for {url} failed with error: {e}")
            if attempt < retries:
                await asyncio.sleep(min(2 ** attempt, 5))
            else:
                return f"Failed after {retries} attempts: {url}"
        except Exception as e:
            logging.error(f"Unhandled error fetching {url}: {e}")
            return f"Error: {e}"

def fetch_page_content(url: str) -> str:
    crawler = EnhancedWebCrawler()
    return crawler.fetch_sync(url)

# -------------------------------------------------------------------
# Google Search Handler
# -------------------------------------------------------------------
class GoogleSearchHandler:
    def __init__(self, base_delay: float = 2.0):
        self.base_delay = base_delay
        self._last_request_time = 0
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def _build_google_url(self, query: str) -> str:
        encoded_query = urllib.parse.quote(query)
        return f"https://www.google.com/search?q={encoded_query}&hl=en"

    def _is_ad_link(self, element: Any) -> bool:
        ad_indicators = ['data-text-ad', 'data-dtld="ads"', 'class="ads-fr"', 'class="ads-ad"']
        element_str = str(element)
        return any(indicator in element_str for indicator in ad_indicators)

    async def _extract_search_results(self, html_content: str, max_results: int) -> List[Dict[str, Any]]:
        results = []
        soup = BeautifulSoup(html_content, 'html.parser')
        search_divs = soup.find_all('div', class_='g')
        for div in search_divs:
            if len(results) >= max_results:
                break
            if self._is_ad_link(div):
                continue
            try:
                link_elem = div.find('a')
                if not link_elem:
                    continue
                href = link_elem.get('href', '')
                if not href.startswith('http'):
                    continue
                href = re.sub(r'/url\?q=([^&]+).*', r'\1', href)
                title_elem = div.find('h3')
                title = title_elem.get_text() if title_elem else ''
                snippet_elem = div.find('div', class_='VwiC3b')
                snippet = snippet_elem.get_text() if snippet_elem else ''
                if href and title:
                    results.append({'title': title, 'href': href, 'snippet': snippet})
            except Exception as e:
                logging.error(f"Error extracting result: {e}")
                continue
        return results

    async def search(self, query: str, original_max_results: int = 5, max_results: int = 5) -> List[Dict[str, Any]]:
        search_url = self._build_google_url(query)
        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url=search_url, screenshot=False, output="html", timeout=20)
                if not result or not result.html:
                    logging.error(f"No HTML content retrieved for query: {query}")
                    return []
                results = await self._extract_search_results(result.html, max_results)
                for r in results:
                    r['source'] = r['href']
                    r['query'] = query
                # If you want to re-rank with Jina, you can enable the call below:
                # results = await rerank_results(query, original_max_results, results)
                return results
        except Exception as e:
            logging.error(f"Error performing Google search for query '{query}': {e}")
            return []

# -------------------------------------------------------------------
# Enhanced Web Search
# -------------------------------------------------------------------
async def deep_web_search(query: str, max_results: int = 5) -> list:
    try:
        search_handler = GoogleSearchHandler()
        results = await search_handler.search(query, original_max_results=max_results, max_results=max_results)
        if not results:
            return []
        tasks = []
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)'}
        async with aiohttp.ClientSession(headers=headers) as session:
            for res in results:
                url = res.get("href", "")
                if url:
                    tasks.append(asyncio.create_task(async_fetch_page_content(session, url)))
            contents = await asyncio.gather(*tasks, return_exceptions=True)
            for res, content in zip(results, contents):
                if isinstance(content, Exception):
                    if isinstance(content, TokenLimitExceededException):
                        raise content
                    logging.error(f"Error fetching content for {res.get('href', '')}: {content}")
                    res["full_content"] = ""
                else:
                    res["full_content"] = content
                res["fetched_at"] = get_current_utc_timestamp()
        return results
    except Exception as e:
        logging.error(f"Error in deep_web_search for query '{query}': {e}")
        raise e

# -------------------------------------------------------------------
# Optional Jina Deep Search (2 RPM no key)
# -------------------------------------------------------------------
@sleep_and_retry
@limits(calls=JINA_DEEPSEARCH_CALLS, period=60)
async def jina_deep_search(query: str) -> str:
    """
    Asynchronous version using Jina's DeepSearch API,
    returning text for the user query. No API key => 2 RPM limit.
    """
    if not USE_JINA_SERVICES:
        return ""
    endpoint = "https://deepsearch.jina.ai/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    payload = {
        "messages": [
            {"role": "user", "content": query}
        ]
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers, json=payload, timeout=500) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    choices = data.get("choices", [])
                    if choices:
                        message = choices[0].get("message", {})
                        return message.get("content", "")
                    else:
                        logging.warning("No choices returned in response")
                        return ""
                else:
                    logging.warning(f"DeepSearch API returned status {resp.status}: {await resp.text()}")
                    return ""
    except Exception as e:
        logging.error(f"Error with Jina DeepSearch: {e}")
        return ""

# -------------------------------------------------------------------
# Helper to parse the Jina deep search summary into main text + references
# -------------------------------------------------------------------
def parse_jina_summary(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Parses the Jina deep search result text.
    Returns a tuple:
      - main_summary: The text before the first footnote line (if any).
      - references: A list of dicts with keys {'ref_number', 'ref_text'}.
    """
    footnote_index = text.find("\n[^")
    if footnote_index == -1:
        footnote_index = text.find("[^")
    if footnote_index != -1:
        main_summary = text[:footnote_index].strip()
        footnotes_text = text[footnote_index:].strip()
    else:
        main_summary = text.strip()
        footnotes_text = ""

    references = []
    matches = re.findall(r'\[\^(\d+)\]:\s*(.+)', footnotes_text)
    for (ref_number, ref_text) in matches:
        references.append({
            "ref_number": ref_number,
            "ref_text": ref_text.strip()
        })
    return main_summary, references

# -------------------------------------------------------------------
# Link newly created nodes with existing nodes using content similarity
# -------------------------------------------------------------------
def link_new_subnodes_by_context(knowledge_graph: "KnowledgeGraph", new_nodes: List[str], threshold: float = 0.4):
    content_processor = ContentProcessor()
    # Filter existing nodes that are currently in the graph
    existing_nodes = [n for n in knowledge_graph.graph.nodes if n not in new_nodes]
    
    for new_node in new_nodes:
        if new_node not in knowledge_graph.graph:
            logging.warning(f"Node {new_node} not found in knowledge graph, skipping linking for it.")
            continue
        new_data = knowledge_graph.graph.nodes[new_node]
        new_text = (
            (new_data.get('title', '') or '') + ' ' +
            (new_data.get('summary', '') or '') + ' ' +
            (new_data.get('content', '') or '')
        ).strip()
        if not new_text:
            continue
        for existing_node in existing_nodes:
            if existing_node not in knowledge_graph.graph:
                logging.warning(f"Existing node {existing_node} not found, skipping linking.")
                continue
            existing_data = knowledge_graph.graph.nodes[existing_node]
            existing_text = (
                (existing_data.get('title', '') or '') + ' ' +
                (existing_data.get('summary', '') or '') + ' ' +
                (existing_data.get('content', '') or '')
            ).strip()
            if not existing_text:
                continue
            sim = content_processor.compute_similarity(new_text, existing_text)
            if sim > threshold:
                knowledge_graph.add_relation(
                    new_node,
                    existing_node,
                    f"context_overlap_{sim:.2f}"
                )


# -------------------------------------------------------------------
# Iterative Research Flow
# -------------------------------------------------------------------
async def deep_research_flow(query: str, depth: int, breadth: int, agent_name: str = "DeepResearchAgent") -> str:
    global GLOBAL_SESSION_ID, fetched_links_data, knowledge_graph_data, SEARCH_RESULTS_FILENAME, total_tokens_fetched
    global last_query_node_global, last_run_subnodes_global  # Access the global variables
    logging.info("Starting deep research flow...")
    context = []
    current_query = query
    knowledge_graph = KnowledgeGraph()
    cache_system = CacheSystem()

    if not SEARCH_RESULTS_FILENAME:
        ts = get_current_utc_timestamp()
        SEARCH_RESULTS_FILENAME = f"search_results_{ts}.txt"
        with open(SEARCH_RESULTS_FILENAME, "w", encoding='utf-8') as f:
            f.write(f"=== Research Session Started: {ts} ===\nSubject: {query}\n\n")
    
    GLOBAL_SESSION_ID = f"session_{int(time.time())}"
    session_id = GLOBAL_SESSION_ID

    prev_query_node = None

    # Keep track of all subnodes in this run
    this_run_subnodes = []

    for iteration in range(1, depth + 1):
        timestamp = get_current_utc_timestamp()
        logging.info(f"[{agent_name}] Iteration {iteration} at {timestamp}: Searching for '{current_query}'")
        cache_key = f"search:{current_query}:{breadth}"
        try:
            results = cache_system.get(cache_key)
            if not results:
                results = await deep_web_search(current_query, max_results=breadth)
                cache_system.set(cache_key, results)
        except TokenLimitExceededException as e:
            termination_msg = str(e)
            with open(SEARCH_RESULTS_FILENAME, "a", encoding='utf-8') as f:
                f.write("\nToken limit reached during fetch. Terminating research iterations.\n")
                f.write("\n" + termination_msg + "\n")
            logs_data.append({"sender": "System", "content": termination_msg})
            report = generate_research_report(context, knowledge_graph)
            final_output = termination_msg + "\n\n" + report
            with open(SEARCH_RESULTS_FILENAME, "a", encoding='utf-8') as f:
                f.write("\n=== Final Research Report ===\n")
                f.write(final_output)
                f.write("\n" + "="*80 + "\n")
            return final_output

        if not results:
            logging.warning(f"No valid results for query '{current_query}'; terminating loop.")
            break

        # Create a node for this iteration's query
        query_node_id = f"{session_id}_query_{iteration}"
        knowledge_graph.add_node(query_node_id, {"title": f"Query: {current_query}", "timestamp": timestamp})
        
        # If this is the first iteration in this run, link it to the last_query_node_global if any
        if iteration == 1 and last_query_node_global is not None:
            knowledge_graph.add_relation(last_query_node_global, query_node_id, "previous_run_query_sequence")

        # Link with the previous query node in this same run
        if prev_query_node:
            knowledge_graph.add_relation(prev_query_node, query_node_id, "next_query")
        prev_query_node = query_node_id
        
        # ----------------------------------------------------------
        # Insert Jina deep search integration (avoid duplicates).
        # ----------------------------------------------------------
        already_has_jina = any(r.get('source') == 'Jina Deep Search' for r in results)

        # Initialize references to None/empty to avoid NameError if Jina fails
        jina_summary_node_id = None
        references = []

        if not already_has_jina:
            jina_deep_text = await jina_deep_search(current_query)
            if jina_deep_text:
                jina_result = {
                    "title": "Jina Deep Search Summary",
                    "href": f"jina://deepsearch/{urllib.parse.quote(current_query)}",
                    "snippet": jina_deep_text[:300] + ("..." if len(jina_deep_text) > 300 else ""),
                    "full_content": jina_deep_text,
                    "source": "Jina Deep Search",
                    "query": current_query
                }
                # Append the Jina result as a “virtual” search outcome
                results.append(jina_result)

                # Parse the Jina text for references
                main_summary, references = parse_jina_summary(jina_deep_text)

                # Create knowledge graph node for the Jina summary
                jina_summary_node_id = f"{session_id}_jina_summary_{iteration}"
                knowledge_graph.add_node(
                    jina_summary_node_id,
                    {
                        "title": "Jina Deep Search Summary",
                        "snippet": main_summary[:200] + ("..." if len(main_summary) > 200 else ''),
                        "timestamp": timestamp,
                        "entities": {},
                        "summary": main_summary,
                        "content": main_summary
                    }
                )
                # Link the query node to the Jina summary node
                knowledge_graph.add_relation(query_node_id, jina_summary_node_id, "jina_summary")

                # For each reference, create sub-nodes & link them
                for ref in references:
                    ref_node_id = f"{session_id}_jina_ref_{iteration}_{ref['ref_number']}"
                    ref_text = ref['ref_text']
                    knowledge_graph.add_node(
                        ref_node_id,
                        {
                            "title": f"Jina Reference {ref['ref_number']}",
                            "snippet": ref_text[:200] + ("..." if len(ref_text) > 200 else ""),
                            "timestamp": timestamp,
                            "entities": {},
                            "summary": ref_text,
                            "content": ref_text
                        }
                    )
                    # Link the summary node -> reference node only
                    knowledge_graph.add_relation(jina_summary_node_id, ref_node_id, "reference")
                    this_run_subnodes.append(ref_node_id)

                # Also treat the Jina summary node as a subnode
                this_run_subnodes.append(jina_summary_node_id)

        content_processor = ContentProcessor()
        new_nodes_created_this_iter = [query_node_id]

        # Add each search result as a node with a safe, generated ID (do not use the URL as the node ID)
        for idx, result in enumerate(results):
            node_id = f"{session_id}_result_{iteration}_{idx}"
            snippet = result.get("snippet", "")
            full_content = result.get("full_content", "")
            combined_text = f"{snippet}\n\n{full_content}"
            info_dict = content_processor.extract_key_info(combined_text)
            node_attributes = {
                "title": result.get("title", ""),
                "url": result.get("href", ""),  # store the URL as an attribute
                "snippet": snippet,
                "timestamp": timestamp,
                "entities": info_dict.get("entities", {}),
                "summary": info_dict.get("summary", ""),
                "content": combined_text,
            }
            knowledge_graph.add_node(node_id, node_attributes)
            knowledge_graph.add_relation(query_node_id, node_id, "search_result")
            new_nodes_created_this_iter.append(node_id)
            this_run_subnodes.append(node_id)
        
        # Link new subnodes across iteration boundaries if textual context overlaps
        link_new_subnodes_by_context(knowledge_graph, new_nodes_created_this_iter, threshold=0.6)

        # Summaries for logging
        learnings = " ".join([res.get("snippet", "").strip() for res in results if res.get("snippet")])
        directions = " ".join([res.get("title", "").strip() for res in results if res.get("title")])
        context.append({
            "iteration": iteration,
            "query": current_query,
            "timestamp": timestamp,
            "results": results,
            "learnings": learnings,
            "directions": directions
        })
        
        with open(SEARCH_RESULTS_FILENAME, "a", encoding='utf-8') as f:
            f.write(f"\n=== Iteration {iteration} ===\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Query: {current_query}\n")
            f.write(f"Key Learnings: {learnings}\n")
            f.write(f"Explored Directions: {directions}\n")
            f.write("\n" + "="*80 + "\n")
        
        if total_tokens_fetched >= 920000:
            logging.info("Token limit reached after iteration logging. Terminating further research iterations.")
            with open(SEARCH_RESULTS_FILENAME, "a", encoding='utf-8') as f:
                f.write("\nToken limit reached. Terminating research iterations.\n")
            termination_msg = "AssessmentCommander We need to terminate the Chat Because Token Limit has been exceeded TERMINATE_CHAT"
            logs_data.append({"sender": "System", "content": termination_msg})
            report = generate_research_report(context, knowledge_graph)
            final_output = termination_msg + "\n\n" + report
            with open(SEARCH_RESULTS_FILENAME, "a", encoding='utf-8') as f:
                f.write("\n=== Final Research Report ===\n")
                f.write(final_output)
                f.write("\n" + "="*80 + "\n")
            return final_output
        
        for res in results:
            if res.get("href") and res.get("title"):
                fetched_links_data.append({"url": res.get("href"), "title": res.get("title")})
        
        global knowledge_graph_data
        knowledge_graph_data = {
            "nodes": [{"id": node, **data} for node, data in knowledge_graph.graph.nodes(data=True)],
            "links": [{"source": source, "target": target, **data} for source, target, data in knowledge_graph.graph.edges(data=True)]
        }
        logging.info(f"After iteration {iteration}, knowledge_graph_data nodes: " +
                     f"{[node for node, _ in knowledge_graph.graph.nodes(data=True)]}")
        
        if iteration < depth:
            current_query = generate_next_query(current_query, learnings, directions)
    
    if knowledge_graph.graph.number_of_nodes() == 0:
        knowledge_graph.add_node("dummy", {"title": "No search results found", "snippet": "", "timestamp": get_current_utc_timestamp()})
    
    report = generate_research_report(context, knowledge_graph)
    with open(SEARCH_RESULTS_FILENAME, "a", encoding='utf-8') as f:
        f.write("\n=== Final Research Report ===\n")
        f.write(report)
        f.write("\n" + "="*80 + "\n")
    
    # After finishing this run, let's link newly created subnodes in this run with last_run_subnodes_global
    # to handle cross-run similarity.
    link_new_subnodes_by_context(knowledge_graph, this_run_subnodes + last_run_subnodes_global, threshold=0.6)

    # Update the last_run_subnodes_global to be all the subnodes from this run for the next run
    last_run_subnodes_global = this_run_subnodes

    # Also update the global last_query_node_global to be the final query node of this run
    if prev_query_node:
        last_query_node_global = prev_query_node

    global persistent_knowledge_graph_data
    # Merge local knowledge graph into persistent_knowledge_graph
    for node, data in knowledge_graph.graph.nodes(data=True):
        if node not in persistent_knowledge_graph.graph:
            persistent_knowledge_graph.add_node(node, data)
    for source, target, data_e in knowledge_graph.graph.edges(data=True):
        if not persistent_knowledge_graph.graph.has_edge(source, target):
            persistent_knowledge_graph.add_relation(source, target, data_e.get("relation", ""))
    
    # Also link newly added nodes in the persistent graph with existing nodes in the persistent graph
    # to handle cross-run context
    new_nodes_all = list(knowledge_graph.graph.nodes())
    link_new_subnodes_by_context(persistent_knowledge_graph, new_nodes_all, threshold=0.6)

    persistent_knowledge_graph_data = {
        "nodes": [{"id": node, **data} for node, data in persistent_knowledge_graph.graph.nodes(data=True)],
        "links": [{"source": source, "target": target, **data} for source, target, data in persistent_knowledge_graph.graph.edges(data=True)]
    }
    
    logging.info("Final session knowledge_graph_data nodes: " +
                 f"{[node for node, _ in knowledge_graph.graph.nodes(data=True)]}")
    logging.info("Final persistent knowledge_graph_data nodes: " +
                 f"{[node for node, _ in persistent_knowledge_graph.graph.nodes(data=True)]}")
    logging.info("Sleeping for 15 seconds to avoid rate limit from Gemini")
    time.sleep(15)
    return report


def generate_next_query(current_query: str, learnings: str, directions: str) -> str:
    prompt = (
        "You are an insightful research assistant. Based on the current findings, generate a focused query to "
        "explore new aspects of the topic in no more than 20 words.\n\n"
        f"Current query: {current_query}\n"
        f"Learnings: {learnings}\n"
        f"Directions: {directions}\n\n"
        "Next query:"
    )
    model = genai.GenerativeModel(model_name="gemini-2.0-pro-exp-02-05")
    response = model.generate_content(prompt)
    new_query = response.text.strip() if response.text else ""
    logging.info(f"Generated next query: '{new_query}'")
    return new_query

def generate_research_report(context: List[Dict[str, Any]], knowledge_graph: KnowledgeGraph) -> str:
    report_lines = ["# Deep Research Report\n"]
    report_lines.append("## Research Overview")
    report_lines.append(f"Total Iterations: {len(context)}")
    if context:
        report_lines.append(f"Research Timeline: {context[0]['timestamp']} to {context[-1]['timestamp']}\n")
    for item in context:
        report_lines.append(f"## Iteration {item['iteration']} - Query: {item['query']}")
        report_lines.append(f"*Timestamp: {item['timestamp']}*\n")
        report_lines.append(f"**Key Learnings:** {item['learnings']}\n")
        report_lines.append(f"**Explored Directions:** {item['directions']}\n")
        report_lines.append("### Detailed Findings:")
        for result in item['results']:
            report_lines.append(f"- **Source:** {result.get('source', 'Unknown')}")
            report_lines.append(f"  - **Title:** {result.get('title', 'No Title')}")
            report_lines.append(f"  - **Summary:** {result.get('snippet', 'No Summary')}")
            if result.get('full_content'):
                content_str = result['full_content']
                report_lines.append(f"  - **Content:** {content_str}\n")
    """
    report_lines.append("## Knowledge Graph Analysis")
    nodes = knowledge_graph.graph.nodes(data=True)
    report_lines.append(f"\n### Key Sources ({len(nodes)} total)")
    for node, data in nodes:
        title = data.get('title', 'Untitled')
        snippet = data.get('snippet', '')
        summary = data.get('summary', '')
        entity_info = data.get('entities', {})
        report_lines.append(f"- **{title}** (Node ID: {node})")
        report_lines.append(f"  - Snippet: {snippet[:200]}{'...' if len(snippet) > 200 else ''}")
        report_lines.append(f"  - Summary: {summary}")
        if entity_info:
            report_lines.append(f"  - Extracted Entities: {entity_info}")
    edges = knowledge_graph.graph.edges(data=True)
    if edges:
        report_lines.append("\n### Content Relationships")
        for source, target, data_e in edges:
            report_lines.append(f"- {source} → {target}")
            if 'relation' in data_e:
                report_lines.append(f"  - Relation: {data_e['relation']}")
    report_lines.append("\n## Research Statistics")
    total_sources = len(set(result['source'] for item in context for result in item['results']))
    report_lines.append(f"- Total Unique Sources: {total_sources}")
    report_lines.append(f"- Total Iterations: {len(context)}")
    if context:
        avg_results = sum(len(item['results']) for item in context) / len(context)
        report_lines.append(f"- Average Results per Iteration: {avg_results:.2f}")
    report_lines.append("\n## References")
    all_sources = sorted(set(result['source'] for item in context for result in item['results']))
    for idx, source in enumerate(all_sources, 1):
        report_lines.append(f"{idx}. {source}")
    """
    return "\n".join(report_lines)

# -------------------------------------------------------------------
# RAG Utilities: Chunking, Indexing, and Retrieval with LanceDB
# -------------------------------------------------------------------
@sleep_and_retry
@limits(calls=JINA_EMBEDDING_CALLS, period=60)
async def jina_embedding(text: str) -> List[float]:
    """
    Calls Jina AI's embedding endpoint WITHOUT an API key => 500 RPM.
    If the service fails or times out, fallback to local embedding.
    """
    if not USE_JINA_SERVICES:
        return _fallback_embedding_model.encode(text).tolist()
    endpoint = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
    }
    payload = {
        "input": [text],
        "normalized": False
    }
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(endpoint, json=payload, headers=headers, timeout=30) as resp:
                if resp.status != 200:
                    logging.warning(f"Jina embedding API returned {resp.status}, using fallback local embedding.")
                    return _fallback_embedding_model.encode(text).tolist()
                data = await resp.json()
                emb = data["data"][0]["embedding"]
                return emb
        except Exception as e:
            logging.error(f"Error calling Jina embedding, fallback local: {e}")
            return _fallback_embedding_model.encode(text).tolist()

async def embed_text(text: str) -> np.ndarray:
    vec = await jina_embedding(text)
    return np.array(vec, dtype=np.float32)

def split_text_naive(text: str, max_length: int = 2000) -> List[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_length += len(word) + 1
        current_chunk.append(word)
        if current_length >= max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

@sleep_and_retry
@limits(calls=JINA_SEGMENT_CALLS, period=60)
async def jina_semantic_chunking(text: str) -> List[str]:
    """
    Use Jina's semantic chunker, no key => 20 RPM. 
    Fallback to naive if fails or service is unavailable.
    """
    if not USE_JINA_SERVICES:
        return split_text_naive(text, max_length=2000)
    endpoint = "https://api.jina.ai/v1/segment"
    headers = {
        "Content-Type": "application/json",
    }
    payload = {"text": text}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(endpoint, headers=headers, json=payload, timeout=30) as resp:
                if resp.status != 200:
                    logging.warning(f"Jina segment API returned {resp.status}, fallback to naive chunking.")
                    return split_text_naive(text, max_length=2000)
                data = await resp.json()
                chunks = data.get("chunks", [])
                if not chunks:
                    return split_text_naive(text, max_length=2000)
                return chunks
        except Exception as e:
            logging.error(f"Error with Jina segment API: {e}")
            return split_text_naive(text, max_length=2000)

def embed_and_index_document(text: str, lance_db_path: str = "tmp/lancedb", table_name: str = "research_chunks"):
    """
    Splits text using Jina chunker if available, calls embeddings,
    and stores in LanceDB.
    """
    chunks = asyncio.run(jina_semantic_chunking(text))
    records = []
    embeddings = asyncio.run(asyncio.gather(*(jina_embedding(chunk) for chunk in chunks)))
    for idx, (chunk, embedding_list) in enumerate(zip(chunks, embeddings)):
        embedding = np.array(embedding_list, dtype=np.float32)
        record = {"id": idx, "text": chunk, "vector": embedding}
        records.append(record)

    db = lancedb.connect(lance_db_path)
    try:
        table = db.open_table(table_name)
        table.add(records)
    except Exception:
        table = db.create_table(table_name, data=records)
    if hasattr(table, "commit"):
        table.commit()

def retrieve_relevant_chunks(query: str, lance_db_path: str = "tmp/lancedb", table_name: str = "research_chunks", top_n: int = 50) -> str:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        query_embedding_list = loop.run_until_complete(jina_embedding(query))
    finally:
        loop.close()
    query_embedding = np.array(query_embedding_list, dtype=np.float32)

    db = lancedb.connect(lance_db_path)
    table = db.open_table(table_name)
    results = table.search(query_embedding).limit(top_n).to_list()
    return "\n".join([record["text"] for record in results])

# -------------------------------------------------------------------
# LLM Configurations
# -------------------------------------------------------------------
config_list = [
    {'model': 'gpt-4o-mini', 'api_key': f'{openai_api_key}'},
    {'model': 'gpt-4o', 'api_key': f'{openai_api_key}'},
    {'model': 'o1-mini', 'api_key': f'{openai_api_key}'},
    {'model': 'o1-preview', 'api_key': f'{openai_api_key}'},
    {'model': 'o1', 'api_key': f'{openai_api_key}'},
    {'model': 'o3-mini', 'api_key': f'{openai_api_key}'},
    {'model': 'gemini-2.0-flash-thinking-exp-01-21', 'api_key': f'{gemini_api_key}', 'api_type': 'google'},
    {'model': 'gemini-2.0-pro-exp-02-05', 'api_key': f'{gemini_api_key}', 'api_type': 'google'},
    {'model': 'gemini-2.0-flash-exp', 'api_key': f'{gemini_api_key}', 'api_type': 'google'},
]
selected_config_o3_mini = next(config for config in config_list if config['model'] == 'o3-mini')
selected_config_gemini_2_0_flash_thinking = next(config for config in config_list if config['model'] == 'gemini-2.0-flash-thinking-exp-01-21')
selected_config_gemini_2_0_pro = next(config for config in config_list if config['model'] == 'gemini-2.0-pro-exp-02-05')
selected_config_gemini_2_0_flash = next(config for config in config_list if config['model'] == 'gemini-2.0-flash-exp')
llm_config_o3_mini_high = {
    "timeout": 600,
    "cache_seed": 42,
    "config_list": [selected_config_o3_mini],
    "reasoning_effort": "high"
}
llm_config_gemini_2_0_flash_thinking = {
    "timeout": 600,
    "cache_seed": 42,
    "config_list": [selected_config_gemini_2_0_flash_thinking],
    "temperature": 0,
    "model_info": {
         "function_calling": False,
         "json_output": True,
         "vision": True,
         "family": "google"
    }
}
llm_config_gemini_2_0_pro = {
    "timeout": 600,
    "cache_seed": 42,
    "config_list": [selected_config_gemini_2_0_pro],
    "temperature": 0
}
llm_config_gemini_2_0_flash = {
    "timeout": 600,
    "cache_seed": 42,
    "config_list": [selected_config_gemini_2_0_flash],
    "temperature": 0
}

# -------------------------------------------------------------------
# Enhanced Agents
# -------------------------------------------------------------------
class EnhancedResearchAgent(autogen.AssistantAgent):
    def __init__(self, name, llm_config, **kwargs):
        super().__init__(name=name, llm_config=llm_config, **kwargs)
        self.cache_system = CacheSystem()
        self.knowledge_graph = KnowledgeGraph()
        self.resource_manager = ResourceManager()

    async def process_search_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        processed_results = []
        content_processor = ContentProcessor()
        for result in results:
            combined_text = result.get('snippet', '') + "\n" + result.get('full_content', '')
            info = content_processor.extract_key_info(combined_text)
            self.knowledge_graph.add_node(
                result.get('href', ''),
                {
                    'title': result.get('title', ''),
                    'content': combined_text[:500],
                    'entities': info.get('entities', {}),
                    'summary': info.get('summary', '')
                }
            )
            processed_results.append(result)
        return processed_results

class EnhancedAssessmentCommander(autogen.AssistantAgent):
    def __init__(self, name, llm_config, **kwargs):
        super().__init__(name=name, llm_config=llm_config, **kwargs)
        self.command_history = []
        self.search_count = 0

    def generate_next_command(self, results: List[Dict[str, Any]]) -> str:
        self.search_count += 1
        command = self._analyze_results(results)
        self.command_history.append(command)
        return command

    def _analyze_results(self, results: List[Dict[str, Any]]) -> str:
        if self.search_count >= 6:
            return "TERMINATE_CHAT"
        return "continue"

class EnhancedSummarizer(autogen.AssistantAgent):
    def __init__(self, name, llm_config, **kwargs):
        super().__init__(name=name, llm_config=llm_config, **kwargs)
        self.content_processor = ContentProcessor()

    def generate_summary(self, research_data: str) -> str:
        system_prompt = (
            "You are an advanced Summarizer that generates comprehensive, PhD-level documents.\n"
            "Your goal: produce a deep, well-structured summary (≥2000 words) using advanced academic style.\n"
            "Include citations to relevant sources, highlight key arguments, methodology, findings, and future directions.\n"
            "Organize content with sections, headings, subheadings, and references.\n"
            "Make the summary coherent, addressing the subject's background, current state, and potential expansions.\n"
            "You must link ideas, concepts and facts from the different sources that were not necessarily obvious.\n"
            "These sources are fetched from the internet, \n"
            "Some text may be unrelated to the topic due to the scraping precedure(example: Page Navigation Buttons).\n"
            "Finally if you're using text from one of the Jina Deep Search Summaries, \n"
            "when citing please refer to the approriate link within the specific Jina Deep Search Summary Content.\n"
            "PS: You must write the Summary once, do not output the same summary multiple times, Additionally there should be duplicates on the references section.\n\n"
        )
        prompt = system_prompt + "=== RESEARCH DATA ===\n" + research_data + "\n=== END DATA ===\n"

        generation_config = genai.GenerationConfig(temperature=self.llm_config.get("temperature", 0))
        model_info = self.llm_config.get("config_list", [{}])[0]
        model_name = model_info.get("model", "gemini-2.0-flash-thinking-exp-01-21")
        model = genai.GenerativeModel(model_name=model_name)
        response = model.generate_content(prompt, generation_config=generation_config)
        return response.text.strip() if response.text else ""

def termination_check(msg):
    return "TERMINATE_CHAT" in msg["content"]

DeepResearchAgent = EnhancedResearchAgent(
    name="DeepResearchAgent",
    llm_config=llm_config_gemini_2_0_flash,
    system_message=(
        "You are an enhanced DeepResearchAgent with advanced capabilities including "
        "knowledge graph maintenance, optional Jina-based content processing, and "
        "search result re-ranking. Execute research tasks as commanded by the "
        "AssessmentCommander, performing web searches and analyzing results thoroughly."
    )
)
AssessmentCommander = EnhancedAssessmentCommander(
    name="AssessmentCommander",
    llm_config=llm_config_gemini_2_0_pro,
    system_message=(
        "You are an enhanced AssessmentCommander. Review findings using the knowledge "
        "graph and issue intelligent research commands. You must issue at least three "
        "separate search commands before indicating that the research is complete. "
        "You can only issue one search command at a time in one prompt. "
        "You can choose different levels of breadth and depth for each request, but you must always specify them. "
        "Once satisfied, ONLY output 'TERMINATE_CHAT' and nothing else. "
        "Do not use terminating string unless the three rounds of commands have been reached"
    ),
    is_termination_msg=termination_check
)
Summarizer = EnhancedSummarizer(
    name="Summarizer",
    llm_config=llm_config_gemini_2_0_flash_thinking,
    system_message=(
        "Generate a comprehensive, PhD-level summary. The final document must exceed 2000 words, "
        "include proper academic citations and references, highlight advanced knowledge. "
        "Focus on synthesizing multi-source data, analyzing key topics, and structuring your text logically."
    ),
    human_input_mode="NEVER"
)
autogen.register_function(
    deep_research_flow,
    caller=DeepResearchAgent,
    executor=DeepResearchAgent,
    description="Advanced iterative research with knowledge graph integration"
)

# -------------------------------------------------------------------
# Setup Group Chat (using global logs_data)
# -------------------------------------------------------------------
groupchat = autogen.GroupChat(
    agents=[DeepResearchAgent, AssessmentCommander],
    messages=logs_data,
    max_round=50,
    speaker_selection_method="round_robin"
)
manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config_gemini_2_0_pro,
    is_termination_msg=termination_check
)

# -------------------------------------------------------------------
# New helper: run_full_research
# -------------------------------------------------------------------
def run_full_research(subject: str) -> str:
    global SEARCH_RESULTS_FILENAME
    ts = get_current_utc_timestamp()
    SEARCH_RESULTS_FILENAME = f"search_results_{ts}.txt"
    try:
        with open(SEARCH_RESULTS_FILENAME, "w", encoding='utf-8') as f:
            f.write(f"=== Research Session Started: {ts} ===\n")
            f.write(f"Subject: {subject}\n\n")
    except IOError as e:
        logging.error(f"Failed to create results file: {e}")
        sys.exit(1)
    initial_message = f"""
Subject: {subject}

Please perform enhanced deep research on the above subject using:
- Breadth: 3-10 (SERP links per iteration)
- Depth: 1-3 (iterative research rounds)
- Knowledge Graph tracking
- Content analysis and processing
- Intelligent query evolution
- Optionally use Jina for improved ranking, chunking, embeddings (no API key)

Start with:
- Breadth = 5
- Depth = 3

Research flow:
1. DeepResearchAgent performs web search with content processing
2. AssessmentCommander analyzes findings and knowledge graph
3. Process continues until AssessmentCommander has issued at least three distinct commands
4. A final comprehensive, PhD-level report will be generated
"""
    try:
        DeepResearchAgent.initiate_chat(manager, message=initial_message)
    except Exception as e:
        if "exceeds the maximum number of tokens allowed" in str(e):
            logging.warning(f"Token count exceeded allowed maximum during initiate_chat: {e}. Continuing execution.")
        else:
            logging.error(f"Error during initiate_chat: {e}. Continuing execution.")
    with open(SEARCH_RESULTS_FILENAME, 'r', encoding='utf-8') as file:
        full_text = file.read()
    
    def count_tokens_approx(text: str) -> int:
        return int(len(text) / 2.7)
    
    token_count = count_tokens_approx(full_text)
    logging.info(f"Total token count of research results: {token_count}")
    
    final_report = None
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            final_report = Summarizer.generate_summary(full_text)
            logging.info(f"Summary generated successfully on attempt {attempt}.")
            break
        except Exception as e:
            logging.error(f"Attempt {attempt} failed to generate summary: {e}")
            if attempt == max_attempts:
                raise Exception(f"Failed to generate summary after {max_attempts} attempts: {e}")
            time.sleep(5)

    final_summary_filename = f"final_summary_{ts}.txt"
    with open(final_summary_filename, 'w', encoding='utf-8') as output_file:
        output_file.write(final_report)
    print(f"\nResearch completed. Final report saved to {final_summary_filename}")
    
    global persistent_knowledge_graph_data
    for node, data in DeepResearchAgent.knowledge_graph.graph.nodes(data=True):
        if node not in persistent_knowledge_graph.graph:
            persistent_knowledge_graph.add_node(node, data)
    for source, target, data_e in DeepResearchAgent.knowledge_graph.graph.edges(data=True):
        if not persistent_knowledge_graph.graph.has_edge(source, target):
            persistent_knowledge_graph.add_relation(source, target, data_e.get("relation", ""))

    # Also link newly added nodes in the persistent graph with existing nodes in the persistent graph
    # to handle cross-run context once more
    new_nodes_all = list(DeepResearchAgent.knowledge_graph.graph.nodes())
    link_new_subnodes_by_context(persistent_knowledge_graph, new_nodes_all, threshold=0.6)

    persistent_knowledge_graph_data = {
        "nodes": [{"id": node, **data} for node, data in persistent_knowledge_graph.graph.nodes(data=True)],
        "links": [{"source": source, "target": target, **data} for source, target, data in persistent_knowledge_graph.graph.edges(data=True)]
    }
    
    logging.info("Final session knowledge_graph_data nodes: " +
                 f"{[node for node, _ in DeepResearchAgent.knowledge_graph.graph.nodes(data=True)]}")
    logging.info("Final persistent knowledge_graph_data nodes: " +
                 f"{[node for node, _ in persistent_knowledge_graph.graph.nodes(data=True)]}")
    logging.info("Sleeping for 45 seconds to avoid rate limit from Gemini")
    time.sleep(45)
    return final_report

# -------------------------------------------------------------------
# Main Execution (standalone mode)
# -------------------------------------------------------------------
if __name__ == "__main__":
    try:
        print("\033[91mPlease enter the subject for research: \033[0m")
        subject = input().strip()
        if not subject:
            raise ValueError("Subject cannot be empty")
        final_report = run_full_research(subject)
    except KeyboardInterrupt:
        print("\nResearch process interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Critical error in main execution: {e}")
        sys.exit(1)

