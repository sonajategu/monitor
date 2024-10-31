import logging
import logging.handlers
import json
import time
import os
from pathlib import Path
from functools import wraps
import traceback
from zipfile import error
import concurrent.futures
import redis
import requests
import openai
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse  # For URL joining
from typing import List, Dict, Any, Optional, Union  # For better type hints
from bs4 import BeautifulSoup
import feedparser
from datetime import datetime

from dotenv import load_dotenv

from news_sources import NEWS_SOURCES, NewsSource


class ActivityLogger:
    """
    Handles structured logging for the promise tracker application
    """

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Set up different log files for different purposes
        self._setup_loggers()

    def _setup_loggers(self):
        # Main activity logger
        self.activity_logger = logging.getLogger('activity')
        self.activity_logger.setLevel(logging.INFO)

        # Error logger
        self.error_logger = logging.getLogger('error')
        self.error_logger.setLevel(logging.ERROR)

        # Performance logger
        self.perf_logger = logging.getLogger('performance')
        self.perf_logger.setLevel(logging.INFO)

        # Set up handlers
        self._setup_handler('activity', 'activity.log')
        self._setup_handler('error', 'error.log')
        self._setup_handler('performance', 'performance.log')

    def _setup_handler(self, logger_name: str, filename: str):
        logger = logging.getLogger(logger_name)

        # File handler with rotation
        handler = logging.handlers.RotatingFileHandler(
            self.log_dir / filename,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )

        # JSON formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)

        # Also add a console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    def log_activity(self,
                     activity_type: str,
                     status: str,
                     details: Dict[str, Any],
                     duration_ms: Optional[float] = None):
        """Log an activity with structured data"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'activity_type': activity_type,
            'status': status,
            'details': details
        }
        if duration_ms is not None:
            log_entry['duration_ms'] = duration_ms

        self.activity_logger.info(json.dumps(log_entry))

        # Log performance data if duration is provided
        if duration_ms is not None:
            self.perf_logger.info(json.dumps({
                'timestamp': datetime.now().isoformat(),
                'activity_type': activity_type,
                'duration_ms': duration_ms
            }))

    def log_error(self,
                  activity_type: str,
                  error: Exception,
                  details: Dict[str, Any]):
        """Log an error with full stack trace"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'activity_type': activity_type,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'stack_trace': traceback.format_exc(),
            'details': details
        }
        self.error_logger.error(json.dumps(log_entry))


def log_activity(activity_type: str):
    """Decorator to log function execution and timing"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            status = 'success'
            details = {'function': func.__name__}

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                status = 'error'
                details['error'] = str(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                ActivityLogger().log_activity(
                    activity_type=activity_type,
                    status=status,
                    details=details,
                    duration_ms=duration_ms
                )

        return wrapper

    return decorator

class NewsMonitor:
# Update the main classes to include logging
    def __init__(self, news_sources: List[NewsSource], redis_client: Optional[redis.Redis] = None, article_limit: int = 100):
        self.news_sources = news_sources
        self.article_limit = article_limit
        self.logger = ActivityLogger()
        self.base_url = "https://www.err.ee"
        self.redis_client = redis_client
        self.max_workers = 4
        
        # Add allowed/blocked domain configuration
        # self.allowed_domains = {"www.err.ee"}  # Only allow articles from this domain
        self.blocked_domains = {"news.err.ee", "rus.err.ee"}  # Block these domains

        # Initialize Redis client if provided
        if redis_client is None:
            try:
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=0,
                    socket_timeout=2,
                    decode_responses=True
                )
                self.redis_client.ping()  # Test connection
            except redis.RedisError as e:
                self.logger.log_error(
                    activity_type="redis_init",
                    error=e,
                    details={"message": "Failed to connect to Redis, running without caching"}
                )
                self.redis_client = None
        else:
            self.redis_client = redis_client

    def _is_allowed_domain(self, url: str) -> bool:
            """Check if the URL's domain is allowed"""
            try:
                parsed_url = urlparse(url)
                domain = parsed_url.netloc.lower()
                
                # First check if domain is explicitly blocked
                if domain in self.blocked_domains:
                    return False
                    
                # Then check if domain is explicitly allowed
                # if domain in self.allowed_domains:
                #     return True
                    
                # If domain is neither blocked nor allowed, block it by default
                return False
                
            except Exception as e:
                self.logger.log_error(
                    activity_type="domain_check",
                    error=e,
                    details={"url": url}
                )
                return False
                
    def is_cached(self, url: str) -> bool:
        """Check if article URL is cached with fallback"""
        if self.redis_client is None:
            return False
        try:
            return bool(self.redis_client.get(url))
        except redis.RedisError as e:
            self.logger.log_error(
                activity_type="redis_cache_check",
                error=e,
                details={"url": url}
            )
            return False

    def cache_article(self, url: str) -> None:
        """Cache article URL with error handling"""
        if self.redis_client is None:
            return
        try:
            self.redis_client.set(url, "1", ex=86400)  # Cache for 24 hours
        except redis.RedisError as e:
            self.logger.log_error(
                activity_type="redis_cache_set",
                error=e,
                details={"url": url}
            )

    def _parse_articles_from_rss(self, news_source: NewsSource) -> List[Dict]:
        """Parse articles from RSS feed"""
        articles = []
        try:
            time.sleep(news_source.rate_limit)
            feed = feedparser.parse(news_source.url)
    
            if not hasattr(feed, 'entries'):
                self.logger.log_error(
                    activity_type="rss_feed_parsing",
                    error=Exception("No entries in feed"),
                    details={"source": news_source.to_dict()}
                )
                return []
    
            entries = feed.entries[:self.article_limit]
    
            for entry in entries:
                try:
                    # Get the article URL
                    article_url = getattr(entry, 'link', '')
                    
                    # Skip if URL is not from allowed domain
                    if not self._is_allowed_domain(article_url):
                        continue
    
                    article = {
                        "url": article_url,
                        "title": getattr(entry, 'title', ''),
                        "date": datetime(*getattr(entry, 'published_parsed', time.localtime())[:6]).isoformat(),
                        "source": news_source.to_dict(),
                        "description": getattr(entry, 'description', '')
                    }
    
                    # Skip if missing required fields
                    if not article["url"] or not article["title"]:
                        continue
    
                    if not self.is_cached(article["url"]):
                        full_content = self._fetch_article_content(article["url"])
                        if full_content:
                            article["content"] = full_content
                            articles.append(article)
    
                except Exception as e:
                    self.logger.log_error(
                        activity_type="article_parsing",
                        error=e,
                        details={"source": news_source.to_dict()}
                    )
                    continue
    
            return articles
    
        except Exception as e:
            self.logger.log_error(
                activity_type="rss_feed_parsing",
                error=e,
                details={"source": news_source.to_dict()}
            )
            return []

    def _fetch_article_content(self, url: str) -> Optional[str]:
            """Fetch and parse full article content with rate limiting"""
            try:
                time.sleep(1)  # Basic rate limiting between requests

                response = requests.get(url, timeout=1000)
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                # Find article content div
                content_div = soup.find("div", class_="text lead2") or soup.find("div", class_="text")
                if not content_div:
                    return None

                # Extract text content safely
                text_parts = []
                for element in content_div.select('p, div'):  # Using CSS selector instead of find_all
                    text = element.get_text(strip=True)
                    if text:
                        text_parts.append(text)

                return "\n".join(text_parts) if text_parts else None
            except Exception as e:
                self.logger.log_error(
                    activity_type="article_content_fetch",
                    error=e,
                    details={"url": url}
                )
                return None


    def _parse_articles(self, soup: BeautifulSoup, source_url: str) -> List[Dict]:
            """Parse articles from ERR news pages"""
            articles = []

            try:
                # Find all article items on the page
                article_items = soup.find_all("div", class_="category-item") if hasattr(soup, 'find_all') else []

                # Limit the number of articles to process
                article_items = article_items[:self.article_limit]

                for item in article_items:
                    try:
                        if not hasattr(item, 'find'):
                            continue

                        # Get article header with link
                        header = item.find("p", class_="category-news-header")
                        if not header or not hasattr(header, 'find'):
                            continue

                        link = header.find("a")
                        if not link:
                            continue

                        # Extract article URL and title
                        article_url = urljoin(self.base_url, link.get("href", ""))
                        title = link.get_text(strip=True) if hasattr(link, 'get_text') else ""

                        if not article_url or not title:
                            continue

                        # Get article timestamp
                        timestamp_div = item.find("div", class_="news-time")
                        date = timestamp_div.get_text(strip=True) if timestamp_div and hasattr(timestamp_div, 'get_text') else None

                        # Fetch full content
                        full_content = self._fetch_article_content(article_url)
                        if full_content:
                            articles.append({
                                "url": article_url,
                                "title": title,
                                "date": date,
                                "source": source_url,
                                "content": full_content
                            })

                    except Exception as e:
                        self.logger.log_error(
                            activity_type="article_parsing",
                            error=e,
                            details={"source": source_url}
                        )
                        continue

                return articles

            except Exception as e:
                self.logger.log_error(
                    activity_type="articles_parsing",
                    error=e,
                    details={"source": source_url}
                )
                return []


    def _fetch_article_content(self, url: str) -> Optional[str]:
            """Fetch and parse full article content with rate limiting"""
            try:
                time.sleep(1)  # Basic rate limiting between requests

                response = requests.get(url, timeout=(5, 60))
                response.raise_for_status()

                soup = BeautifulSoup(response.content, 'html.parser')

                # Find article content div
                content_div = soup.find("div", class_="text lead2") or soup.find("div", class_="text")
                if not content_div:
                    return None

                # Extract text content safely
                text_parts = []
                if hasattr(content_div, 'select'):
                    for element in content_div.select('p, div'):
                        if hasattr(element, 'get_text'):
                            text = element.get_text(strip=True)
                            if text:
                                text_parts.append(text)

                return "\n".join(text_parts) if text_parts else None

            except Exception as e:
                self.logger.log_error(
                    activity_type="article_content_fetch",
                    error=e,
                    details={"url": url}
                )
                return None

    def fetch_articles(self) -> List[Dict]:
        self.logger.log_activity(
            activity_type="fetch_start",
            status="info",
            details={"sources_count": len(self.news_sources)}
        )

        """Fetch and parse articles from all RSS sources in parallel"""
        all_articles = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks for each source
            future_to_source = {
                executor.submit(self._parse_articles_from_rss, source): source
                for source in self.news_sources
            }

            # Process completed tasks
            for future in concurrent.futures.as_completed(future_to_source):
                source = future_to_source[future]
                try:
                    articles = future.result()
                    all_articles.extend(articles)

                    self.logger.log_activity(
                        activity_type="fetch_complete",
                        status="success",
                        details={
                            "source": source.to_dict(),
                            "articles_found": len(articles)
                        }
                    )
                except Exception as e:
                    self.logger.log_error(
                        activity_type="fetch_failed",
                        error=e,
                        details={"source": source.to_dict()}
                    )

        # Sort articles by date (newest first) and limit the total number
        all_articles.sort(key=lambda x: x['date'], reverse=True)
        return all_articles[:self.article_limit]


@dataclass
class Promise:
    title: str
    content: str
    source_url: str
    politician: str
    date: str
    similar_promises: Optional[List["Promise"]] = None  # Use Optional for nullable field


class PromiseDetector:
    def __init__(self, openai_api_key: str):
        self.openai_client = openai.Client(api_key=openai_api_key)
        self.logger = ActivityLogger()

        # Set up specific OpenAI logger
        self.openai_logger = logging.getLogger('openai')
        self.openai_logger.setLevel(logging.INFO)

        # Create OpenAI log file handler
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        handler = logging.handlers.RotatingFileHandler(
            log_dir / 'openai_responses.log',
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.openai_logger.addHandler(handler)

    def _parse_gpt_response(self, response_text: str) -> List[Dict]:
        """Parse GPT response into basic promise data format"""
        try:
            response_data = json.loads(response_text)
            promises = response_data.get('promises', [])

            cleaned_promises = []
            for promise in promises:
                if all(k in promise for k in ['title', 'content', 'source_url', 'person', 'date']):
                    cleaned_promise = {
                        'title': promise['title'],
                        'content': promise['content'],
                        'source_url': promise['source_url'],
                        'person': promise['person'],
                        'date': promise['date'],
                        'authority': promise.get('authority', ''),
                        'topic': promise.get('topic', ''),
                        'budget': promise.get('budget', ''),
                        'deadline': promise.get('deadline', '')
                    }
                    cleaned_promises.append(cleaned_promise)

            return cleaned_promises
        except Exception as e:
            self.logger.log_error(
                activity_type="gpt_response_parsing",
                error=e,
                details={"response_text": response_text}
            )
            return []

    def _calculate_cost(self, usage: Dict[str, int], model: str) -> float:
        """Calculate approximate cost of the API call"""
        pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-32k": {"input": 0.06, "output": 0.12},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-3.5-turbo-0125": {"input": 0.0005, "output": 0.0015}
        }

        if model not in pricing:
            return 0.00

        rates = pricing[model]
        input_cost = (usage['prompt_tokens'] / 1000) * rates['input']
        output_cost = (usage['completion_tokens'] / 1000) * rates['output']

        return round(input_cost + output_cost, 6)

    def _log_openai_interaction(self, article_text: str, response: Any, error: Optional[Exception] = None):
        """Log OpenAI interaction details"""
        try:
            cost = None
            if response and hasattr(response, 'usage'):
                usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
                cost = self._calculate_cost(usage, response.model)

            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'request': {
                    'article_length': len(article_text),
                    'article_preview': article_text[:500] + '...' if len(article_text) > 500 else article_text
                },
                'response': {
                    'raw_content': response.choices[0].message.content if response else None,
                    'model': response.model if response else None,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens if response else None,
                        'completion_tokens': response.usage.completion_tokens if response else None,
                        'total_tokens': response.usage.total_tokens if response else None
                    } if response else None,
                    'cost_usd': cost
                },
                'error': {
                    'type': type(error).__name__,
                    'message': str(error)
                } if error else None
            }

            self.openai_logger.info(json.dumps(log_entry, ensure_ascii=False, indent=2))
            self.logger.log_activity(
                activity_type="openai_interaction",
                status="success" if not error else "error",
                details={
                    'article_length': len(article_text),
                    'tokens_used': response.usage.total_tokens if response else None,
                    'cost_usd': cost,
                    'error': str(error) if error else None
                }
            )

        except Exception as logging_error:
            self.logger.log_error(
                activity_type="openai_logging",
                error=logging_error,
                details={
                    'article_length': len(article_text),
                    'original_error': str(error) if error else None
                }
            )

    @log_activity("promise_detection")
    def detect_promises(self, article_text: str) -> List[Dict]:
            response = None
            try:
                system_prompt = """Sa oled assistent, mis on spetsialiseerunud lubaduste tuvastamisele ja väljavõtmisele uudisartiklitest. Sinu ülesanne on analüüsida artikli teksti ning tuvastada kõik konkreetsed lubadused, keskendudes eelkõige poliitika ja sotsiaalsete küsimustega seotud lubadustele. Lubadus võib olla ametlik avaldus, kohustus või kinnitus, mille on esitanud poliitiline tegelane või asutus seoses tulevaste tegevuste või plaanidega. Tuvasta lubaduse puhul järgmised kriteeriumid (kui need puuduvad, jäta väli tühjaks): eesmärk (mida lubatakse teha), sihtrühm (kellele mõju avaldub), tähtaeg, eelarve (kui on mainitud), vastutav isik ja/või asutus.

                Samuti loo lühike mitmuses pealkiri eesti keeles, vältides pronoomenite kasutust. Pealkirjas võib kajastada erakonna nime, kui see on artiklis mainitud.

                Tuvasta ja too infot ainult antud teksti põhjal. Vasta küsimustele täpselt, kasutades üksnes artikli sisu ja tekstis sisalduvaid lubadusi. Lisatõlgendused või oletused lubaduste kohta on keelatud.

                Konverteeri leitud lubadused alljärgneva JSON-vormingusse:
                {
                    "promises": [
                        {
                            "title": "Lühike lubaduse pealkiri",
                            "content": "Lubaduse täielik tekst",
                            "source_url": "Artikli URL",
                            "person": "Lubaduse andja nimi",
                            "date": "Lubaduse kuupäev (formaadis %Y-%m-%d)",
                            "authority": "Lubadusega seotud asutus (kui on mainitud)",
                            "topic": "Teema vastavalt artikli sisule (kui on mainitud või tuvastatav)",
                            "budget": "Lubaduse eelarve eurodes (kui on mainitud)",
                            "deadline": "Lubaduse tähtaeg (kui on mainitud)"
                        }
                    ]
                }

                Kui lubadusi ei leidu, tagasta tühi massiiv: {"promises": []}.
                Kui lubadus on leitud, kuid mõni väli, näiteks eelarve, puudub, siis jäta see väli tühjaks.

                Veendu, et vastus oleks alati eesti keeles ja vastaks eesti keele grammatikareeglitele, isegi kui algtekst on mõnes muus keeles."""

                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": article_text}
                    ],
                    response_format={"type": "json_object"}
                )

                # Log the interaction
                self._log_openai_interaction(article_text, response)

                # Use _parse_gpt_response to format the response
                return self._parse_gpt_response(response.choices[0].message.content)

            except Exception as e:
                self._log_openai_interaction(article_text, response, e)
                return []


class WordPressRestAPI:
    def __init__(self, api_base: str, token: str):
        self.api_base = api_base.rstrip('/')  # Remove trailing slash if present
        self.token = token
        self.logger = ActivityLogger()

    def _get_headers(self) -> Dict[str, str]:
        """Get headers with JWT token"""
        return {
            'Authorization': f'Bearer {self.token}',
            'Content-Type': 'application/json'
        }

    def _handle_api_response(self, response: requests.Response, operation: str) -> Dict:
        """Handle API response with proper error handling"""
        try:
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            error_detail = response.json() if response.text else str(e)
            self.logger.log_error(
                activity_type=f"api_{operation}",
                error=e,
                details={"response": error_detail}
            )
            raise

    def _process_taxonomy_terms(self, promise: Dict) -> Dict[str, List[int]]:
        """Process taxonomy terms and return their IDs"""
        taxonomy_ids = {
            'person': [],
            'topic': [],
            'authority': []
        }

        # Map promise fields to taxonomies
        taxonomy_map = {
            'person': 'person',
            'topic': 'topic',
            'authority': 'authority'
        }

        for tax_name, promise_field in taxonomy_map.items():
            if promise.get(promise_field):
                term_id = self._get_or_create_term(tax_name, promise[promise_field])
                if term_id:
                    taxonomy_ids[tax_name].append(term_id)

        return taxonomy_ids

    def _prepare_acf_fields(self, promise: Dict, taxonomy_ids: Dict[str, List[int]]) -> Dict:
        """Prepare ACF fields with proper taxonomy IDs"""
        acf_fields = {
                'status': 58,  # Default status "Lubatud"
                'promise_description': promise['content'],
                'source': promise['source_url'],
                'promise_date': datetime.strptime(promise['date'], '%Y-%m-%d').strftime('%Y%m%d'),
                'persons': taxonomy_ids['person'],
                'responsible_authorities': taxonomy_ids['authority'],
                'topic': taxonomy_ids['topic'],
                'promise_deadline': promise.get('deadline') or None,  # Use None instead of empty string
                # 'repeater_expected_result': [{
                #     'expected_result_update_date': datetime.now().strftime('%Y%m%d'),
                #     'expected_result_update_text': promise['content'],
                #     'expected_result_update_source': promise['source_url']
                # }]
            }

        # Only add budget if it's a valid number
        budget = promise.get('budget')
        if budget and str(budget).strip():  # Check if budget exists and is not empty
                try:
                    # Try to convert to float first to handle both integer and decimal values
                    budget_value = float(budget)
                    acf_fields['budget'] = budget_value
                except (ValueError, TypeError):
                    # If conversion fails, don't include budget field
                    pass
        else:
            acf_fields['budget'] = None  # Use None instead of empty string

        return acf_fields

    @log_activity("wordpress_submission")
    def submit_promise(self, promise: Dict) -> Dict:
        """Submit a promise to WordPress"""
        try:
            # First process taxonomy terms
            taxonomy_ids = self._process_taxonomy_terms(promise)

            # Prepare ACF fields with taxonomy IDs
            acf_fields = self._prepare_acf_fields(promise, taxonomy_ids)

            # Prepare the complete post data
            post_data = {
                'title': promise['title'],
                'content': promise['content'],
                'status': 'publish',
                'type': 'promise',
                'acf': acf_fields
            }

            # Add taxonomy terms to post data
            for tax_name, ids in taxonomy_ids.items():
                if ids:
                    post_data[tax_name] = ids

            # Make the API request
            response = requests.post(
                f"{self.api_base}/wp-json/wp/v2/promise",
                json=post_data,
                headers=self._get_headers(),
                timeout=30
            )

            # Log the response
            self.logger.log_activity(
                activity_type="promise_submission_response",
                status="info",
                details={
                    "status_code": response.status_code,
                    "response_text": response.text,
                    "request_data": post_data
                }
            )

            if response.status_code not in [200, 201]:
                return {
                    "error": True,
                    "message": f"HTTP {response.status_code}: {response.text}",
                    "status_code": response.status_code
                }

            return response.json()

        except Exception as e:
            self.logger.log_error(
                activity_type="promise_submission",
                error=e,
                details={"promise_data": promise}
            )
            return {
                "error": True,
                "message": str(e),
                "error_type": type(e).__name__
            }

    def _get_or_create_term(self, taxonomy: str, term_name: str) -> Optional[int]:
        """Get or create a taxonomy term and return its ID"""
        try:
            # First try to find existing term
            response = requests.get(
                f"{self.api_base}/wp-json/wp/v2/{taxonomy}",
                params={'search': term_name},
                headers=self._get_headers()
            )

            terms = self._handle_api_response(response, f"get_{taxonomy}")

            if terms:
                return terms[0]['id']

            # If not found, create new term
            response = requests.post(
                f"{self.api_base}/wp-json/wp/v2/{taxonomy}",
                json={'name': term_name},
                headers=self._get_headers()
            )

            new_term = self._handle_api_response(response, f"create_{taxonomy}")
            return new_term['id']

        except Exception as e:
            self.logger.log_error(
                activity_type=f"taxonomy_term_operation",
                error=e,
                details={"taxonomy": taxonomy, "term": term_name}
            )
            return None


def _prepare_post_data(self, promise: Dict) -> Dict:
    """Prepare promise data for WordPress submission"""
    try:
        # Convert date to YYYYMMDD format if available
        promise_date = datetime.strptime(promise['date'], '%Y-%m-%d').strftime('%Y%m%d') if promise.get(
            'date') else None

        # Start with minimal required fields
        post_data = {
            'title': promise['title'],
            'content': promise['content'],
            'status': 'publish',
            'type': 'promise'
        }

        # Build ACF fields, only including non-empty values
        acf_data = {
            'status': 58,
            'promise_description': promise['content'],
            'source': promise.get('source_url', '')
        }

        # Only add date if available
        if promise_date:
            acf_data['promise_date'] = promise_date

        # Add expected result only if we have content
        # if promise.get('content'):
        #     acf_data['repeater_expected_result'] = [{
        #         'expected_result_update_date': datetime.now().strftime('%Y%m%d'),
        #         'expected_result_update_text': promise['content'],
        #         'exprected_result_update_source': promise.get('source_url', ''),
        #     }]

        # Add optional fields only if they exist and have values
        optional_fields = {
            'budget': 'budget',
            'target_group': 'target_group',
            'promise_deadline': 'deadline',
            'source_document': 'source_document'
        }

        for wp_field, promise_field in optional_fields.items():
            if promise.get(promise_field):
                acf_data[wp_field] = promise[promise_field]

        # Add ACF data to post_data
        post_data['acf'] = acf_data

        # Handle taxonomies
        taxonomies = {
            'person': ('persons', 'person'),
            'topic': ('topic', 'topic'),
            'authority': ('responsible_authorities', 'authority')
        }

        for tax_name, (acf_field, promise_field) in taxonomies.items():
            if promise.get(promise_field):
                term_id = self._get_or_create_term(tax_name, promise[promise_field])
                if term_id:
                    # Only add taxonomy if we successfully got/created the term
                    if tax_name not in post_data:
                        post_data[tax_name] = []
                    post_data[tax_name].append(term_id)

                    # Also add to ACF fields
                    if acf_field not in acf_data:
                        acf_data[acf_field] = []
                    acf_data[acf_field].append(term_id)

        # Add status taxonomy (required)
        post_data['staatus'] = [58]

        return post_data

    except Exception as e:
        self.logger.log_error(
            activity_type="prepare_post_data",
            error=e,
            details={"promise": promise}
        )
        raise

    # @log_activity("wordpress_fetch")
    # def get_existing_promises(self) -> List[Promise]:
    #     """Fetch existing promises from WordPress"""
    #     try:
    #         response = requests.get(
    #             f"{self.api_base}/wp-json/wp/v2/promise",
    #             headers=self._get_headers()
    #         )
    #         response.raise_for_status()
    #
    #         promises = []
    #         for item in response.json():
    #             try:
    #                 promise = Promise(
    #                     title=item['title']['rendered'],
    #                     content=item['content']['rendered'],
    #                     source_url=item['fields'].get('source_url', ''),
    #                     politician=item['fields'].get('politician', ''),
    #                     date=item['fields'].get('promise_date', ''),
    #                     similar_promises=None
    #                 )
    #                 promises.append(promise)
    #             except KeyError as e:
    #                 self.logger.log_error(
    #                     activity_type="promise_parsing",
    #                     error=e,
    #                     details={"item_id": item.get('id')}
    #                 )
    #                 continue
    #
    #         return promises
    #
    #     except Exception as e:
    #         self.logger.log_error(
    #             activity_type="wordpress_fetch",
    #             error=e,
    #             details={}
    #         )
    #         return []
    #


def main():
    logger = ActivityLogger()

    # Load configuration
    load_dotenv()

    WP_API_BASE = os.getenv('WP_API_BASE')
    WP_JWT_TOKEN = os.getenv('WP_JWT_TOKEN')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    if not WP_JWT_TOKEN:
        raise ValueError("WordPress JWT token not configured")

    try:
        # Initialize components
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        news_monitor = NewsMonitor(
            news_sources=NEWS_SOURCES,  # Contains NewsSource objects
            redis_client=redis_client,
            article_limit=100
        )

        # promise_detector = PromiseDetector(os.getenv('OPENAI_API_KEY'))
        promise_detector = PromiseDetector(
            OPENAI_API_KEY
        )

        wp_client = WordPressRestAPI(
            api_base=WP_API_BASE,
            token=WP_JWT_TOKEN
        )

        while True:
            cycle_start_time = time.time()

            try:
                # Fetch articles in parallel
                articles = news_monitor.fetch_articles()

                logger.log_activity(
                    activity_type="fetch_cycle",
                    status="success",
                    details={
                        "total_articles": len(articles),
                        "sources_processed": len(NEWS_SOURCES)
                    }
                )

                # Process articles
                for article in articles:
                    if 'content' not in article or not article['content']:
                        continue

                    if news_monitor.is_cached(article['url']):
                        continue

                    article_text = f"""
                    Title: {article['title']}
                    Date: {article['date']}
                    URL: {article['url']}

                    Content:
                    {article['content']}
                    """

                    # Get promises as dictionaries
                    promises = promise_detector.detect_promises(article_text)

                    if promises:
                        for promise_data in promises:
                            try:
                                # Submit the promise data
                                result = wp_client.submit_promise(promise_data)

                                if result.get('error'):
                                    logger.log_error(
                                        activity_type="promise_submission",
                                        error=Exception(result.get('message', 'Unknown error')),
                                        details={
                                            "promise_data": promise_data,
                                            "error_details": result
                                        }
                                    )
                                    continue

                                logger.log_activity(
                                    activity_type="promise_submission",
                                    status="success",
                                    details={
                                        "promise_title": promise_data['post_title'],
                                        "post_id": result.get('id')
                                    }
                                )

                            except Exception as e:
                                logger.log_error(
                                    activity_type="promise_submission",
                                    error=e,
                                    details={"promise_data": promise_data}
                                )

                    news_monitor.cache_article(article['url'])

                    logger.log_activity(
                        activity_type="article_processing",
                        status="success",
                        details={
                            "url": article['url'],
                            "promises_found": len(promises)
                        }
                    )

            except Exception as e:
                logger.log_error(
                    activity_type="processing_cycle",
                    error=e,
                    details={
                        "cycle_start_time": cycle_start_time,
                        "sources": [source.to_dict() for source in NEWS_SOURCES]
                    }
                )

            time.sleep(3600)  # 1 hour

    except Exception as e:
        logger.log_error(
            activity_type="application_startup",
            error=e,
            details={}
        )
        raise


if __name__ == "__main__":
    main()
