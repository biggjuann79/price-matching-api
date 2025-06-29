import asyncio
import aiohttp
import feedparser
import re
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from urllib.parse import urljoin, urlparse
import sqlite3
from contextlib import asynccontextmanager
import hashlib
from difflib import SequenceMatcher
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import requests
import time
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(**name**)

@dataclass
class CraigslistListing:
id: str
title: str
price: float
description: str
location: str
url: str
posted_date: datetime
category: str
normalized_title: str = “”
# Enhanced categorization fields
subcategory: str = “”
brand: str = “”
model: str = “”
condition: str = “”
color: str = “”
size: str = “”
material: str = “”
age_estimate: str = “”  # “new”, “like_new”, “used”, “vintage”
price_range: str = “”   # “budget”, “mid_range”, “premium”, “luxury”
keywords: List[str] =     async def run_full_analysis(self, cities: List[str], categories: List[str]) -> List[PriceAnalysis]:
“”“Run complete price matching analysis”””
logger.info(“Starting full price matching analysis”)

```
    # Step 1: Fetch Craigslist listings
    logger.info("Fetching Craigslist listings...")
    cl_listings = await self.craigslist_scraper.fetch_listings(cities, categories)
    logger.info(f"Found {len(cl_listings)} Craigslist listings")
    
    # Step 2: Process each listing
    analyses = []
    for listing in cl_listings:
        try:
            # Extract detailed product information
            listing = self.product_matcher.extract_product_details(listing)
            
            # Normalize the title for eBay search
            listing.normalized_title = self.product_matcher.normalize_title(
                listing.title, listing.category
            )
            
            # Save to database
            self.db.save_craigslist_listing(listing)
            
            # Search eBay for similar items
            ebay_listings = await self.ebay_client.search_sold_listings(
                listing.normalized_title
            )
            
            if ebay_listings:
                # Save eBay data
                self.db.save_ebay_listings(ebay_listings, listing.normalized_title)
                
                # Find best matches
                matches = self.product_matcher.find_best_matches(
                    listing.title, [eb.title for eb in ebay_listings]
                )
                
                if matches:
                    # Filter eBay listings to only matched ones
                    matched_ebay = [eb for eb in ebay_listings 
                                  if any(eb.title == match[0] for match, score in matches)]
                    
                    # Calculate analysis
                    analysis = self._analyze_pricing(listing, matched_ebay, matches)
                    
                    # Calculate and store deal score
                    listing.deal_score = self._calculate_deal_score(analysis)
                    
                    analyses.append(analysis)
                    
                    # Save analysis and update listing with deal score
                    self.db.save_price_analysis(analysis)
                    self.db.save_craigslist_listing(listing)  # Update with deal score
            
            # Be respectful with API calls
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error processing listing {listing.id}: {e}")
    
    logger.info(f"Completed analysis for {len(analyses)} listings")
    return analyses

def _calculate_deal_score(self, analysis: PriceAnalysis) -> float:
    """Calculate a 0-100 deal score based on price difference and confidence"""
    if analysis.avg_sold_price == 0:
        return 0.0
    
    # Calculate percentage savings
    savings_pct = (analysis.avg_sold_price - analysis.craigslist_listing.price) / analysis.avg_sold_price
    
    # Base score from savings (0-100)
    base_score = min(max(savings_pct * 100, 0), 100)
    
    # Adjust by confidence score (0-1)
    final_score = base_score * analysis.confidence_score
    
    return round(final_score, 1)

def get_filtered_listings(self, filters: Dict) -> List[Dict]:
    """Get filtered listings for the iOS app"""
    query = """
        SELECT * FROM craigslist_listings 
        WHERE 1=1
    """
    params = []
    
    # Apply filters
    if filters.get('category'):
        query += " AND category = ?"
        params.append(filters['category'])
    
    if filters.get('subcategory'):
        query += " AND subcategory = ?"
        params.append(filters['subcategory'])
    
    if filters.get('brand'):
        query += " AND brand LIKE ?"
        params.append(f"%{filters['brand']}%")
    
    if filters.get('condition'):
        query += " AND condition = ?"
        params.append(filters['condition'])
    
    if filters.get('color'):
        query += " AND color = ?"
        params.append(filters['color'])
    
    if filters.get('price_range'):
        query += " AND price_range = ?"
        params.append(filters['price_range'])
    
    if filters.get('min_price'):
        query += " AND price >= ?"
        params.append(filters['min_price'])
    
    if filters.get('max_price'):
        query += " AND price <= ?"
        params.append(filters['max_price'])
    
    if filters.get('location'):
        query += " AND location LIKE ?"
        params.append(f"%{filters['location']}%")
    
    if filters.get('min_deal_score'):
        query += " AND deal_score >= ?"
        params.append(filters['min_deal_score'])
    
    # Sorting
    sort_by = filters.get('sort_by', 'deal_score')
    sort_order = filters.get('sort_order', 'DESC')
    query += f" ORDER BY {sort_by} {sort_order}"
    
    # Limit
    limit = filters.get('limit', 100)
    query += f" LIMIT {limit}"
    
    with sqlite3.connect(self.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, params)
        
        results = []
        for row in cursor.fetchall():
            result = dict(row)
            # Parse JSON fields
            result['keywords'] = json.loads(result['keywords'] or '[]')
            result['image_urls'] = json.loads(result['image_urls'] or '[]')
            results.append(result)
        
        return results

def get_filter_options(self) -> Dict:
    """Get available filter options for the iOS app"""
    with sqlite3.connect(self.db_path) as conn:
        cursor = conn.cursor()
        
        # Get unique values for each filter
        filter_options = {}
        
        # Categories
        cursor.execute("SELECT DISTINCT category FROM craigslist_listings WHERE category != ''")
        filter_options['categories'] = [row[0] for row in cursor.fetchall()]
        
        # Subcategories
        cursor.execute("SELECT DISTINCT subcategory FROM craigslist_listings WHERE subcategory != ''")
        filter_options['subcategories'] = [row[0] for row in cursor.fetchall()]
        
        # Brands
        cursor.execute("SELECT DISTINCT brand FROM craigslist_listings WHERE brand != '' ORDER BY brand")
        filter_options['brands'] = [row[0] for row in cursor.fetchall()]
        
        # Conditions
        cursor.execute("SELECT DISTINCT condition FROM craigslist_listings WHERE condition != ''")
        filter_options['conditions'] = [row[0] for row in cursor.fetchall()]
        
        # Colors
        cursor.execute("SELECT DISTINCT color FROM craigslist_listings WHERE color != ''")
        filter_options['colors'] = [row[0] for row in cursor.fetchall()]
        
        # Price ranges
        cursor.execute("SELECT DISTINCT price_range FROM craigslist_listings WHERE price_range != ''")
        filter_options['price_ranges'] = [row[0] for row in cursor.fetchall()]
        
        # Locations
        cursor.execute("SELECT DISTINCT location FROM craigslist_listings WHERE location != '' ORDER BY location")
        filter_options['locations'] = [row[0] for row in cursor.fetchall()]
        
        # Price statistics
        cursor.execute("SELECT MIN(price), MAX(price), AVG(price) FROM craigslist_listings WHERE price > 0")
        price_stats = cursor.fetchone()
        filter_options['price_stats'] = {
            'min': price_stats[0] or 0,
            'max': price_stats[1] or 0,
            'avg': price_stats[2] or 0
        }
        
        # Deal score statistics
        cursor.execute("SELECT MIN(deal_score), MAX(deal_score), AVG(deal_score) FROM craigslist_listings")
        deal_stats = cursor.fetchone()
        filter_options['deal_score_stats'] = {
            'min': deal_stats[0] or 0,
            'max': deal_stats[1] or 0,
            'avg': deal_stats[2] or 0
        }
        
        return filter_options
image_urls: List[str] = None
deal_score: float = 0.0  # 0-100 score for how good the deal is
```

@dataclass
class EbaySoldListing:
title: str
sold_price: float
sold_date: datetime
condition: str
shipping_cost: float
url: str

@dataclass
class PriceAnalysis:
craigslist_listing: CraigslistListing
ebay_matches: List[EbaySoldListing]
avg_sold_price: float
price_difference: float
confidence_score: float
recommendation: str

class ProductMatcher:
“”“AI-powered product name matching and categorization system”””

```
def __init__(self):
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')
    
    self.stop_words = set(stopwords.words('english'))
    self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Enhanced categorization patterns
    self.brand_patterns = {
        'electronics': {
            'phones': ['apple', 'samsung', 'google', 'oneplus', 'huawei', 'xiaomi'],
            'laptops': ['apple', 'dell', 'hp', 'lenovo', 'asus', 'acer', 'microsoft'],
            'tablets': ['apple', 'samsung', 'microsoft', 'amazon', 'lenovo'],
            'cameras': ['canon', 'nikon', 'sony', 'fujifilm', 'olympus', 'panasonic'],
            'audio': ['bose', 'sony', 'beats', 'jbl', 'sennheiser', 'audio-technica'],
            'gaming': ['sony', 'microsoft', 'nintendo', 'steam', 'razer', 'logitech'],
            'tv': ['samsung', 'lg', 'sony', 'tcl', 'vizio', 'hisense']
        },
        'furniture': {
            'seating': ['ikea', 'ashley', 'la-z-boy', 'herman miller', 'steelcase'],
            'storage': ['ikea', 'cb2', 'west elm', 'pottery barn', 'crate and barrel'],
            'bedroom': ['ikea', 'ashley', 'pottery barn', 'west elm', 'cb2'],
            'dining': ['ikea', 'pottery barn', 'west elm', 'crate and barrel'],
            'office': ['herman miller', 'steelcase', 'ikea', 'hon', 'humanscale']
        },
        'automotive': {
            'sedans': ['honda', 'toyota', 'nissan', 'hyundai', 'mazda'],
            'suvs': ['toyota', 'honda', 'ford', 'chevrolet', 'jeep'],
            'trucks': ['ford', 'chevrolet', 'dodge', 'toyota', 'nissan'],
            'luxury': ['bmw', 'mercedes', 'audi', 'lexus', 'acura', 'infiniti'],
            'sports': ['bmw', 'mercedes', 'audi', 'porsche', 'corvette']
        },
        'appliances': {
            'kitchen': ['kitchenaid', 'whirlpool', 'ge', 'frigidaire', 'bosch'],
            'laundry': ['whirlpool', 'lg', 'samsung', 'ge', 'maytag'],
            'small': ['cuisinart', 'kitchenaid', 'breville', 'ninja', 'vitamix']
        }
    }
    
    # Subcategory detection patterns
    self.subcategory_patterns = {
        'electronics': {
            'iphone|android|smartphone|phone': 'phones',
            'macbook|laptop|notebook|chromebook': 'laptops',
            'ipad|tablet|surface': 'tablets',
            'camera|dslr|mirrorless': 'cameras',
            'headphones|earbuds|speaker|soundbar': 'audio',
            'xbox|playstation|nintendo|gaming|console': 'gaming',
            'tv|television|monitor|display': 'tv'
        },
        'furniture': {
            'chair|sofa|couch|sectional|recliner': 'seating',
            'dresser|bookshelf|cabinet|shelf|storage': 'storage',
            'bed|mattress|nightstand|bedroom': 'bedroom',
            'table|dining|kitchen': 'dining',
            'desk|office|chair': 'office'
        },
        'automotive': {
            'sedan|civic|camry|accord|altima': 'sedans',
            'suv|crossover|explorer|pilot|highlander': 'suvs',
            'truck|pickup|f150|silverado|ram': 'trucks',
            'bmw|mercedes|audi|lexus|luxury': 'luxury',
            'sports|convertible|coupe|mustang': 'sports'
        }
    }
    
    # Condition indicators
    self.condition_patterns = {
        'new': ['new', 'brand new', 'never used', 'in box', 'sealed'],
        'like_new': ['like new', 'excellent', 'mint', 'perfect', 'barely used'],
        'good': ['good', 'great', 'works well', 'minor wear'],
        'fair': ['fair', 'some wear', 'used', 'shows wear'],
        'parts': ['parts', 'repair', 'broken', 'not working', 'for parts']
    }
    
    # Color detection
    self.color_patterns = [
        'black', 'white', 'silver', 'gray', 'grey', 'blue', 'red', 'green',
        'yellow', 'orange', 'purple', 'pink', 'brown', 'gold', 'rose gold'
    ]
    
    # Size patterns
    self.size_patterns = {
        'clothing': ['xs', 'small', 'medium', 'large', 'xl', 'xxl'],
        'furniture': ['twin', 'full', 'queen', 'king', 'loveseat', '2-seater', '3-seater'],
        'electronics': ['32"', '55"', '65"', '75"', 'inch', '"']
    }
    
def normalize_title(self, title: str, category: str = 'general') -> str:
    """Normalize product title for better matching"""
    # Convert to lowercase
    normalized = title.lower()
    
    # Remove common Craigslist noise
    noise_patterns = [
        r'\b(obo|or best offer|firm|cash only|pick up only)\b',
        r'\b(like new|excellent condition|mint|barely used)\b',
        r'\b(downtown|uptown|near|close to|area)\b',
        r'\$\d+',  # Remove price mentions
        r'\b\d{4}\b',  # Remove years unless they're model numbers
        r'[^\w\s]'  # Remove punctuation
    ]
    
    for pattern in noise_patterns:
        normalized = re.sub(pattern, ' ', normalized)
    
    # Extract brand and model information
    normalized = self._extract_brand_model(normalized, category)
    
    # Remove stop words but keep important descriptors
    words = word_tokenize(normalized)
    filtered_words = [w for w in words if w not in self.stop_words or len(w) > 4]
    
    return ' '.join(filtered_words).strip()

def _extract_brand_model(self, text: str, category: str) -> str:
    """Extract brand and model information"""
    # Look for brand names in the category
    brands = self.brand_patterns.get(category, [])
    found_brands = [brand for brand in brands if brand in text]
    
    # Extract model numbers (sequences of letters and numbers)
    model_pattern = r'\b[a-z]*\d+[a-z]*\d*\b'
    models = re.findall(model_pattern, text)
    
    # Combine brand and model info with remaining text
    result_parts = found_brands + models
    
    # Add remaining meaningful words
    remaining_text = text
    for brand in found_brands:
        remaining_text = remaining_text.replace(brand, '')
    for model in models:
        remaining_text = remaining_text.replace(model, '')
        
    remaining_words = [w for w in remaining_text.split() if len(w) > 2]
    result_parts.extend(remaining_words)
    
def extract_product_details(self, listing: CraigslistListing) -> CraigslistListing:
    """Extract detailed product information for enhanced filtering"""
    text = (listing.title + " " + listing.description).lower()
    
    # Extract brand
    listing.brand = self._extract_brand(text, listing.category)
    
    # Extract subcategory
    listing.subcategory = self._extract_subcategory(text, listing.category)
    
    # Extract condition
    listing.condition = self._extract_condition(text)
    
    # Extract color
    listing.color = self._extract_color(text)
    
    # Extract size
    listing.size = self._extract_size(text, listing.category)
    
    # Extract material (for furniture)
    listing.material = self._extract_material(text)
    
    # Estimate age
    listing.age_estimate = self._estimate_age(text)
    
    # Calculate price range
    listing.price_range = self._calculate_price_range(listing.price, listing.category)
    
    # Extract keywords
    listing.keywords = self._extract_keywords(text)
    
    # Extract image URLs if available
    listing.image_urls = self._extract_image_urls(listing.description)
    
    return listing

def _extract_brand(self, text: str, category: str) -> str:
    """Extract brand from text"""
    category_brands = self.brand_patterns.get(category, {})
    
    # Check all subcategories within the main category
    for subcategory, brands in category_brands.items():
        for brand in brands:
            if brand in text:
                return brand.title()
    
    return ""

def _extract_subcategory(self, text: str, category: str) -> str:
    """Extract subcategory from text"""
    patterns = self.subcategory_patterns.get(category, {})
    
    for pattern, subcategory in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return subcategory
    
    return ""

def _extract_condition(self, text: str) -> str:
    """Extract condition from text"""
    for condition, indicators in self.condition_patterns.items():
        for indicator in indicators:
            if indicator in text:
                return condition
    
    return "used"  # Default assumption

def _extract_color(self, text: str) -> str:
    """Extract color from text"""
    for color in self.color_patterns:
        if color in text:
            return color
    
    return ""

def _extract_size(self, text: str, category: str) -> str:
    """Extract size information"""
    size_patterns = self.size_patterns.get(category, [])
    
    for size in size_patterns:
        if size in text:
            return size
    
    # Look for numeric sizes (like TV inches, clothing sizes)
    size_match = re.search(r'(\d+)["′″]|\b(\d+)\s*inch', text)
    if size_match:
        return f"{size_match.group(1) or size_match.group(2)}\""
    
    return ""

def _extract_material(self, text: str) -> str:
    """Extract material information (especially for furniture)"""
    materials = ['wood', 'metal', 'glass', 'leather', 'fabric', 'plastic', 
                'oak', 'pine', 'mahogany', 'steel', 'aluminum', 'iron']
    
    for material in materials:
        if material in text:
            return material
    
    return ""

def _estimate_age(self, text: str) -> str:
    """Estimate item age based on text clues"""
    new_indicators = ['new', 'brand new', 'never used', 'unopened', 'sealed']
    like_new_indicators = ['like new', 'excellent', 'mint', 'barely used']
    vintage_indicators = ['vintage', 'antique', 'retro', 'classic']
    
    for indicator in new_indicators:
        if indicator in text:
            return "new"
    
    for indicator in like_new_indicators:
        if indicator in text:
            return "like_new"
    
    for indicator in vintage_indicators:
        if indicator in text:
            return "vintage"
    
    return "used"

def _calculate_price_range(self, price: float, category: str) -> str:
    """Calculate price range category"""
    # Define price thresholds by category
    thresholds = {
        'electronics': {'budget': 100, 'mid_range': 500, 'premium': 1500},
        'furniture': {'budget': 200, 'mid_range': 800, 'premium': 2000},
        'automotive': {'budget': 5000, 'mid_range': 20000, 'premium': 50000},
        'appliances': {'budget': 150, 'mid_range': 600, 'premium': 1500}
    }
    
    category_thresholds = thresholds.get(category, {'budget': 100, 'mid_range': 500, 'premium': 1500})
    
    if price <= category_thresholds['budget']:
        return "budget"
    elif price <= category_thresholds['mid_range']:
        return "mid_range"
    elif price <= category_thresholds['premium']:
        return "premium"
    else:
        return "luxury"

def _extract_keywords(self, text: str) -> List[str]:
    """Extract relevant keywords for search"""
    # Remove common noise words
    words = word_tokenize(text)
    keywords = []
    
    for word in words:
        if (len(word) > 2 and 
            word not in self.stop_words and 
            not word.isdigit() and 
            word.isalnum()):
            keywords.append(word.lower())
    
    # Remove duplicates and return top 10 most relevant
    return list(set(keywords))[:10]

def _extract_image_urls(self, description: str) -> List[str]:
    """Extract image URLs from description"""
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+\.(?:jpg|jpeg|png|gif|webp)')
    return url_pattern.findall(description)

def calculate_similarity(self, title1: str, title2: str) -> float:
    """Calculate semantic similarity between two titles"""
    # Get embeddings
    embeddings = self.model.encode([title1, title2])
    
    # Calculate cosine similarity
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    # Also use fuzzy matching as a backup
    fuzzy_score = SequenceMatcher(None, title1, title2).ratio()
    
    # Combine both scores (weighted average)
    combined_score = (similarity * 0.7) + (fuzzy_score * 0.3)
    
    return combined_score

def find_best_matches(self, craigslist_title: str, ebay_titles: List[str], 
                     threshold: float = 0.6) -> List[Tuple[str, float]]:
    """Find best matching eBay titles for a Craigslist listing"""
    normalized_cl = self.normalize_title(craigslist_title)
    
    matches = []
    for ebay_title in ebay_titles:
        normalized_eb = self.normalize_title(ebay_title)
        similarity = self.calculate_similarity(normalized_cl, normalized_eb)
        
        if similarity >= threshold:
            matches.append((ebay_title, similarity))
    
    # Sort by similarity score descending
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:5]  # Return top 5 matches
```

class CraigslistScraper:
“”“Handles Craigslist RSS feed processing”””

```
def __init__(self):
    self.base_urls = {
        'electronics': 'https://{city}.craigslist.org/search/ela?format=rss',
        'furniture': 'https://{city}.craigslist.org/search/fua?format=rss',
        'automotive': 'https://{city}.craigslist.org/search/cta?format=rss',
        'appliances': 'https://{city}.craigslist.org/search/ppa?format=rss'
    }
    
async def fetch_listings(self, cities: List[str], categories: List[str]) -> List[CraigslistListing]:
    """Fetch listings from Craigslist RSS feeds"""
    listings = []
    
    async with aiohttp.ClientSession() as session:
        for city in cities:
            for category in categories:
                url = self.base_urls[category].format(city=city)
                try:
                    listings.extend(await self._parse_rss_feed(session, url, category))
                    await asyncio.sleep(1)  # Be respectful with requests
                except Exception as e:
                    logger.error(f"Error fetching {city} {category}: {e}")
    
    return listings

async def _parse_rss_feed(self, session: aiohttp.ClientSession, 
                        url: str, category: str) -> List[CraigslistListing]:
    """Parse individual RSS feed"""
    async with session.get(url) as response:
        if response.status == 200:
            content = await response.text()
            feed = feedparser.parse(content)
            
            listings = []
            for entry in feed.entries:
                try:
                    listing = self._parse_entry(entry, category)
                    if listing:
                        listings.append(listing)
                except Exception as e:
                    logger.warning(f"Error parsing entry: {e}")
            
            return listings
    return []

def _parse_entry(self, entry, category: str) -> Optional[CraigslistListing]:
    """Parse individual RSS entry into CraigslistListing"""
    try:
        # Extract price from title
        price_match = re.search(r'\$(\d+(?:,\d+)?)', entry.title)
        price = float(price_match.group(1).replace(',', '')) if price_match else 0.0
        
        # Generate unique ID
        listing_id = hashlib.md5(entry.link.encode()).hexdigest()
        
        # Parse location from title or description
        location = self._extract_location(entry.title)
        
        # Create base listing
        listing = CraigslistListing(
            id=listing_id,
            title=entry.title,
            price=price,
            description=getattr(entry, 'description', ''),
            location=location,
            url=entry.link,
            posted_date=datetime.now(),  # RSS doesn't always have precise dates
            category=category,
            keywords=[],
            image_urls=[]
        )
        
        return listing
    except Exception as e:
        logger.warning(f"Error parsing entry: {e}")
        return None

def _extract_location(self, title: str) -> str:
    """Extract location information from title"""
    # Look for common location patterns
    location_patterns = [
        r'\(([^)]+)\)$',  # Location in parentheses at end
        r'- ([^-]+)$',    # Location after dash at end
    ]
    
    for pattern in location_patterns:
        match = re.search(pattern, title)
        if match:
            return match.group(1).strip()
    
    return "Unknown"
```

class EbayClient:
“”“eBay API client for fetching sold listings”””

```
def __init__(self, app_id: str):
    self.app_id = app_id
    self.base_url = "https://svcs.ebay.com/services/search/FindingService/v1"
    
async def search_sold_listings(self, query: str, category_id: str = None, 
                             days_back: int = 90) -> List[EbaySoldListing]:
    """Search eBay sold listings"""
    params = {
        'OPERATION-NAME': 'findCompletedItems',
        'SERVICE-VERSION': '1.0.0',
        'SECURITY-APPNAME': self.app_id,
        'RESPONSE-DATA-FORMAT': 'JSON',
        'keywords': query,
        'itemFilter(0).name': 'SoldItemsOnly',
        'itemFilter(0).value': 'true',
        'itemFilter(1).name': 'EndTimeFrom',
        'itemFilter(1).value': (datetime.now() - timedelta(days=days_back)).isoformat(),
        'paginationInput.entriesPerPage': '100'
    }
    
    if category_id:
        params['categoryId'] = category_id
    
    async with aiohttp.ClientSession() as session:
        async with session.get(self.base_url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                return self._parse_ebay_response(data)
    
    return []

def _parse_ebay_response(self, data: dict) -> List[EbaySoldListing]:
    """Parse eBay API response"""
    listings = []
    
    try:
        search_result = data['findCompletedItemsResponse'][0]['searchResult'][0]
        if 'item' in search_result:
            items = search_result['item']
            
            for item in items:
                try:
                    sold_price = float(item['sellingStatus'][0]['currentPrice'][0]['__value__'])
                    shipping_cost = 0.0
                    
                    if 'shippingInfo' in item and 'shippingServiceCost' in item['shippingInfo'][0]:
                        shipping_cost = float(item['shippingInfo'][0]['shippingServiceCost'][0]['__value__'])
                    
                    listing = EbaySoldListing(
                        title=item['title'][0],
                        sold_price=sold_price,
                        sold_date=datetime.fromisoformat(item['listingInfo'][0]['endTime'][0].replace('Z', '+00:00')),
                        condition=item.get('condition', [{'conditionDisplayName': ['Unknown']}])[0]['conditionDisplayName'][0],
                        shipping_cost=shipping_cost,
                        url=item['viewItemURL'][0]
                    )
                    listings.append(listing)
                except (KeyError, IndexError, ValueError) as e:
                    logger.warning(f"Error parsing eBay item: {e}")
    except (KeyError, IndexError) as e:
        logger.error(f"Error parsing eBay response: {e}")
    
    return listings
```

class DatabaseManager:
“”“Handles data persistence”””

```
def __init__(self, db_path: str = "price_matching.db"):
    self.db_path = db_path
    self.init_database()

def init_database(self):
    """Initialize database tables"""
    with sqlite3.connect(self.db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS craigslist_listings (
                id TEXT PRIMARY KEY,
                title TEXT,
                price REAL,
                description TEXT,
                location TEXT,
                url TEXT,
                posted_date TEXT,
                category TEXT,
                normalized_title TEXT,
                subcategory TEXT,
                brand TEXT,
                model TEXT,
                condition TEXT,
                color TEXT,
                size TEXT,
                material TEXT,
                age_estimate TEXT,
                price_range TEXT,
                keywords TEXT,  -- JSON array as string
                image_urls TEXT,  -- JSON array as string
                deal_score REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS ebay_sold_listings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT,
                sold_price REAL,
                sold_date TEXT,
                condition TEXT,
                shipping_cost REAL,
                url TEXT,
                query_used TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS price_analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                craigslist_id TEXT,
                avg_sold_price REAL,
                price_difference REAL,
                confidence_score REAL,
                recommendation TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (craigslist_id) REFERENCES craigslist_listings (id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_craigslist_category ON craigslist_listings(category);
            CREATE INDEX IF NOT EXISTS idx_craigslist_subcategory ON craigslist_listings(subcategory);
            CREATE INDEX IF NOT EXISTS idx_craigslist_brand ON craigslist_listings(brand);
            CREATE INDEX IF NOT EXISTS idx_craigslist_price_range ON craigslist_listings(price_range);
            CREATE INDEX IF NOT EXISTS idx_craigslist_condition ON craigslist_listings(condition);
            CREATE INDEX IF NOT EXISTS idx_craigslist_location ON craigslist_listings(location);
            CREATE INDEX IF NOT EXISTS idx_craigslist_deal_score ON craigslist_listings(deal_score);
            CREATE INDEX IF NOT EXISTS idx_ebay_query ON ebay_sold_listings(query_used);
        """)

def save_craigslist_listing(self, listing: CraigslistListing):
    """Save Craigslist listing to database"""
    with sqlite3.connect(self.db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO craigslist_listings 
            (id, title, price, description, location, url, posted_date, category, 
             normalized_title, subcategory, brand, model, condition, color, size, 
             material, age_estimate, price_range, keywords, image_urls, deal_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            listing.id, listing.title, listing.price, listing.description,
            listing.location, listing.url, listing.posted_date.isoformat(),
            listing.category, listing.normalized_title, listing.subcategory,
            listing.brand, listing.model, listing.condition, listing.color,
            listing.size, listing.material, listing.age_estimate, listing.price_range,
            json.dumps(listing.keywords or []), json.dumps(listing.image_urls or []),
            listing.deal_score
        ))

def save_ebay_listings(self, listings: List[EbaySoldListing], query: str):
    """Save eBay sold listings to database"""
    with sqlite3.connect(self.db_path) as conn:
        for listing in listings:
            conn.execute("""
                INSERT INTO ebay_sold_listings 
                (title, sold_price, sold_date, condition, shipping_cost, url, query_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                listing.title, listing.sold_price, listing.sold_date.isoformat(),
                listing.condition, listing.shipping_cost, listing.url, query
            ))

def save_price_analysis(self, analysis: PriceAnalysis):
    """Save price analysis to database"""
    with sqlite3.connect(self.db_path) as conn:
        conn.execute("""
            INSERT INTO price_analyses 
            (craigslist_id, avg_sold_price, price_difference, confidence_score, recommendation)
            VALUES (?, ?, ?, ?, ?)
        """, (
            analysis.craigslist_listing.id, analysis.avg_sold_price,
            analysis.price_difference, analysis.confidence_score, analysis.recommendation
        ))
```

class PriceMatchingService:
“”“Main service orchestrating the price matching process”””

```
def __init__(self, ebay_app_id: str):
    self.craigslist_scraper = CraigslistScraper()
    self.ebay_client = EbayClient(ebay_app_id)
    self.product_matcher = ProductMatcher()
    self.db = DatabaseManager()
    
async def run_full_analysis(self, cities: List[str], categories: List[str]) -> List[PriceAnalysis]:
    """Run complete price matching analysis"""
    logger.info("Starting full price matching analysis")
    
    # Step 1: Fetch Craigslist listings
    logger.info("Fetching Craigslist listings...")
    cl_listings = await self.craigslist_scraper.fetch_listings(cities, categories)
    logger.info(f"Found {len(cl_listings)} Craigslist listings")
    
    # Step 2: Process each listing
    analyses = []
    for listing in cl_listings:
        try:
            # Normalize the title
            listing.normalized_title = self.product_matcher.normalize_title(
                listing.title, listing.category
            )
            
            # Save to database
            self.db.save_craigslist_listing(listing)
            
            # Search eBay for similar items
            ebay_listings = await self.ebay_client.search_sold_listings(
                listing.normalized_title
            )
            
            if ebay_listings:
                # Save eBay data
                self.db.save_ebay_listings(ebay_listings, listing.normalized_title)
                
                # Find best matches
                matches = self.product_matcher.find_best_matches(
                    listing.title, [eb.title for eb in ebay_listings]
                )
                
                if matches:
                    # Filter eBay listings to only matched ones
                    matched_ebay = [eb for eb in ebay_listings 
                                  if any(eb.title == match[0] for match, score in matches)]
                    
                    # Calculate analysis
                    analysis = self._analyze_pricing(listing, matched_ebay, matches)
                    analyses.append(analysis)
                    
                    # Save analysis
                    self.db.save_price_analysis(analysis)
            
            # Be respectful with API calls
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error processing listing {listing.id}: {e}")
    
    logger.info(f"Completed analysis for {len(analyses)} listings")
    return analyses

def _analyze_pricing(self, cl_listing: CraigslistListing, 
                    ebay_listings: List[EbaySoldListing],
                    matches: List[Tuple[str, float]]) -> PriceAnalysis:
    """Analyze pricing for a Craigslist listing"""
    if not ebay_listings:
        return PriceAnalysis(
            craigslist_listing=cl_listing,
            ebay_matches=[],
            avg_sold_price=0.0,
            price_difference=0.0,
            confidence_score=0.0,
            recommendation="No comparable eBay sales found"
        )
    
    # Calculate average sold price
    total_price = sum(eb.sold_price + eb.shipping_cost for eb in ebay_listings)
    avg_sold_price = total_price / len(ebay_listings)
    
    # Calculate price difference
    price_difference = cl_listing.price - avg_sold_price
    price_difference_pct = (price_difference / avg_sold_price) * 100 if avg_sold_price > 0 else 0
    
    # Calculate confidence score based on match quality and sample size
    avg_match_score = sum(score for _, score in matches) / len(matches)
    sample_size_factor = min(len(ebay_listings) / 10, 1.0)  # Max factor of 1.0 at 10+ samples
    confidence_score = (avg_match_score * 0.7) + (sample_size_factor * 0.3)
    
    # Generate recommendation
    if price_difference_pct < -20:
        recommendation = "Excellent deal - significantly below market price"
    elif price_difference_pct < -10:
        recommendation = "Good deal - below market price"
    elif price_difference_pct < 10:
        recommendation = "Fair price - close to market value"
    elif price_difference_pct < 25:
        recommendation = "Overpriced - above market value"
    else:
        recommendation = "Significantly overpriced - well above market value"
    
    return PriceAnalysis(
        craigslist_listing=cl_listing,
        ebay_matches=ebay_listings,
        avg_sold_price=avg_sold_price,
        price_difference=price_difference,
        confidence_score=confidence_score,
        recommendation=recommendation
    )
```

# Example usage

async def main():
“”“Example usage of the price matching system”””
# Initialize the service (you’ll need a real eBay App ID)
service = PriceMatchingService(ebay_app_id=“YOUR_EBAY_APP_ID”)

```
# Define cities and categories to monitor
cities = ['sfbay', 'losangeles', 'seattle', 'chicago', 'newyork']
categories = ['electronics', 'furniture']

# Run analysis
analyses = await service.run_full_analysis(cities, categories)

# Print results
for analysis in analyses[:5]:  # Show first 5 results
    print(f"\n{'='*50}")
    print(f"Craigslist: {analysis.craigslist_listing.title}")
    print(f"Price: ${analysis.craigslist_listing.price}")
    print(f"Average eBay Sold: ${analysis.avg_sold_price:.2f}")
    print(f"Difference: ${analysis.price_difference:.2f}")
    print(f"Confidence: {analysis.confidence_score:.2f}")
    print(f"Recommendation: {analysis.recommendation}")
```

if **name** == “**main**”:
# Run the example
asyncio.run(main())