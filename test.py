from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import asyncio
import os
import json
import sqlite3
from datetime import datetime
import logging
import aiohttp
import feedparser
import re
import hashlib
from difflib import SequenceMatcher
import requests
import time
import numpy as np

# Configure logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(**name**)

# Initialize FastAPI app

app = FastAPI(
title=“Price Matching API”,
description=“API for matching Craigslist listings with eBay sold prices”,
version=“1.0.0”,
docs_url=”/docs”,
redoc_url=”/redoc”
)

# Enable CORS for iOS app

app.add_middleware(
CORSMiddleware,
allow_origins=[”*”],
allow_credentials=True,
allow_methods=[“GET”, “POST”, “PUT”, “DELETE”],
allow_headers=[”*”],
)

# Environment variables

EBAY_APP_ID = os.getenv(“EBAY_APP_ID”, “test-app-id”)
DATABASE_PATH = os.getenv(“DATABASE_PATH”, “price_matching.db”)

# Pydantic models

class AnalysisRequest(BaseModel):
cities: List[str] = [“sfbay”, “losangeles”, “seattle”]
categories: List[str] = [“electronics”, “furniture”]

class ApiResponse(BaseModel):
success: bool
data: Any = None
message: str = “”
count: Optional[int] = None

# Simplified data classes (without external dependencies)

class CraigslistListing:
def **init**(self, id, title, price, description, location, url, posted_date, category):
self.id = id
self.title = title
self.price = price
self.description = description
self.location = location
self.url = url
self.posted_date = posted_date
self.category = category
self.subcategory = ""
self.brand = “”
self.condition = “”
self.color = “”
self.deal_score = 0.0

class EbaySoldListing:
def **init**(self, title, sold_price, sold_date, condition, shipping_cost, url):
self.title = title
self.sold_price = sold_price
self.sold_date = sold_date
self.condition = condition
self.shipping_cost = shipping_cost
self.url = url

# Simplified ProductMatcher

class ProductMatcher:
def **init**(self):
self.brand_patterns = {
‘electronics’: [‘apple’, ‘samsung’, ‘sony’, ‘lg’, ‘canon’, ‘nikon’],
‘furniture’: [‘ikea’, ‘ashley’, ‘pottery barn’, ‘west elm’],
‘automotive’: [‘honda’, ‘toyota’, ‘ford’, ‘chevrolet’, ‘bmw’]
}

```
def normalize_title(self, title, category="general"):
    # Simple normalization
    normalized = title.lower()
    # Remove price mentions and common noise
    noise_patterns = [
        r'\$\d+',
        r'\b(obo|or best offer|firm)\b',
        r'\b(like new|excellent condition)\b'
    ]
    for pattern in noise_patterns:
        normalized = re.sub(pattern, ' ', normalized)
    return ' '.join(normalized.split())

def extract_brand(self, text, category):
    brands = self.brand_patterns.get(category, [])
    for brand in brands:
        if brand in text.lower():
            return brand.title()
    return ""

def extract_condition(self, text):
    text = text.lower()
    if any(word in text for word in ['new', 'brand new', 'never used']):
        return "new"
    elif any(word in text for word in ['like new', 'excellent', 'mint']):
        return "like_new"
    elif any(word in text for word in ['good', 'great']):
        return "good"
    elif any(word in text for word in ['fair', 'used']):
        return "fair"
    return "used"

def calculate_similarity(self, title1, title2):
    return SequenceMatcher(None, title1.lower(), title2.lower()).ratio()
```

# Simplified CraigslistScraper

class CraigslistScraper:
def **init**(self):
self.base_urls = {
‘electronics’: ‘https://{city}.craigslist.org/search/ela?format=rss’,
‘furniture’: ‘https://{city}.craigslist.org/search/fua?format=rss’
}

```
async def fetch_listings(self, cities, categories):
    listings = []
    async with aiohttp.ClientSession() as session:
        for city in cities[:2]:  # Limit to 2 cities for now
            for category in categories:
                if category in self.base_urls:
                    url = self.base_urls[category].format(city=city)
                    try:
                        listings.extend(await self._parse_rss_feed(session, url, category))
                        await asyncio.sleep(1)
                    except Exception as e:
                        logger.error(f"Error fetching {city} {category}: {e}")
    return listings

async def _parse_rss_feed(self, session, url, category):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                content = await response.text()
                feed = feedparser.parse(content)
                
                listings = []
                for entry in feed.entries[:10]:  # Limit to 10 entries
                    try:
                        listing = self._parse_entry(entry, category)
                        if listing:
                            listings.append(listing)
                    except Exception as e:
                        logger.warning(f"Error parsing entry: {e}")
                return listings
    except Exception as e:
        logger.error(f"Error parsing RSS feed: {e}")
    return []

def _parse_entry(self, entry, category):
    try:
        # Extract price from title
        price_match = re.search(r'\$(\d+(?:,\d+)?)', entry.title)
        price = float(price_match.group(1).replace(',', '')) if price_match else 0.0
        
        # Generate unique ID
        listing_id = hashlib.md5(entry.link.encode()).hexdigest()
        
        # Extract location
        location = "Unknown"
        location_match = re.search(r'\(([^)]+)\)$', entry.title)
        if location_match:
            location = location_match.group(1).strip()
        
        return CraigslistListing(
            id=listing_id,
            title=entry.title,
            price=price,
            description=getattr(entry, 'description', ''),
            location=location,
            url=entry.link,
            posted_date=datetime.now(),
            category=category
        )
    except Exception as e:
        logger.warning(f"Error parsing entry: {e}")
        return None
```

# Simplified eBayClient

class EbayClient:
def **init**(self, app_id):
self.app_id = app_id
self.base_url = “https://svcs.ebay.com/services/search/FindingService/v1”

```
async def search_sold_listings(self, query, days_back=90):
    if self.app_id == "test-app-id":
        # Return mock data if no real API key
        return [
            EbaySoldListing(
                title=f"Similar to {query}",
                sold_price=100.0,
                sold_date=datetime.now(),
                condition="Used",
                shipping_cost=10.0,
                url="https://ebay.com/example"
            )
        ]
    
    # Real eBay API call would go here
    return []
```

# Database Manager

class DatabaseManager:
def **init**(self, db_path=“price_matching.db”):
self.db_path = db_path
self.init_database()

```
def init_database(self):
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
                subcategory TEXT,
                brand TEXT,
                condition TEXT,
                color TEXT,
                deal_score REAL DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE INDEX IF NOT EXISTS idx_category ON craigslist_listings(category);
            CREATE INDEX IF NOT EXISTS idx_deal_score ON craigslist_listings(deal_score);
        """)

def save_listing(self, listing):
    with sqlite3.connect(self.db_path) as conn:
        conn.execute("""
            INSERT OR REPLACE INTO craigslist_listings 
            (id, title, price, description, location, url, posted_date, category, 
             subcategory, brand, condition, color, deal_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            listing.id, listing.title, listing.price, listing.description,
            listing.location, listing.url, listing.posted_date.isoformat(),
            listing.category, listing.subcategory, listing.brand,
            listing.condition, listing.color, listing.deal_score
        ))

def get_listings(self, filters):
    query = "SELECT * FROM craigslist_listings WHERE 1=1"
    params = []
    
    if filters.get('category'):
        query += " AND category = ?"
        params.append(filters['category'])
    
    if filters.get('min_price'):
        query += " AND price >= ?"
        params.append(filters['min_price'])
    
    if filters.get('max_price'):
        query += " AND price <= ?"
        params.append(filters['max_price'])
    
    if filters.get('search_query'):
        query += " AND (title LIKE ? OR description LIKE ?)"
        search_term = f"%{filters['search_query']}%"
        params.extend([search_term, search_term])
    
    sort_by = filters.get('sort_by', 'deal_score')
    sort_order = filters.get('sort_order', 'DESC')
    query += f" ORDER BY {sort_by} {sort_order}"
    
    limit = filters.get('limit', 100)
    query += f" LIMIT {limit}"
    
    with sqlite3.connect(self.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
```

# Main Service

class PriceMatchingService:
def **init**(self, ebay_app_id):
self.craigslist_scraper = CraigslistScraper()
self.ebay_client = EbayClient(ebay_app_id)
self.product_matcher = ProductMatcher()
self.db = DatabaseManager()

```
async def run_analysis(self, cities, categories):
    logger.info("Starting price matching analysis")
    
    # Fetch Craigslist listings
    cl_listings = await self.craigslist_scraper.fetch_listings(cities, categories)
    logger.info(f"Found {len(cl_listings)} Craigslist listings")
    
    # Process each listing
    for listing in cl_listings:
        try:
            # Extract product details
            listing.brand = self.product_matcher.extract_brand(
                listing.title + " " + listing.description, listing.category
            )
            listing.condition = self.product_matcher.extract_condition(
                listing.title + " " + listing.description
            )
            
            # Simple deal score calculation
            if listing.price > 0:
                # Mock comparison with average prices
                avg_price = listing.price * 1.2  # Assume market is 20% higher
                savings = max(0, avg_price - listing.price)
                listing.deal_score = min(100, (savings / avg_price) * 100)
            
            # Save to database
            self.db.save_listing(listing)
            
        except Exception as e:
            logger.error(f"Error processing listing {listing.id}: {e}")
    
    return cl_listings

def get_filtered_listings(self, filters):
    return self.db.get_listings(filters)
```

# Initialize service

service = PriceMatchingService(ebay_app_id=EBAY_APP_ID)

# API Endpoints

@app.on_event(“startup”)
async def startup_event():
logger.info(“Starting Price Matching API…”)

@app.get(”/”, response_model=Dict[str, str])
async def root():
return {
“message”: “Price Matching API is running!”,
“version”: “1.0.0”,
“docs”: “/docs”,
“status”: “healthy”
}

@app.get(”/health”)
async def health_check():
try:
with sqlite3.connect(DATABASE_PATH) as conn:
cursor = conn.execute(“SELECT COUNT(*) FROM craigslist_listings”)
listing_count = cursor.fetchone()[0]

```
    return {
        "status": "healthy",
        "database": "connected",
        "listings_count": listing_count,
        "timestamp": datetime.now().isoformat()
    }
except Exception as e:
    raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")
```

@app.get(”/listings”, response_model=ApiResponse)
async def get_listings(
category: Optional[str] = Query(None, description=“Filter by category”),
min_price: Optional[float] = Query(None, description=“Minimum price”),
max_price: Optional[float] = Query(None, description=“Maximum price”),
search_query: Optional[str] = Query(None, description=“Search in title/description”),
sort_by: str = Query(“deal_score”, description=“Sort field”),
sort_order: str = Query(“DESC”, description=“Sort order (ASC/DESC)”),
limit: int = Query(50, description=“Number of results”)
):
try:
filters = {
“category”: category,
“min_price”: min_price,
“max_price”: max_price,
“search_query”: search_query,
“sort_by”: sort_by,
“sort_order”: sort_order,
“limit”: limit
}

```
    # Remove None values
    filters = {k: v for k, v in filters.items() if v is not None}
    
    listings = service.get_filtered_listings(filters)
    
    return ApiResponse(
        success=True,
        data=listings,
        count=len(listings),
        message=f"Found {len(listings)} listings"
    )

except Exception as e:
    logger.error(f"Error getting listings: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
```

@app.get(”/deals”, response_model=ApiResponse)
async def get_featured_deals(
limit: int = Query(20, description=“Number of deals to return”),
min_score: float = Query(50.0, description=“Minimum deal score”)
):
try:
filters = {
“sort_by”: “deal_score”,
“sort_order”: “DESC”,
“limit”: limit
}

```
    deals = service.get_filtered_listings(filters)
    # Filter by deal score
    deals = [d for d in deals if d.get('deal_score', 0) >= min_score]
    
    return ApiResponse(
        success=True,
        data=deals[:limit],
        count=len(deals[:limit]),
        message=f"Found {len(deals[:limit])} featured deals"
    )

except Exception as e:
    logger.error(f"Error getting featured deals: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
```

@app.get(”/search”, response_model=ApiResponse)
async def search_listings(
q: str = Query(…, description=“Search query”),
category: Optional[str] = Query(None, description=“Filter by category”),
limit: int = Query(50, description=“Number of results”)
):
try:
filters = {
“search_query”: q,
“category”: category,
“sort_by”: “deal_score”,
“sort_order”: “DESC”,
“limit”: limit
}

```
    # Remove None values
    filters = {k: v for k, v in filters.items() if v is not None}
    
    results = service.get_filtered_listings(filters)
    
    return ApiResponse(
        success=True,
        data=results,
        count=len(results),
        message=f"Found {len(results)} results for '{q}'"
    )

except Exception as e:
    logger.error(f"Error searching listings: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
```

@app.post(”/analyze”, response_model=ApiResponse)
async def trigger_analysis(
background_tasks: BackgroundTasks,
request: AnalysisRequest
):
try:
# Add analysis task to background
background_tasks.add_task(run_analysis_task, request.cities, request.categories)

```
    return ApiResponse(
        success=True,
        message=f"Analysis started for {len(request.cities)} cities and {len(request.categories)} categories"
    )

except Exception as e:
    logger.error(f"Error triggering analysis: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
```

@app.get(”/stats”, response_model=ApiResponse)
async def get_statistics():
try:
with sqlite3.connect(DATABASE_PATH) as conn:
cursor = conn.cursor()

```
        # Total listings
        cursor.execute("SELECT COUNT(*) FROM craigslist_listings")
        total_listings = cursor.fetchone()[0]
        
        # Average deal score
        cursor.execute("SELECT AVG(deal_score) FROM craigslist_listings WHERE deal_score > 0")
        avg_deal_score = cursor.fetchone()[0] or 0
        
        # Top deals count
        cursor.execute("SELECT COUNT(*) FROM craigslist_listings WHERE deal_score >= 70")
        top_deals_count = cursor.fetchone()[0]
        
        stats = {
            "total_listings": total_listings,
            "average_deal_score": round(avg_deal_score, 1),
            "top_deals_count": top_deals_count,
            "last_updated": datetime.now().isoformat()
        }
        
        return ApiResponse(
            success=True,
            data=stats,
            message="Statistics retrieved"
        )

except Exception as e:
    logger.error(f"Error getting statistics: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
```

# Background task

async def run_analysis_task(cities, categories):
try:
logger.info(f"Starting analysis for cities: {cities}, categories: {categories}")
analyses = await service.run_analysis(cities, categories)
logger.info(f"Analysis completed: {len(analyses)} listings processed")
except Exception as e:
logger.error(f"Analysis task failed: {str(e)}")

if **name** == "**main**":
import uvicorn
port = int(os.getenv("PORT", 8000)
uvicorn.run(app, host="0.0.0.0", port=port)
