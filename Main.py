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

# Import your backend (assuming it’s in a file called price_matching_backend.py)

from price_matching_backend import (
    PriceMatchingService, 
    CraigslistListing, 
    PriceAnalysis, 
    DatabaseManager
)

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
allow_origins=[”*”],  # In production, specify your iOS app’s domain
allow_credentials=True,
allow_methods=[“GET”, “POST”, “PUT”, “DELETE”],
allow_headers=[”*”],
)

# Environment variables

EBAY_APP_ID = os.getenv(“EBAY_APP_ID”, “test-app-id”)
DATABASE_PATH = os.getenv(“DATABASE_PATH”, “price_matching.db”)

# Initialize services

service = PriceMatchingService(ebay_app_id=EBAY_APP_ID)
db = DatabaseManager(db_path=DATABASE_PATH)

# Pydantic models for request/response validation

class AnalysisRequest(BaseModel):
cities: List[str] = [“sfbay”, “losangeles”, “seattle”, “chicago”, “newyork”]
categories: List[str] = [“electronics”, “furniture”, “appliances”]

class ListingFilters(BaseModel):
category: Optional[str] = None
subcategory: Optional[str] = None
brand: Optional[str] = None
condition: Optional[str] = None
color: Optional[str] = None
price_range: Optional[str] = None
min_price: Optional[float] = None
max_price: Optional[float] = None
location: Optional[str] = None
min_deal_score: Optional[float] = None
search_query: Optional[str] = None
sort_by: str = “deal_score”
sort_order: str = “DESC”
limit: int = 100
offset: int = 0

class ApiResponse(BaseModel):
success: bool
data: Any = None
message: str = “”
count: Optional[int] = None

# Startup event

@app.on_event(“startup”)
async def startup_event():
“”“Initialize database and services on startup”””
logger.info(“Starting Price Matching API…”)

```
# Ensure database is initialized
db.init_database()

logger.info("API startup complete")
```

# Root endpoint

@app.get(”/”, response_model=Dict[str, str])
async def root():
“”“API health check and welcome message”””
return {
“message”: “Price Matching API is running!”,
“version”: “1.0.0”,
“docs”: “/docs”,
“status”: “healthy”
}

# Health check endpoint

@app.get(”/health”)
async def health_check():
“”“Detailed health check”””
try:
# Test database connection
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

# Get filtered listings

@app.get(”/listings”, response_model=ApiResponse)
async def get_listings(
category: Optional[str] = Query(None, description=“Filter by category”),
subcategory: Optional[str] = Query(None, description=“Filter by subcategory”),
brand: Optional[str] = Query(None, description=“Filter by brand”),
condition: Optional[str] = Query(None, description=“Filter by condition”),
color: Optional[str] = Query(None, description=“Filter by color”),
price_range: Optional[str] = Query(None, description=“Filter by price range”),
min_price: Optional[float] = Query(None, description=“Minimum price”),
max_price: Optional[float] = Query(None, description=“Maximum price”),
location: Optional[str] = Query(None, description=“Filter by location”),
min_deal_score: Optional[float] = Query(None, description=“Minimum deal score”),
search_query: Optional[str] = Query(None, description=“Search in title/description”),
sort_by: str = Query(“deal_score”, description=“Sort field”),
sort_order: str = Query(“DESC”, description=“Sort order (ASC/DESC)”),
limit: int = Query(100, description=“Number of results”),
offset: int = Query(0, description=“Pagination offset”)
):
“”“Get filtered and sorted listings”””
try:
filters = {
“category”: category,
“subcategory”: subcategory,
“brand”: brand,
“condition”: condition,
“color”: color,
“price_range”: price_range,
“min_price”: min_price,
“max_price”: max_price,
“location”: location,
“min_deal_score”: min_deal_score,
“search_query”: search_query,
“sort_by”: sort_by,
“sort_order”: sort_order,
“limit”: limit,
“offset”: offset
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

# Get specific listing by ID

@app.get(”/listings/{listing_id}”, response_model=ApiResponse)
async def get_listing_detail(listing_id: str):
“”“Get detailed information for a specific listing”””
try:
listing = get_listing_by_id(listing_id)
if not listing:
raise HTTPException(status_code=404, detail=“Listing not found”)

```
    # Get price analysis for this listing
    analysis = get_price_analysis(listing_id)
    
    return ApiResponse(
        success=True,
        data={
            "listing": listing,
            "analysis": analysis
        },
        message="Listing details retrieved"
    )

except HTTPException:
    raise
except Exception as e:
    logger.error(f"Error getting listing {listing_id}: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
```

# Get filter options

@app.get(”/filters”, response_model=ApiResponse)
async def get_filter_options():
“”“Get available filter options for the UI”””
try:
options = service.get_filter_options()

```
    return ApiResponse(
        success=True,
        data=options,
        message="Filter options retrieved"
    )

except Exception as e:
    logger.error(f"Error getting filter options: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
```

# Get featured deals (high deal scores)

@app.get(”/deals”, response_model=ApiResponse)
async def get_featured_deals(
limit: int = Query(20, description=“Number of deals to return”),
min_score: float = Query(70.0, description=“Minimum deal score”)
):
“”“Get featured deals with high deal scores”””
try:
filters = {
“min_deal_score”: min_score,
“sort_by”: “deal_score”,
“sort_order”: “DESC”,
“limit”: limit
}

```
    deals = service.get_filtered_listings(filters)
    
    return ApiResponse(
        success=True,
        data=deals,
        count=len(deals),
        message=f"Found {len(deals)} featured deals"
    )

except Exception as e:
    logger.error(f"Error getting featured deals: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
```

# Search listings

@app.get(”/search”, response_model=ApiResponse)
async def search_listings(
q: str = Query(…, description=“Search query”),
category: Optional[str] = Query(None, description=“Filter by category”),
limit: int = Query(50, description=“Number of results”),
sort_by: str = Query(“deal_score”, description=“Sort field”)
):
“”“Search listings by title and description”””
try:
filters = {
“search_query”: q,
“category”: category,
“sort_by”: sort_by,
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

# Get statistics

@app.get(”/stats”, response_model=ApiResponse)
async def get_statistics():
“”“Get API statistics”””
try:
with sqlite3.connect(DATABASE_PATH) as conn:
cursor = conn.cursor()

```
        # Total listings
        cursor.execute("SELECT COUNT(*) FROM craigslist_listings")
        total_listings = cursor.fetchone()[0]
        
        # Listings by category
        cursor.execute("""
            SELECT category, COUNT(*) 
            FROM craigslist_listings 
            GROUP BY category 
            ORDER BY COUNT(*) DESC
        """)
        category_counts = dict(cursor.fetchall())
        
        # Average deal score
        cursor.execute("SELECT AVG(deal_score) FROM craigslist_listings WHERE deal_score > 0")
        avg_deal_score = cursor.fetchone()[0] or 0
        
        # Top deals count
        cursor.execute("SELECT COUNT(*) FROM craigslist_listings WHERE deal_score >= 70")
        top_deals_count = cursor.fetchone()[0]
        
        stats = {
            "total_listings": total_listings,
            "category_breakdown": category_counts,
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

# Trigger new analysis (admin endpoint)

@app.post(”/analyze”, response_model=ApiResponse)
async def trigger_analysis(
background_tasks: BackgroundTasks,
request: AnalysisRequest
):
“”“Trigger new analysis in the background”””
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

# Get analysis status

@app.get(”/analyze/status”)
async def get_analysis_status():
“”“Get status of the latest analysis”””
try:
with sqlite3.connect(DATABASE_PATH) as conn:
cursor = conn.cursor()

```
        # Get latest analysis timestamp
        cursor.execute("""
            SELECT MAX(created_at) FROM price_analyses
        """)
        latest_analysis = cursor.fetchone()[0]
        
        # Get total analyses today
        cursor.execute("""
            SELECT COUNT(*) FROM price_analyses 
            WHERE DATE(created_at) = DATE('now')
        """)
        analyses_today = cursor.fetchone()[0]
        
        return {
            "latest_analysis": latest_analysis,
            "analyses_today": analyses_today,
            "status": "idle"  # In a real app, you'd track actual background task status
        }

except Exception as e:
    logger.error(f"Error getting analysis status: {str(e)}")
    raise HTTPException(status_code=500, detail=str(e))
```

# Helper functions

def get_listing_by_id(listing_id: str) -> Optional[Dict]:
“”“Get a single listing by ID”””
with sqlite3.connect(DATABASE_PATH) as conn:
conn.row_factory = sqlite3.Row
cursor = conn.execute(
“SELECT * FROM craigslist_listings WHERE id = ?”,
(listing_id,)
)
row = cursor.fetchone()

```
    if row:
        result = dict(row)
        result['keywords'] = json.loads(result['keywords'] or '[]')
        result['image_urls'] = json.loads(result['image_urls'] or '[]')
        return result
    
    return None
```

def get_price_analysis(listing_id: str) -> Optional[Dict]:
“”“Get price analysis for a listing”””
with sqlite3.connect(DATABASE_PATH) as conn:
conn.row_factory = sqlite3.Row
cursor = conn.execute(
“SELECT * FROM price_analyses WHERE craigslist_id = ? ORDER BY created_at DESC LIMIT 1”,
(listing_id,)
)
row = cursor.fetchone()

```
    return dict(row) if row else None
```

async def run_analysis_task(cities: List[str], categories: List[str]):
“”“Background task for running analysis”””
try:
logger.info(f”Starting analysis for cities: {cities}, categories: {categories}”)
analyses = await service.run_full_analysis(cities, categories)
logger.info(f”Analysis completed: {len(analyses)} listings processed”)
except Exception as e:
logger.error(f”Analysis task failed: {str(e)}”)

# Exception handlers

@app.exception_handler(404)
async def not_found_handler(request, exc):
return JSONResponse(
status_code=404,
content=ApiResponse(
success=False,
message=“Resource not found”
).dict()
)

@app.exception_handler(500)
async def internal_error_handler(request, exc):
return JSONResponse(
status_code=500,
content=ApiResponse(
success=False,
message=“Internal server error”
).dict()
)

if **name** == “**main**”:
import uvicorn

```
# Get port from environment (for deployment)
port = int(os.getenv("PORT", 8000))

# Run the app
uvicorn.run(
    app, 
    host="0.0.0.0", 
    port=port,
    log_level="info"
)
```
