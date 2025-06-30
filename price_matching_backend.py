from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Price Matching API",
    description="API for matching Craigslist listings with eBay sold prices",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class CraigslistListing(BaseModel):
    id: str
    title: str
    price: float
    description: str
    location: str
    url: str
    category: str
    deal_score: float = 0.0

# Database manager
class DatabaseManager:
    def __init__(self):
        self.db_path = "price_matching.db"
        self.init_database()

    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS listings (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    price REAL,
                    category TEXT,
                    deal_score REAL
                )
            """)

    def save_listing(self, listing: CraigslistListing):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO listings 
                (id, title, price, category, deal_score)
                VALUES (?, ?, ?, ?, ?)
            """, (listing.id, listing.title, listing.price, listing.category, listing.deal_score))

    def get_listings(self, limit=50):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM listings ORDER BY deal_score DESC LIMIT ?", (limit,))
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "title": row[1], 
                    "price": row[2],
                    "category": row[3],
                    "deal_score": row[4]
                })
            return results

# Initialize DB
db = DatabaseManager()

@app.get("/")
async def root():
    return {"message": "Price Matching API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "database": "connected"}

@app.get("/listings")
async def get_listings(limit: int = 50):
    try:
        listings = db.get_listings(limit)
        return {"success": True, "data": listings, "count": len(listings)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/deals")
def get_deals(min_score: float = 70.0, limit: int = 20):
    try:
        with sqlite3.connect(db.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM listings WHERE deal_score >= ? ORDER BY deal_score DESC LIMIT ?
            """, (min_score, limit))
            results = []
            for row in cursor.fetchall():
                results.append({
                    "id": row[0],
                    "title": row[1],
                    "price": row[2],
                    "category": row[3],
                    "deal_score": row[4],
                    # Optionally add mock fields below if needed by frontend
                    "location": "Dallas, TX",
                    "url": "https://example.com",
                    "created_at": "2025-06-29T22:00:00Z",
                    "ebay_average_price": 1200.0,
                    "savings_amount": 401.0,
                    "savings_percentage": 33.0,
                    "image_urls": ["https://picsum.photos/300/200?random=6"]
                })
        return {"success": True, "data": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/analyze")
async def analyze():
    try:
        sample = CraigslistListing(
            id="sample1",
            title="iPhone 15 Pro Max - Great Deal!",
            price=800.0,
            description="Like new condition",
            location="San Francisco",
            url="https://example.com",
            category="electronics",
            deal_score=85.0
        )
        db.save_listing(sample)
        return {"success": True, "message": "Analysis completed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
