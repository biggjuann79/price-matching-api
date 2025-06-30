from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import os

# -----------------------------------
# Initialize FastAPI
# -----------------------------------
app = FastAPI(
    title="Price Matching API",
    description="API for matching Craigslist listings with eBay sold prices",
    version="1.0.0"
)

# -----------------------------------
# Enable CORS (allow all origins)
# -----------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# Data Model
# -----------------------------------
class CraigslistListing(BaseModel):
    id: str
    title: str
    price: float
    description: str
    location: str
    url: str
    category: str
    deal_score: float = 0.0

# -----------------------------------
# SQLite Database Handler
# -----------------------------------
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

# -----------------------------------
# Initialize DB
# -----------------------------------
db = DatabaseManager()

# -----------------------------------
# Routes
# -----------------------------------
@app.get("/")
def root():
    return {"message": "Price Matching API is running"}

@app.get("/health")
def health():
    return {"status": "healthy", "database": "connected"}

@app.get("/listings")
def get_listings(limit: int = 50):
    try:
        data = db.get_listings(limit)
        return {"success": True, "data": data, "count": len(data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze")
def analyze():
    try:
        listing = CraigslistListing(
            id="sample1",
            title="MacBook Air M2 - Excellent Condition",
            price=799.00,
            description="Like new, barely used",
            location="Dallas, TX",
            url="https://example.com/listing",
            category="electronics",
            deal_score=88.5
        )
        db.save_listing(listing)
        return {"success": True, "message": "Sample listing analyzed and saved."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------------
# Run locally or on Railway
# -----------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
