from flask import Flask, jsonify, request 
import sqlite3 
import os

app = Flask(__name__)

def init_db():
    conn = sqlite3.connect("listings.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS listings (
            id INTEGER PRIMARY KEY, 
            title TEXT, 
            price REAL, 
            category TEXT, 
            deal_score REAL

        ) 
    """) 
    conn.close()
  
init_db()

@app.route("/")
def root():
    return jsonify({"message": "API Running"})
  
@app.route("/health")
def health():
    return jsonify({"status": "healthy"})
  
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
