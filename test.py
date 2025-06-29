from fastapi import FastAPI
app = FastAPI(title="Test API")
@app. get ("/")
def read_root():
return {"message": "Hello World"}
if __name__ == "__main__":
import uvicorn import os
port = intos. getenv ("PORT", 8000) 
uvicorn. run(app, host="0.0.0.0", port=port)
