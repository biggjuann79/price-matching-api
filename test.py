from fastapi import FastAPI
import os

app = FastAPI()

@app. get ("/")
def read_root():
    return {"message": "Hello World"}
  
@app.get ("/test" )
def test_endpoint():
     return {"status": "working"}
  
if __name__ == " __main__":
    import uvicorn
    port = intos.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
