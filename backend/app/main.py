from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

# Initialize the FastAPI app
app = FastAPI()

# Mount the static directory to serve files like favicons
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to Towards People API"}

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

