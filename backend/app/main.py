from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Initialize the FastAPI app
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow requests from the frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all HTTP headers
)

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

# Blog endpoint
@app.get("/blog")
def get_blogs():
    return [
        {
            "id": 1,
            "title": "Diversity Assignments",
            "content": "A closer look at diversity in team assignments and its impact.",
        },
        {
            "id": 2,
            "title": "Ethical AI",
            "content": "Exploring the principles of ethical AI and how it shapes our world.",
        },
    ]
