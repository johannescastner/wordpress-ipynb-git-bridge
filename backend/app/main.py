from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from app.services.database import get_db
from app.services.models import Blog

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

# Blog endpoint to fetch all blogs
@app.get("/blog")
def get_blogs(db: Session = Depends(get_db)):
    blogs = db.query(Blog).all()
    return blogs

# Blog endpoint to fetch a blog by slug
@app.get("/blog/{slug}")
def get_blog(slug: str, db: Session = Depends(get_db)):
    blog = db.query(Blog).filter(Blog.slug == slug).first()
    if not blog:
        raise HTTPException(status_code=404, detail="Blog not found")
    return blog
