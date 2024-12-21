from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from app.services.database import get_db
from app.services.models import Blog

router = APIRouter()

# Fetch all blogs
@router.get("/blog")
def get_blogs(db: Session = Depends(get_db)):
    blogs = db.query(Blog).all()
    if not blogs:
        raise HTTPException(status_code=404, detail="No blogs found.")
    return [
        {
            "id": blog.id,
            "title": blog.title,
            "content": blog.content[:500],  # Return a snippet of content
            "slug": blog.slug,
            "images": blog.images,  # Include image URLs
        }
        for blog in blogs
    ]

# Fetch a specific blog by slug
@router.get("/blog/{slug}")
def get_blog(slug: str, db: Session = Depends(get_db)):
    blog = db.query(Blog).filter(Blog.slug == slug).first()
    if not blog:
        raise HTTPException(status_code=404, detail="Blog not found.")
    return {
        "id": blog.id,
        "title": blog.title,
        "content": blog.content,
        "images": blog.images,  # Include image URLs
        "slug": blog.slug,
    }
