from fastapi import APIRouter

router = APIRouter()

@router.get("/blog")
def get_blogs():
    return [{"id": 1, "title": "First Blog", "content": "Welcome to the blog!"}]

