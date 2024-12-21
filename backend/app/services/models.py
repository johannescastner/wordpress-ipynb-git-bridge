from sqlalchemy import Column, Integer, String, JSON
from app.services.database import Base

# Define the Blog model
class Blog(Base):
    __tablename__ = "blogs"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False, index=True)
    content = Column(String, nullable=False)
    slug = Column(String, unique=True, index=True)
    images = Column(JSON, nullable=True)  # Dictionary to store image URLs
