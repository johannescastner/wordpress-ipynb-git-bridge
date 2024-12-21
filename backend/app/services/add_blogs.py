import os
import re
import sys
import time
import nbformat
from nbconvert import MarkdownExporter
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from google.auth.transport.requests import Request
from google.auth.credentials import Credentials
from google.cloud import storage
from app.services.database import Base
from app.services.models import Blog

# Load environment variables
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
dotenv_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=dotenv_path)

# Database Configuration
postgres_user = os.getenv("POSTGRES_USER")
postgres_password = os.getenv("POSTGRES_PASSWORD")
postgres_db = os.getenv("POSTGRES_DB")
postgres_host = os.getenv("POSTGRES_HOST")
postgres_port = os.getenv("POSTGRES_PORT")

SQLALCHEMY_DATABASE_URL = f"postgresql://{postgres_user}:{postgres_password}@{postgres_host}:{postgres_port}/{postgres_db}"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Google Cloud Storage configuration
BUCKET_NAME = "towards_people_blog_images"
storage_client = storage.Client()

# Function to create a slug
def create_slug(title):
    title = re.sub(r"([a-z])([A-Z])", r"\1_\2", title)
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+", "_", title)
    return title.lower().strip("_")

# Convert notebook to Markdown and extract resources
def convert_notebook_to_markdown(notebook_path):
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    markdown_exporter = MarkdownExporter()
    markdown_exporter.excluded_input_prompt = True
    markdown, resources = markdown_exporter.from_notebook_node(notebook)

    return markdown, resources

# Generate signed URLs for images in GCS
def generate_signed_url(bucket_name, blob_name, expiration=3600):
    """Generate a signed URL for the blob."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Create a signed URL using HMAC if ADC credentials lack service_account_email
    if isinstance(storage_client._credentials, Credentials):
        signer_email = os.getenv("GOOGLE_SIGNER_EMAIL")
        if not signer_email:
            raise EnvironmentError(
                "GOOGLE_SIGNER_EMAIL must be set in the environment to generate signed URLs."
            )
        credentials = storage_client._credentials

        # Ensure credentials are refreshed
        request = Request()
        credentials.refresh(request)

        # Build the string-to-sign
        expiration_time = int(time.time()) + expiration
        string_to_sign = f"GET\n\n\n{expiration_time}\n/{bucket_name}/{blob_name}"
        signed_blob = credentials.sign_bytes(string_to_sign.encode("utf-8"))
        signature = storage._helpers._base64_encode(signed_blob)

        signed_url = (
            f"https://storage.googleapis.com/{bucket_name}/{blob_name}"
            f"?GoogleAccessId={signer_email}&Expires={expiration_time}&Signature={signature}"
        )
        return signed_url

    # Use default `generate_signed_url` for service account credentials
    return blob.generate_signed_url(expiration=expiration)

# Upload images to Google Cloud Storage and get signed URLs
def upload_images_to_gcs(resources, slug):
    image_urls = {}
    bucket = storage_client.bucket(BUCKET_NAME)

    for filename, content in resources.get("outputs", {}).items():
        # Define the destination blob name based on slug
        blob_name = f"{slug}/{filename}"
        blob = bucket.blob(blob_name)

        # Upload the file to the bucket
        blob.upload_from_string(content, content_type="image/png")

        # Generate a signed URL for the blob
        signed_url = generate_signed_url(BUCKET_NAME, blob_name)
        image_urls[filename] = signed_url

    return image_urls

# Process the notebook and store in the database
def process_notebook(notebook_path, title=None):
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook file '{notebook_path}' does not exist.")
        sys.exit(1)

    # Extract title from the file name if not provided
    if not title:
        title = os.path.splitext(os.path.basename(notebook_path))[0].replace("_", " ").title()

    slug = create_slug(title)

    # Convert notebook to Markdown and extract resources
    markdown, resources = convert_notebook_to_markdown(notebook_path)

    # Upload images to Google Cloud Storage and get signed URLs
    image_urls = upload_images_to_gcs(resources, slug)

    # Update image links in Markdown to use signed URLs
    for filename, url in image_urls.items():
        markdown = markdown.replace(filename, url)

    # Store everything in the database
    try:
        db = SessionLocal()
        existing_blog = db.query(Blog).filter(Blog.slug == slug).first()
        if existing_blog:
            existing_blog.content = markdown
            existing_blog.images = image_urls  # Store signed URLs in the database
            print(f"Blog with slug '{slug}' updated.")
        else:
            new_blog = Blog(title=title, slug=slug, content=markdown, images=image_urls)
            db.add(new_blog)
            print(f"Blog with slug '{slug}' created.")
        db.commit()
    except Exception as e:
        print(f"Error storing blog in the database: {e}")
    finally:
        db.close()

# Main function
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_blogs.py <notebook_path> [<title>]")
        sys.exit(1)

    notebook_path = sys.argv[1]
    blog_title = sys.argv[2] if len(sys.argv) > 2 else None
    process_notebook(notebook_path, blog_title)
