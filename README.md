# Towards People

Towards People is building a **Collective Intelligence System** that integrates human creativity and ethical AI to enhance collaboration, decision-making, and performance. This repository contains the source code for the Towards People website, designed to showcase the organization's philosophy, services, and solutions.

## **Core Objectives**
- **Landing Page**: Showcase the organization’s mission, vision, and principles.
- **Blogging System**: Integrate Jupyter Notebook support for thought leadership and insights.
- **Backend Services**: Provide APIs for managing content and case studies.
- **AI System Vision**:
  - Integrate facts, perspectives, and actionable insights via:
    1. **BigQuery Data Warehouse**
    2. **LangGraph Agents**
    3. **FastAPI RESTful APIs**

---

## **Current Features**
1. **Landing Page**:
   - AMP-compliant React/Next.js frontend for fast, SEO-friendly pages.
   - Responsive design for desktop and mobile.
2. **Python FastAPI Backend**:
   - RESTful API for blog posts and case studies.
   - Integration with Jupyter for notebook-to-blog workflows.
3. **Blogging System**:
   - Convert Jupyter Notebooks to Markdown for easy publishing.
   - Automatic slug generation for SEO optimization.

---

## Current State of the Project

### Frontend
- The **Next.js frontend** is live on [http://localhost:3000](http://localhost:3000).
- Key frontend features:
  - **Landing page (`index.js`)**: Provides the main entry point for users.
  - **Blog listing page (`blog.js`)**: Displays a list of all blogs retrieved from the backend.
- Challenges:
  - Initially, images were stored in the database as base64-encoded strings. Large image sizes caused performance issues when rendering them on the frontend.
  - Future updates will pull images directly from the Google Cloud Storage bucket.

### Backend
- The **FastAPI backend** is live on [http://localhost:8000](http://localhost:8000).
- Key backend features:
  - **Blog endpoints**:
    - `/blog`: Fetches all blogs.
    - `/blog/{slug}`: Fetches a single blog by its unique slug.
  - **Blogging system**:
    - Converts Jupyter Notebooks into Markdown format for blogs.
    - Initially stored both content and images (base64-encoded) in the database. Now uses Google Cloud Storage for image hosting.
- Progress:
  - The `add_blogs.py` script automates the process of uploading Jupyter Notebooks to the blog system. This includes:
    - Converting notebooks to Markdown.
    - Extracting and uploading images to a Google Cloud Storage bucket.
    - Storing blog content in the database with references to image URLs.

### Google Cloud Storage Integration
- A bucket (`towards_people_blog_images`) was created to host blog images.
- Images are organized in folders named after their corresponding blog slugs.
- Challenge:
  - The bucket cannot currently be made publicly accessible due to **IAM policies** enforced at the organization level.
  - Temporary attempts to use signed URLs have also failed due to restricted permissions (e.g., `iam.disableServiceAccountKeyCreation`).

### Blocking Issues
1. **Public Access to Images**:
   - The bucket should ideally be publicly readable to allow the frontend to directly access images.
   - Organization-level restrictions prevent enabling public access or generating signed URLs.
2. **Next Steps on IAM Policies**:
   - Collaborate with administrators to:
     - Enable public access for the bucket.
     - Or allow service account key creation for signed URLs.

---

## Goals for the `add_blogs.py` Functionality

### Current State
- The `add_blogs.py` script currently processes Jupyter Notebooks by:
  - Converting them to Markdown.
  - Extracting images and uploading them to the Google Cloud Storage bucket.
  - Storing blog content and image URLs in the database.

### Future Plans
- Transition `add_blogs.py` into a **Google Cloud Function** to streamline blog entry creation.
- Features to include:
  - **Admin Page**:
    - Accessible via Google account login.
    - Allow authorized users to manage blog uploads.
  - **Upload Options**:
    - Upload a Jupyter Notebook directly from a local computer.
    - Provide a URL pointing to an online Jupyter Notebook (e.g., Google Colab).
    - Support other document types like Microsoft Word and Google Docs.
  - **Automated Processing**:
    - Convert uploaded documents into Markdown.
    - Extract and upload images to the Cloud Storage bucket.
    - Store all content in the database.

### Long-Term Goals
- Develop a user-friendly **admin interface** for bloggers.
- Support a variety of document formats for blog creation.
- Enhance integration with cloud services to ensure scalability and efficiency.

---

## Next Steps
1. Resolve the Google Cloud Storage bucket access issue:
   - Enable public access or resolve signed URL permissions.
2. Test the backend integration once access is resolved.
3. Refactor `add_blogs.py` into a scalable cloud function.
4. Build the admin interface to enable easier blog management.
5. Expand supported document formats for blog creation.

---

## Features to Implement

We have outlined the following features and enhancements to be implemented in future iterations of the project:

1. **Admin Section**  
   - Develop an `/admin` portal for uploading Jupyter Notebooks.  
   - Add secure authentication for admin users.

2. **Blog Automation**  
   - Enable automatic blog creation from uploaded `.ipynb` files.  
   - Store blogs in a database or as static files.

3. **Google Cloud Deployment**  
   - Deploy frontend and backend to Google Cloud Run.  
   - Set up Cloud Storage and a Load Balancer for scalability and high availability.  

4. **Dynamic Blog Features**  
   - Add categories and tagging functionality for blog posts.  
   - Enable user comments for each blog post.

5. **User Surveillance and Analytics**  
   - Track user activity on the website.  
   - Stream user behavior data to a BigQuery warehouse.  
   - Build a visualization dashboard in Apache Superset.

6. **Commenting and Likes**  
   - Implement a commenting system for blog posts with threaded replies.  
   - Add a like button with a counter for each blog post.  

7. **Social Integration**  
   - Allow users to log in via LinkedIn or Facebook to comment and like posts.  
   - Add social sharing buttons for LinkedIn and Facebook on blog posts.

---

### Notes:
- The **Apache Superset** dashboard and **BigQuery data warehousing** are part of a separate repository/project.
- Details for each feature will be worked out as development progresses.


## **File Structure**

```plaintext
towards-people/
├── backend/                   # Python backend (FastAPI)
│   ├── app/                   # Application code
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI entry point
│   │   ├── routes/            # API route files
│   │   │   ├── __init__.py
│   │   │   └── blog.py        # Blog-related routes (to be implemented)
│   │   └── services/          # Logic and service layer
│   │       ├── __init__.py
│   │       └── jupyter_integration.py
│   ├── tests/                 # Unit and integration tests
│   │   ├── __init__.py
│   │   └── test_blog.py       # Blog-related test cases
│   ├── requirements.txt       # Python dependencies
│   └── Dockerfile             # Containerization setup for Google Cloud Run
│
├── frontend/                  # React/Next.js frontend
│   ├── public/                # Static assets (images, favicon, etc.)
│   ├── src/                   # Application source code
│   │   ├── components/        # Reusable React components
│   │   ├── pages/             # Next.js pages
│   │   │   ├── index.js       # Homepage
│   │   │   └── blog.js        # Blog page
│   │   ├── styles/            # CSS/SCSS files
│   │   └── utils/             # Helper utilities
│   ├── package.json           # Node.js dependencies
│   └── next.config.js         # Next.js configuration
│
├── .gitignore                 # Git ignore file
├── README.md                  # Project documentation
└── docker-compose.yml         # Optional: Compose for managing services
```
## **How to Contribute**
1. **Clone the Repository**:
```bash
  git clone https://github.com/your-repo/towards-people.git
  cd towards-people
```

2. **Run Frontend**:

```bash
  cd frontend
  npm install
  npm run dev
```

3. **Run Backend**:
```bash
  cd backend
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  uvicorn app.main:app --reload
```
- Push and submit a pull request:
```bash
   git add .
   git commit -m "Describe your changes here"
   git push origin feature-branch-name
```
   - Open a pull request on GitHub, clearly describing the changes and linking to any related issues.

---

## **License**

This project is licensed under the [MIT License](LICENSE).

---


