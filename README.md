# Towards People

This repository contains the source code for the Towards People website, featuring:
- A **Python FastAPI backend** for APIs and services.
- A **React/Next.js frontend** for the user interface.
- Integration with Jupyter Notebooks for blogging.

## Features
- AMP-compliant frontend for fast, SEO-friendly pages.
- Python backend with RESTful APIs.
- Easy-to-use blogging system with Jupyter Notebook support.

---

## File Structure


towards-people/
├── backend/                   # Python backend (FastAPI)
│   ├── app/                   # Application code
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI entry point
│   │   ├── routes/            # API route files
│   │   │   ├── __init__.py
│   │   │   └── blog.py        # Blog-related routes
│   │   └── services/          # Logic and service layer
│   │       ├── __init__.py
│   │       └── jupyter_integration.py
│   ├── tests/                 # Unit and integration tests
│   │   ├── __init__.py
│   │   └── test_blog.py
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
