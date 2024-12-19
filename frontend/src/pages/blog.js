import { useState, useEffect } from "react";
import Link from "next/link";

export default function Blog() {
  const [blogs, setBlogs] = useState([]);

  useEffect(() => {
    // Fetch blogs dynamically from the backend API
    fetch("http://localhost:8000/blog")
      .then((res) => res.json())
      .then((data) => setBlogs(data))
      .catch((error) => console.error("Error fetching blogs:", error));
  }, []);

  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h1>Blog</h1>
      <ul>
        {/* Render blogs fetched dynamically */}
        {blogs.map((blog) => (
          <li key={blog.id}>
            <h2>{blog.title}</h2>
            <p>{blog.content}</p>
          </li>
        ))}

        {/* Add static Markdown-based blog posts */}
        <li>
          <h2>
            <Link href="/blog/DiversityAssignments">
              Diversity Assignments
            </Link>
          </h2>
          <p>A closer look at diversity in team assignments and its impact.</p>
        </li>
      </ul>
    </div>
  );
}
