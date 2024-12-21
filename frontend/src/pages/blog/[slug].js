import { useRouter } from "next/router";
import { useState, useEffect } from "react";

export default function BlogPost() {
  const router = useRouter();
  const { slug } = router.query;

  const [blog, setBlog] = useState(null);

  useEffect(() => {
    if (slug) {
      fetch(`http://127.0.0.1:8000/blog/${slug}`) // Fetch blog by slug
        .then((res) => res.json())
        .then((data) => setBlog(data))
        .catch((err) => console.error("Failed to fetch blog:", err));
    }
  }, [slug]);

  if (!blog) {
    return <p>Loading...</p>;
  }

  return (
    <div>
      <h1>{blog.title}</h1>
      <div>
        <p>{blog.content}</p>
      </div>
    </div>
  );
}
