import React from "react";
import ReactMarkdown from "react-markdown";
import fs from "fs";
import path from "path";

const DiversityAssignments = ({ content }) => {
  return (
    <div style={{ padding: "20px", fontFamily: "Arial, sans-serif" }}>
      <h1>Diversity Assignments</h1>
      <ReactMarkdown>{content}</ReactMarkdown>
    </div>
  );
};

export async function getStaticProps() {
  const filePath = path.join(process.cwd(), "src", "pages", "blog", "DiversityAssignments.md");
  const content = fs.readFileSync(filePath, "utf-8");

  return {
    props: { content },
  };
}

export default DiversityAssignments;
