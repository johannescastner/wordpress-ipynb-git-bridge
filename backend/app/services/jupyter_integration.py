def convert_notebook_to_html(notebook_path: str) -> str:
    """
    Converts a Jupyter Notebook (.ipynb) file to HTML.

    Args:
        notebook_path (str): Path to the notebook file.

    Returns:
        str: HTML representation of the notebook.
    """
    import nbformat
    from nbconvert import HTMLExporter

    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    html_exporter = HTMLExporter()
    body, _ = html_exporter.from_notebook_node(notebook)
    return body

