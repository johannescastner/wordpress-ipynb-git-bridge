 # wordpress-ipynb-git-bridge
`wordpress-ipynb-git-bridge` is a WordPress plugin for importing and syncronizing .ipynb notebooks from github into a wordpress blog post. This allows you to version control your posts via Github.

### Example usage:
```[ipynb https://github.com/rasbt/matplotlib-gallery/blob/master/ipynb/heatmaps.ipynb]```  

Notebook rendering can be previewed at https://jsvine.github.io/nbpreview/

### Use a raw cell as the first cell in your notebook to set post metadata:
```
%META
title=This is the post title
excerpt=This is the post excerpt
tags=tutorial, python, fibonacci
categories=math,hobby
```
Every key is optional. Metadata will overwrite fields that were manually set in WordPress. Tags and categories will be created if they don't exist.

### Features
- Pull and synchronize ipynb notebooks from github into your WordPress blog
- Base64 embedded images are converted to static webp
- File system caching
- Parse and autoload post title, excerpt, tags, categories and slug from notebook metadata cell

### Planned features&trade;
- [ ] Somehow grab publish_date and update_date from github
- [ ] Auto sync the repository to the blog

### Known bugs (In order of importance)
- Base64 images embedded via markdown are not turned into static media

### Known caveats
- Rendering interactive visualizations is limited, this needs more testing.
- Cache has to be reset manually by clearing wp-content/uploads/ipynb-media/*
- raw.githubusercontent.com caches files for 5 minutes, making manual clearing the cache after an update inconvenient
- When images are removed from a notebook that has been optimized previously, corresponding .webp files can become orphant. Fix this by periodically clearing ipynb-media directory.

### Built using...
Inspired by https://github.com/gis-ops/wordpress-markdown-git  
Rendering done via https://github.com/jsvine/nbpreview  
Notebook styling taken from https://github.com/jsvine/notebookjs  
