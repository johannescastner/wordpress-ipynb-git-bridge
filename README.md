 # wordpress-ipynb-git-bridge
`wordpress-ipynb-git-bridge` is a WordPress plugin for importing and syncronizing .ipynb notebooks from github into a wordpress blog post. This allows you to version control your posts via Github.

### Example usage:
```[ipynb https://github.com/rasbt/matplotlib-gallery/blob/master/ipynb/heatmaps.ipynb]```  

Notebook rendering can be previewed at https://jsvine.github.io/nbpreview/

### Features
- Pull and synchronize ipynb notebooks from github into your WordPress blog
- Base64 embedded images are converted to static webp
- File system caching

### Planned features&trade;
- [ ] Parse and autoload post title, excerpt, tags, categories and slug from notebook metadata cell
- [ ] Somehow grab publish_date and update_date from github
- [ ] Auto sync the repository to the blog

### Known bugs (In order of importance)
- Base64 images embedded via markdown are not turned into static media

### Known caveats
- Rendering interactive visualizations is limited, this needs more testing.
- Article heading must go into the wordpress document or you would have no title for navigation.
- Excerpt must be set manually

### Built using...
Inspired by https://github.com/gis-ops/wordpress-markdown-git  
Rendering done via https://github.com/jsvine/nbpreview  
Notebook styling taken from https://github.com/jsvine/notebookjs  
