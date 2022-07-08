 # wordpress-ipynb-git-bridge
`wordpress-ipynb-git-bridge` is a WordPress plugin for importing .ipynb notebooks from github into a wordpress blog post.


### Example usage:
```[ipynb https://github.com/rasbt/matplotlib-gallery/blob/master/ipynb/heatmaps.ipynb]```  

Notebook rendering can be previewed at https://jsvine.github.io/nbpreview/

### Planned features&trade;
- [x] Caching
- [ ] Parse and autoload post title, excerpt, tags, categories and slug from notebook metadata cell
- [ ] Somehow grab publish_date and update_date from github
- [ ] Auto sync the repository to the blog to make the relevant database state (aside from comments) reproducible
- [x] Extract b64 images and store them statically

### Known caveats
- <s>On a throttled connection initial page load is painfully slow because document size expodes</s>
- Rendering interactive visualizations is limited, this needs more testing.
- Article heading must go into the wordpress document or you would have no title for navigation.
- Excerpt must be set manually

### Built using...
Inspired by https://github.com/gis-ops/wordpress-markdown-git  
Rendering done via https://github.com/jsvine/nbpreview  
Notebook styling taken from https://github.com/jsvine/notebookjs  
