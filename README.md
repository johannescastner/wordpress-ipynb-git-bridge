# wordpress-ipynb-git-bridge
`wordpress-ipynb-git-bridge` is a WordPress plugin for importing .ipynb notebooks from github into a wordpress blog post.


### Example usage:
```[ipynb https://github.com/rasbt/matplotlib-gallery/blob/master/ipynb/heatmaps.ipynb]```  

Notebook rendering can be previewed at https://jsvine.github.io/nbpreview/

### Planned features&trade;
- Caching

### Known caveats
- Rendering interactive visualizations is limited, this needs more testing.
- Article heading must go into the wordpress document or you would have no title for navigation.

### Built using...
Inspired by https://github.com/gis-ops/wordpress-markdown-git  
Rendering done via https://github.com/jsvine/nbpreview  
Notebook styling taken from https://github.com/jsvine/notebookjs  
