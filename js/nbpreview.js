// Credit goes to https://github.com/jsvine/nbpreview/blob/master/js/nbpreview.js, which this file is based on
var $holder = document.querySelector("#main article .entry-content");

// Uses shadow DOM to avoid wordpress css from interfering with the notebook
let shadowroot = $holder.attachShadow({mode: 'open'});
let style = document.createElement('style');
style.innerHTML = ":host {all: initial; display: block;}";
shadowroot.appendChild(style)

let shadowbody = document.createElement("body");
shadowbody.setAttribute("id", "shadowbody");
shadowroot.appendChild(shadowbody);

function render_notebook(jsonstring) {

    json = JSON.parse(jsonstring);

    var notebook = nb.parse(json);
    while ($holder.hasChildNodes()) {
        $holder.removeChild($holder.lastChild);
    }
    shadowbody.appendChild(notebook.render());
    Prism.highlightAllUnder(shadowroot);

    stylesheets = [
        "wp-content/plugins/wordpress-ipynb-git-bridge/css/nbpreview.css",
        "wp-content/plugins/wordpress-ipynb-git-bridge/css/notebook.css",
        "wp-content/plugins/wordpress-ipynb-git-bridge/css/vendor/prism.css",
        "wp-content/plugins/wordpress-ipynb-git-bridge/css/vendor/katex.min.css",
        "wp-content/plugins/wordpress-ipynb-git-bridge/css/hidepseudo.css"
    ];

    for(const i in stylesheets) {
        linkElem = document.createElement('link');
        linkElem.setAttribute('rel', 'stylesheet');
        linkElem.setAttribute('href', stylesheets[i]);
        shadowroot.append(linkElem);
    }

}
