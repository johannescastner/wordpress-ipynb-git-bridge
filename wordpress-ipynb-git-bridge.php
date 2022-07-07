<?php
/**
 * Plugin Name: Jupyter notebooks from github
 * Plugin URI: https://github.com/cdalinghaus/wordpress-ipynb-git-bridge
 * Description: Render Jupyter notebooks as blog posts
 * Version: 0.1.0
 * Author: cdalinghaus
 * Author URI: https://github.com/cdalinghaus
 */


function inject_notebook($atts) {
    /**
     * Loads and injects the necessary .js files, then downloads the notebook and injects it into the page
     *
     * @param array $atts Attribute array from wp shortcode. Should contain notebook url as first element
     * 
     * @return NULL
     */ 
    
    load_js_files();

    add_action( 'wp_footer', function() use( $atts ){
        // Convert github url to raw.githubusercontent.com url
        $raw_url = str_replace("github.com", "raw.githubusercontent.com", reset($atts));
        $raw_url = str_replace("/blob/", "/", $raw_url);

        // Download json from github
        $result = wp_remote_get($raw_url);

        $jsontext = $result["body"];
        $jsontext = str_replace("\n", "", $jsontext);
        $jsontext = json_encode($jsontext);
        ?>
            <script>
            (function() {
            render_notebook(<?php echo $jsontext; ?>);
            })();
            </script> 
        <?php
    }, 10000);
}


function load_js_files() {
    /**
     * Injects the js files into the document
     *
     * @return NULL
     */ 
    
    $plugin_url = plugin_dir_url( __FILE__ );

    $js_files = array(
        "js/vendor/es5-shim.min.js",
        "js/vendor/marked.min.js",
        "js/vendor/purify.min.js",
        "js/vendor/ansi_up.min.js",
        "js/vendor/prism.min.js",
        "js/vendor/katex.min.js",
        "js/vendor/katex-auto-render.min.js",
        "js/vendor/notebook.min.js"
    );

    foreach($js_files as $js_file) {
        wp_enqueue_script( uniqid(), $plugin_url . $js_file, array(), 1.0, true);
    }
    // nbpreview.js has to be last
    wp_enqueue_script( uniqid(), $plugin_url . "js/nbpreview.js", array(), 1.0, true);
}

add_shortcode('ipynb', 'inject_notebook', 10000);
