<?php
/**
 * Plugin Name: Jupyter notebooks from github
 * Plugin URI: https://github.com/cdalinghaus/wordpress-ipynb-git-bridge
 * Description: Render Jupyter notebooks as blog posts
 * Version: 0.1.0
 * Author: cdalinghaus
 * Author URI: https://github.com/cdalinghaus
 */

// Cache for 30 days
$CACHE_DURATION_SECONDS = 60*60*24*30;

function base64_to_jpeg( $base64_string, $file_extension ) {

    $storage_location = plugin_dir_path( __FILE__ ) . "../../uploads/ipynb-media/";

    // Create folder if not exists
    if (!file_exists($storage_location)) {
        mkdir($storage_location, 0777, true);
    }

    $filename = md5($base64_string) . "." . $file_extension;

    $md5filename = md5($base64_string);

    $file = $storage_location . $filename;

    $ifp = fopen( $file, "wa+" ); 
    fwrite( $ifp, base64_decode( $base64_string) ); 
    fclose( $ifp ); 

    $img = imagecreatefrompng($file);
    imagepalettetotruecolor($img);
    imagealphablending($img, true);
    imagesavealpha($img, true);
    imagewebp($img, $dir . $storage_location . $md5filename  . ".webp", 100);
    imagedestroy($img);
    unlink($file);

    return get_site_url() . "/wp-content/uploads/ipynb-media/" . md5($base64_string) . ".webp";
}


function optimize_json($json) {
    $json = json_decode($json, true);
    
    $base64_cells = array("image/png", "png");

    foreach($json["cells"] as $cellindex => &$cell) {
        foreach($cell["outputs"] as $outputindex => &$output) {

            foreach($output["data"] as $dataindex => &$data) {
                if(in_array($dataindex, $base64_cells)) {

                    $data = base64_to_jpeg($data, "png");

                    $output["data"] = array("text/markdown" => "<img src='" . $data . "'>");



                }
            }
        }
    }
    //var_dump($json);
    return json_encode($json);
}



function inject_notebook($atts) {
    /**
     * Loads and injects the necessary .js files, then downloads the notebook and injects it into the page
     *
     * @param array $atts Attribute array from wp shortcode. Should contain notebook url as first element
     * 
     * @return NULL
     */ 
    
    load_js_files();
    wp_localize_script('mylib', 'WPURLS', array( 'siteurl' => get_option('siteurl') ));

    add_action( 'wp_footer', function() use( $atts ){
        $cache_storage_location = plugin_dir_path( __FILE__ ) . "../../uploads/ipynb-media/";
        $cachefilename = md5(reset($atts)) . ".ipynbcache";

        // Check if file is older than 30 days. If it is: Empty cache
        global $CACHE_DURATION_SECONDS;
        if (time()-filemtime($cache_storage_location . $cachefilename) > $CACHE_DURATION_SECONDS) {
            // file older than 2 hours
            unlink($cache_storage_location . $cachefilename);
        } else {
            echo "Render is " . (time()-filemtime($cache_storage_location . $cachefilename)) . " seconds old";
        }

        if(!file_exists($cache_storage_location . $cachefilename)) {

            // Convert github url to raw.githubusercontent.com url
            $raw_url = str_replace("github.com", "raw.githubusercontent.com", reset($atts));
            $raw_url = str_replace("/blob/", "/", $raw_url);

            // Download json from github
            $result = wp_remote_get($raw_url);
            echo "Render is -42 seconds old";

            // Optimize json
            $optimized_json = optimize_json($result["body"]);
            file_put_contents($cache_storage_location . $cachefilename, $optimized_json);
        } else {
            $optimized_json = file_get_contents($cache_storage_location . $cachefilename);
        }

        $jsontext = $result["body"];
        $jsontext = $optimized_json;
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
    wp_enqueue_script( "nbpreview", $plugin_url . "js/nbpreview.js", array(), 1.0, true);
    wp_localize_script('nbpreview', 'WPURLS', array( 'siteurl' => get_option('siteurl') ));
}

add_shortcode('ipynb', 'inject_notebook', 10000);


