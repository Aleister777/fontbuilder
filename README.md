# fontbuilder
PNG image to font file converter

Usage:  
    ```python fontbuilder.py input.png -c "ABC" output.ttf```


    ```python fontbuilder.py input.png -c "ABC" output.ttf --name "MyFont"```

    
The PNG should have dark glyphs on a white background, arranged left to right
(and optionally in multiple rows) in the same order as the characters argument.

when building the png file with all the glyphs, sometimes you may find your computer
is out of ram when exporting and it shows up blank. lower the dpi
(for example 120dpi - 60dpi)

# fontconverter 
Font format converter

Converts a font file from one format to another based on file extension
(.ttf, .otf, .woff, .woff2).

Usage:  
    ```
    python fontconverter.py input.ttf output.woff2  
    python fontconverter.py input.woff2 output.ttf  
    ```

    
Whatever you do, definitely do NOT use this tool to rip Adobe paywalled fonts.
