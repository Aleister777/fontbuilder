#!/usr/bin/env python3
"""
Font format converter

Converts a font file from one format to another based on file extension
(.ttf, .otf, .woff, .woff2).

Usage:
    python fontconverter.py input.ttf output.woff2
    python fontconverter.py input.woff2 output.ttf
"""

import os
import sys
import argparse

from fontTools.ttLib import TTFont

FLAVORS = {
    ".woff":  "woff",
    ".woff2": "woff2",
    ".ttf":   None,
    ".otf":   None,
}


def convert(input_path, output_path):
    in_ext = os.path.splitext(input_path)[1].lower()
    out_ext = os.path.splitext(output_path)[1].lower()

    if out_ext not in FLAVORS:
        raise ValueError(f"Unsupported output format: {out_ext}")

    print(f"Loading {input_path} ...")
    font = TTFont(input_path)

    font.flavor = FLAVORS[out_ext]

    print(f"Saving {output_path} ...")
    font.save(output_path)
    print(f"Done: {in_ext} -> {out_ext}")


def main():
    parser = argparse.ArgumentParser(description="Convert a font file between formats.")
    parser.add_argument("input", help="Input font path (.ttf, .otf, .woff, .woff2)")
    parser.add_argument("output", help="Output font path (.ttf, .otf, .woff, .woff2)")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Cannot open font: {args.input}")
        sys.exit(1)

    convert(args.input, args.output)


if __name__ == "__main__":
    main()
