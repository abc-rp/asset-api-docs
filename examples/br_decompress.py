#!/usr/bin/env python3

import argparse
import os

import brotli


def decompress_and_replace(input_file):
    """
    Decompresses a Brotli-compressed .pcd.br file and replaces it with the decompressed .pcd file.

    :param input_file: Path to the Brotli-compressed .pcd.br file
    """
    try:
        output_file = input_file[:-3]
        with open(input_file, "rb") as compressed_file:
            compressed_data = compressed_file.read()

        decompressed_data = brotli.decompress(compressed_data)

        with open(output_file, "wb") as decompressed_file:
            decompressed_file.write(decompressed_data)

        os.remove(input_file)

        print(f"Decompressed and replaced: {input_file} -> {output_file}")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")


def find_and_replace_pcd_br(directory):
    """
    Finds all Brotli-compressed .pcd.br files in the specified directory recursively,
    decompresses them, and replaces the original files with the decompressed .pcd files.

    :param directory: Path to the directory to search for .pcd.br files
    """
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    # Find all .pcd.br files
    pcd_br_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".pcd.br"):
                pcd_br_files.append(os.path.join(root, file))

    if not pcd_br_files:
        print(f"No .pcd.br files found in directory: {directory}")
        return

    print(f"Found {len(pcd_br_files)} .pcd.br files to decompress.")

    for file_path in pcd_br_files:
        decompress_and_replace(file_path)

    print("All decompression and replacement tasks completed.")


def main():
    parser = argparse.ArgumentParser(
        description="Decompress Brotli-compressed .pcd.br files."
    )
    parser.add_argument(
        "-d",
        "--directory",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "./downloads"),
        help="Directory to search for .pcd.br files (default: ./downloads)",
    )
    args = parser.parse_args()

    print(f"Using directory: {args.directory}")
    find_and_replace_pcd_br(args.directory)


if __name__ == "__main__":
    main()
