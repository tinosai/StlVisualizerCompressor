#!/usr/bin/env python3

import pyvista as pv
import numpy as np
import os
import argparse
import glob


def generate_dummy_data_from_text_file(file_path, output_dir) -> None:
    """
    Generate dummy data from a text file and save it as a text file.

    Parameters:
    - file_path: Path to the input text file.
    - output_dir: Directory to save the output file.
    """
    # Read the data from the text file
    data = np.loadtxt(file_path)

    # Create random dummy data
    dummy_data = np.random.rand(data.shape[0]).tolist()

    # Create file content
    file_content = "# Data\n"
    file_content += "\n".join(map(str, dummy_data))

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Save the dummy data to a new text file
    output_file_path = os.path.join(output_dir, os.path.basename(file_path))
    with open(output_file_path, 'w') as f:
        f.write(file_content)
    print(f"Dummy data generated and saved to {output_file_path}")


def generate_dummy_data_from_all_text_files(input_dir, output_dir) -> None:
    """
    Generate dummy data from all text files in a directory.

    Parameters:
    - input_dir: Directory containing the input text files.
    - output_dir: Directory to save the output files.
    """
    # Get all text files in the input directory
    text_files = glob.glob(os.path.join(input_dir, '*.txt'))
    
    # Generate dummy data for each text file
    for file_name in text_files:
        generate_dummy_data_from_text_file(file_name, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy data from text files.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing the input text files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output text files.")
    args = parser.parse_args()

    generate_dummy_data_from_all_text_files(args.input_dir, args.output_dir)
    