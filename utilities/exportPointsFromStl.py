#!/usr/bin/env python3
import glob
import pyvista as pv
import os
import numpy as np
import argparse

def export_points_from_stl(input_directory, output_directory):
    """
    Export points from STL files in the input directory to CSV files in the output directory.

    Parameters:
    - input_directory: str, path to the directory containing STL files.
    - output_directory: str, path to the directory where CSV files will be saved.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get all STL files in the input directory
    stl_files = glob.glob(os.path.join(input_directory, '*.stl'))

    for stl_file in stl_files:
        # Read the STL file
        mesh = pv.read(stl_file)

        # Extract points
        points = mesh.points

        # Create a DataFrame and save to CSV
        output_file = os.path.join(output_directory, os.path.basename(stl_file).replace('.stl', '.txt'))
        file_content = []
        width = 20
        header = ['#'+'x'.rjust(width-1), 'y'.rjust(width), 'z'.rjust(width)]
        header = "\t".join(header)
        file_content.append(header)
        for point in points:
            line = [str(point[0]).rjust(width), str(point[1]).rjust(width), str(point[2]).rjust(width)]
            line = "\t".join(line)
            file_content.append(line)

        with open(output_file, 'w') as f:
            f.write("\n".join(file_content))

        print(f"Exported points from {stl_file} to {output_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export points from STL files to TXT format.")
    parser.add_argument('--input_directory', type=str, required=True, help="Directory containing STL files.")
    parser.add_argument('--output_directory', type=str, required=True, help="Directory to save the output CSV files.")
    args = parser.parse_args()

    export_points_from_stl(args.input_directory, args.output_directory)
