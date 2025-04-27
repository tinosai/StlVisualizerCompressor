#!/usr/bin/env python3
import pyvista
import os
import glob
import argparse

def transform_stl(input_file: str, output_file: str, binary : bool, scale : float = 1.0, translate : list = [0.0, 0.0, 0.0]) -> None:
    """
    Transforms an STL file by scaling and translating it, and saves the result to a new STL file.

    Args:
        input_file (str): Path to the input STL file.
        output_file (str): Path to the output STL file.
        binary (bool): If True, save the output as a binary STL file. If False, save as ASCII.
        scale (float): Scale factor for the transformation. Default is 1.0 (no scaling).
        translate (list): Translation vector for the transformation. Default is [0.0, 0.0, 0.0] (no translation).
    """
    # Read the STL file
    mesh = pyvista.read(input_file)

    # Apply translation
    mesh.points += translate

    # Apply scaling
    mesh.points *= scale

    # Save the transformed mesh
    mesh.save(output_file, binary=binary)



def translate_all_stls(input_dir: str, output_dir: str, binary : bool, scale : float = 1.0, translate : list = [0.0, 0.0, 0.0]) -> None:
    """
    Transforms all STL files in a directory by scaling and translating them, and saves the results to a new directory.

    Args:
        input_dir (str): Path to the input directory containing STL files.
        output_dir (str): Path to the output directory for transformed STL files.
        binary (bool): If True, save the output as a binary STL file. If False, save as ASCII.
        scale (float): Scale factor for the transformation. Default is 1.0 (no scaling).
        translate (list): Translation vector for the transformation. Default is [0.0, 0.0, 0.0] (no translation).
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all STL files in the input directory
    stl_files = glob.glob(os.path.join(input_dir, '*.stl'))

    # Transform each STL file
    for stl_file in stl_files:
        # Construct the output file path
        output_file = os.path.join(output_dir, os.path.basename(stl_file))
        
        # Transform and save the STL file
        transform_stl(stl_file, output_file, binary=binary, scale=scale, translate=translate)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transform STL files by scaling and translating them. Mind that the translation is applied first and then the scaling.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing STL files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory for transformed STL files.")
    parser.add_argument("--binary", action="store_true", help="If set, save the output as a binary STL file.")
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for the transformation. Default is 1.0 (no scaling).")
    parser.add_argument("--translate", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="Translation vector for the transformation. Default is 0.0 0.0 0.0 (no translation).")
    args = parser.parse_args()
    # Call the function to transform all STL files in the input directory
    translate_all_stls(args.input_dir, args.output_dir, args.binary, args.scale, args.translate)
    # Print a message indicating completion
    print(f"Transformed STL files from {args.input_dir} to {args.output_dir}.")