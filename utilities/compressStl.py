#!/usr/bin/env python3
import pyvista as pv
import glob
import os

def compress_stl(input_file, output_file, binary=True, max_points = 10000) -> None:
    """
    Compress an STL file using PyVista.

    Parameters:
    - input_file: Path to the input STL file.
    - output_file: Path to save the compressed STL file.
    """
    # Load the STL file
    mesh = pv.read(input_file)
    orig_number_of_points = mesh.number_of_points
    # Compress the mesh
    if orig_number_of_points > max_points:
        # Find the compression ratio
        compression_ratio = 1 - max_points / mesh.number_of_points
        print(compression_ratio)
        mesh = mesh.decimate_pro(compression_ratio)
        print(f"Compressed {input_file} from {orig_number_of_points} to {mesh.number_of_points} points.")

    # Save the compressed STL file
    mesh.save(output_file, binary=binary)


def main(max_pts=1000):
    all_sts = glob.glob("*.stl")
    output_folder = "compressed"
    os.makedirs(output_folder, exist_ok=True)
    for stl in all_sts:
        compressed_file = os.path.join(output_folder, os.path.basename(stl))
        compress_stl(stl, compressed_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compress STL files.")
    parser.add_argument("--max_pts", type=int, default=1000, help="Maximum number of points in the compressed STL file.")
    args = parser.parse_args()
    main(args.max_pts)
