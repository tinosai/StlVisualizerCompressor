#!/usr/bin/env python3
import glob
import pyvista as pv
import os
import numpy as np
import argparse

def export_points_and_normals_from_stl(input_directory, output_directory):
    """
    Exports points from STL files to formatted text files.
    This function processes all STL files in a given input directory and exports
    their vertex points and normals to formatted text files in the specified output directory.
    Each point is written in a tab-separated format with right-justified columns.
    Parameters:
        input_directory (str): Path to the directory containing STL files to process
        output_directory (str): Path to the directory where output files will be saved
    Returns:
        None
    Output Format:
        - Creates one text file per STL file
        - File header: #x    y    z    nx    ny    nz
        - Each line contains the x, y, z coordinates and the normal vector (nx, ny, nz)
        - Values are right-justified in 20-character width columns
        - Values are tab-separated
    Example:
        >>> export_points_and_normals_from_stl('/path/to/stls', '/path/to/output')
        Exported points from /path/to/stls/model.stl to /path/to/output/model.txt
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

        # Extract Normals
        normals_array = np.zeros((points.shape[0],3))
        counts = np.zeros(points.shape[0], dtype=int)

        triangles_structure = mesh.faces.reshape(-1,4)[:,1:].ravel()
        normals = np.repeat(mesh.face_normals, 3, axis=0)
        np.add.at(normals_array, triangles_structure, normals)
        np.add.at(counts, triangles_structure, np.ones(triangles_structure.shape[0],))
        normals_array /= counts[:, np.newaxis]
        points = np.hstack((points, normals_array))

        # Create a DataFrame and save to CSV
        output_file = os.path.join(output_directory, os.path.basename(stl_file).replace('.stl', '.txt'))
        file_content = []
        width = 20
        header = ['#'+'x'.rjust(width-1), 'y'.rjust(width), 'z'.rjust(width), 'nx'.rjust(width), 'ny'.rjust(width), 'nz'.rjust(width)]
        header = "\t".join(header)
        file_content.append(header)
        for point in points:
            line = [str(point[0]).rjust(width), str(point[1]).rjust(width), str(point[2]).rjust(width), str(point[3]).rjust(width), str(point[4]).rjust(width), str(point[5]).rjust(width)]
            line = "\t".join(line)
            file_content.append(line)

        with open(output_file, 'w') as f:
            f.write("\n".join(file_content))

        print(f"Exported points and normals from {stl_file} to {output_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export points and normals from STL files to TXT format.")
    parser.add_argument('--input_directory', type=str, required=True, help="Directory containing STL files.")
    parser.add_argument('--output_directory', type=str, required=True, help="Directory to save the output TXT files.")
    args = parser.parse_args()

    export_points_and_normals_from_stl(args.input_directory, args.output_directory)
