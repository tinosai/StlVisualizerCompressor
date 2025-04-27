#!/usr/bin/env python3
import pyvista as pv
import glob
import datetime
import numpy as np
import functools
import colorsys

def printText(string : str) -> None:
    """
    Prints a string with a timestamp prefix.

    This function takes a string input and prints it to console with the current
    datetime as a prefix in the format 'YYYY-MM-DD HH:MM:SS.mmmmmm\tstring'.

    Args:
        string (str): The text to be printed.

    Returns:
        None: This function doesn't return anything, it prints to console.

    Example:
        >>> printText("Hello World")
        2023-07-21 10:30:45.123456    Hello World
    """
    print(f"{datetime.datetime.now()}\t{string}")

def nice_random_color() -> tuple:
    """
    Generates a random RGB color tuple with visually pleasing properties.

    The function uses HSV (Hue, Saturation, Value) color space to generate colors:
    - Hue is completely random
    - Saturation is restricted to medium values (0.4-0.7) 
    - Value/Brightness is restricted to bright values (0.7-0.9)

    Returns:
        tuple: RGB color values as (red, green, blue), each in range [0.0, 1.0]
    """
    h = np.random.rand()            
    s = np.random.uniform(0.4, 0.7)  
    v = np.random.uniform(0.7, 0.9)  
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (r, g, b)

def main() -> None:
    """
    Interactive STL file visualization tool using PyVista.
    This function provides an interactive 3D visualization environment for STL files with the following features:
    - Loads all STL files from the current directory
    - Displays meshes with randomly generated colors
    - Provides checkbox interface to:
        - Toggle edge visibility for all meshes
        - Toggle visibility of individual parts
    - Enables mesh picking functionality with click interactions to:
        - Toggle bold text for selected part name
        - Toggle edge visibility for selected part
    The visualization includes:
    - A main viewport showing all 3D meshes
    - A control panel with:
        - Global edge visibility toggle
        - Individual part visibility toggles
        - Part names listed with corresponding color indicators
    Returns:
        None
    Note: 
        Requires PyVista library for 3D visualization
        Expects STL files to be present in the current working directory
    """
    # 0. set the seed for reproducibility
    np.random.seed(42)

    # 1. get all the stls
    stls = sorted(glob.glob("*.stl"))
    
    # 2. read the stls
    meshes = []
    for stl in stls:
        printText(f"START:\tReading...{stl}")
        mesh = pv.read(stl)
        meshes.append(mesh)
        printText(f"END:\tReading...{stl}")

    # 3. plotting
    printText("START:\tPlotting")
    p = pv.Plotter()
    #p.background_color = [97,161,198] # The color of the background. Default is white.
    window_width, window_height = p.window_size
    
    # 3.1 generate colors for the meshes using a random color generator
    colors = [nice_random_color() for i in range(len(stls))]
    actors = []
    for i, mesh in enumerate(meshes):
        actors.append(p.add_mesh(mesh, color=colors[i]))
    
    # 3.2 toggle edge visibility.
    def toggle_edge_vis(flag):
        if flag:
            for actor in actors:
                actor.GetProperty().EdgeVisibilityOn()
        else:
            for actor in actors:
                actor.GetProperty().EdgeVisibilityOff()

    checkboxes_actors = {}
    text_actors = {}
    _ = p.add_checkbox_button_widget(toggle_edge_vis, value=False, position=(10,10), color_on="red", color_off="gray", size=40)
    p.add_text("Visualize Triangles", position=(60, 10), font_size=20)

    # 3.3 add boxes for the visualization of parts
    def toggle_visibility(actor, flag):
        actor.SetVisibility(flag)
    
    gap = 100
    for i, actor in enumerate(actors[::-1]):
        checkboxes_actors[actor] = p.add_checkbox_button_widget(functools.partial(toggle_visibility, actor), value = True, position = (10, 10+((i+1)*40)+gap), size=40, color_on=colors[::-1][i], color_off="gray")
        text_actors[actor] = p.add_text(stls[::-1][i].replace(".stl",""), position=(60, 10+(i+1)*40+gap), font_size=20)
    p.add_text("Parts", position=(10, 10+(i+2)*40+gap), font_size=20)
    p.add_text("Visualization", position=(10, 50), font_size=20)
    

    def mesh_click_callback(mesh) -> None: 
        """
        Callback function triggered when a mesh in the 3D visualization is clicked.
        This function toggles the visibility of edges and text boldness for the selected mesh actor.
        When a mesh is clicked, if it has an associated text label, the function alternates between:
        - Bold text and visible edges
        - Normal text and hidden edges
        Parameters:
            mesh: The mesh object that was clicked (unused in current implementation)
        Returns:
            None
        Note:
            The function relies on global/class variables:
            - p: Plotter instance containing the picked_actor
            - text_actors: Dictionary mapping mesh actors to their text labels
        """

        actor = p.picked_actor
        if actor:
            label = text_actors.get(actor)
            if label:
                bold_condition = label.GetTextProperty().GetBold()
                if bold_condition:
                    label.GetTextProperty().BoldOff()
                    actor.GetProperty().EdgeVisibilityOff()
                else:
                    label.GetTextProperty().BoldOn() 
                    actor.GetProperty().EdgeVisibilityOn() 


        p.render()

    # 3.4 add the click callback
    p.enable_mesh_picking(mesh_click_callback, show=False)#, color='black')
    
    # 3.5 add the mesh picking
    p.show()
    printText("END:\tPlotting")
    

if __name__ == "__main__":
    main()