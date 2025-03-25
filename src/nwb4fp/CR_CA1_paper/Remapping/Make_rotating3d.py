import math
import numpy as np
from vedo import *
import imageio
import os

def main():
    # Load the STL file
    mesh = Mesh(r"Q:\sachuriga\Documents/Presentations/Kavili_JC/3d/star_earring_.stl")  # Your STL file path

    # Customize the visualization
    mesh.color("blue")    # Set color
    mesh.alpha(1)         # Set transparency (0-1)
    mesh.linewidth(0)     # Set edge line width

    # Create a plotter (offscreen for rendering frames)
    plotter = Plotter(offscreen=True)  # Offscreen mode for frame generation
    plotter += mesh

    # Automatically adjust the camera to fit the mesh
    plotter.reset_camera()

    # Parameters for rotation
    frames = 80  # Number of frames for one full rotation
    radius = 200  # Distance from the mesh
    y = 50        # Height above the mesh
    images = []   # List to store frame images
    temp_files = []  # List to store temporary file paths

    # Generate frames by rotating the camera
    for i in range(frames):
        angle = i * (360 / frames)  # Full 360-degree rotation
        x = radius * math.cos(math.radians(angle))
        z = radius * math.sin(math.radians(angle))

        # Set camera position for this frame
        plotter.camera.SetPosition([x, y, z])
        plotter.camera.SetFocalPoint([0, 0, 0])  # Look at the center
        plotter.camera.SetViewUp([0, 1, 0])      # Z-axis is up

        # Render the scene and save the frame as a temporary file
        temp_filename = f"frame_{i:03d}.png"  # e.g., frame_000.png
        plotter.show(axes=0)  # Render without displaying
        plotter.screenshot(temp_filename)  # Save the frame to a file
        temp_files.append(temp_filename)

        # Read the saved image into memory
        img = imageio.imread(temp_filename)
        images.append(img)

    # Save frames as a .gif
    output_path = r"Q:/sachuriga/Documents/Presentations/Kavili_JC/star.gif"
    imageio.mimsave(output_path, images, duration=0.005,loop=0)  # Adjust duration for speed
    print(f"Rotating .gif saved as {output_path}")

    # Clean up temporary files
    for temp_file in temp_files:
        os.remove(temp_file)
    print("Temporary files cleaned up.")

if __name__ == "__main__":
    main()