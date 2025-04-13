import numpy as np
import brainrender
import brainrender as br
from brainrender import Scene
from brainrender.actors import Points, Line
from probeinterface import get_probe
import shapely.geometry as sg  # For checking contour type

# Note: vedo is imported indirectly via brainrender
from vedo import settings
# Step 1: Create the brainrender scene with the Waxholm Space rat atlas
scene = Scene(atlas_name="whs_sd_rat_39um")  # 39μm resolution
settings.default_backend = 'vtk'

from brainrender import settings
settings.SHADER_STYLE = "cartoon"  # Or "metallic", "shiny", etc.

# scene.root.GetProperty().SetEdgeVisibility(0)
# scene.root.GetProperty().SetEdgeVisibility(0)

# Add brain regions for context (transparent)
scene.add_brain_region("CA1", alpha=0.4, color="blue")
scene.add_brain_region("PrL", alpha=0.4, color="green")  # Prelimbic area as mPFC proxy

# Get region centers dynamically (from mesh center of mass)
ca1_actor = [a for a in scene.actors if a.name == "CA1"][0]
ca1_center = ca1_actor.mesh.center_of_mass()

prl_actor = [a for a in scene.actors if a.name == "PrL"][0]
prl_center = prl_actor.mesh.center_of_mass()

print(f"CA1 center: {ca1_center} μm")
print(f"PrL center: {prl_center} μm")

# Step 2: Define planned trajectories (vertical insertion along z/DV axis)
# Assume dorsal surface approx at z min of region - implant depth; extend upward for pre-implant view
implant_depth_ca1 = 2000  # μm, typical for dorsal CA1 in rat
entry_ca1 = ca1_center.copy()
entry_ca1[2] -= implant_depth_ca1  # Decrease z to move dorsal (pre-implant position above surface)

implant_depth_prl = 4000  # μm, typical for mPFC/PrL in rat
entry_prl = prl_center.copy()
entry_prl[2] -= implant_depth_prl

# Add trajectory lines (shank approximations)
traj_ca1 = Line([entry_ca1, ca1_center])
traj_ca1.mesh.color("red").alpha(0.8).linewidth(3)
scene.add(traj_ca1)

traj_prl = Line([entry_prl, prl_center])
traj_prl.mesh.color("orange").alpha(0.8).linewidth(3)
scene.add(traj_prl)

# Step 3: Model the probe with probeinterface (e.g., Neuropixels)
probe_ca1 = get_probe('cambridgeneurotech', 'ASSY-156-P-1')  # Or custom

# Optionally get shank outline before transforming to 3D
if probe_ca1.probe_planar_contour is not None:
    contour = probe_ca1.probe_planar_contour
    if isinstance(contour, sg.Polygon):
        shank_points_2d_ca1 = np.array(contour.exterior.coords)  # (n, 2)
    else:
        shank_points_2d_ca1 = np.array(contour)  # Assume it's already array-like (n, 2)
    shank_points_3d_ca1 = np.zeros((shank_points_2d_ca1.shape[0], 3))
    shank_points_3d_ca1[:, 0] = shank_points_2d_ca1[:, 0]  # x
    shank_points_3d_ca1[:, 1] = shank_points_2d_ca1[:, 1]  # y (updated for 'xy' axes)

# Convert to 3D (assign back as it returns a new probe)
probe_ca1 = probe_ca1.to_3d(axes='xy')

# Align probe to CA1 trajectory (vertical, so no rotation needed; just move to entry)
probe_ca1.move(entry_ca1)  # Move to entry point

# Get contact positions after transformation
contact_positions_ca1 = probe_ca1.contact_positions

# Add contacts as points in brainrender
contacts_actor_ca1 = Points(contact_positions_ca1, radius=50, colors="yellow", alpha=0.7)  # Points for simplicity
scene.add(contacts_actor_ca1)

# Add shank outline if available
if 'shank_points_3d_ca1' in locals():
    shank_points_3d_ca1 += entry_ca1  # Apply the same move
    shank_line_ca1 = Line(shank_points_3d_ca1)
    shank_line_ca1.mesh.color("gray").alpha(1.0).linewidth(2)
    scene.add(shank_line_ca1)

# Duplicate for PrL probe
probe_prl = get_probe('cambridgeneurotech', 'ASSY-156-P-1')  # Or custom

if probe_prl.probe_planar_contour is not None:
    contour = probe_prl.probe_planar_contour
    if isinstance(contour, sg.Polygon):
        shank_points_2d_prl = np.array(contour.exterior.coords)
    else:
        shank_points_2d_prl = np.array(contour)
    shank_points_3d_prl = np.zeros((shank_points_2d_prl.shape[0], 3))
    shank_points_3d_prl[:, 0] = shank_points_2d_prl[:, 0]
    shank_points_3d_prl[:, 1] = shank_points_2d_prl[:, 1]  # y (updated for 'xy' axes)

probe_prl = probe_prl.to_3d(axes='xy')

probe_prl.move(entry_prl)

contact_positions_prl = probe_prl.contact_positions

contacts_actor_prl = Points(contact_positions_prl, radius=50, colors="cyan", alpha=0.7)  # Different color for distinction
scene.add(contacts_actor_prl)

if 'shank_points_3d_prl' in locals():
    shank_points_3d_prl += entry_prl
    shank_line_prl = Line(shank_points_3d_prl)
    shank_line_prl.mesh.color("lightgray").alpha(1.0).linewidth(2)
    scene.add(shank_line_prl)

# Step 4: Render the interactive 3D scene
scene.render(silhouette=False)