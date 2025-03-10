import os
import sys
import numpy as np
import laspy
import trimesh
import glob
from pathlib import Path

project_id = '488'  # Hardcoded project ID

# Configuration for feet-based point cloud
FEET_PER_METER = 3.28084
TARGET_DISTANCE_METERS = 0.09  # 10cm threshold in meters
DISTANCE_THRESHOLD = TARGET_DISTANCE_METERS * FEET_PER_METER  # ~0.328 feet
SEARCH_RADIUS = DISTANCE_THRESHOLD  # Expanded search radius

# Read transform matrix
transform_path = f"/mnt/c/transforms/{project_id}.txt"
with open(transform_path, 'r') as f:
    lines = f.readlines()

transform_data = {}
for line in lines:
    key, *values = line.strip().split(',')
    transform_data[key] = [float(v) for v in values]

# Create 4x4 transformation matrix
transform_matrix = np.eye(4)
transform_matrix[0:3, 0] = transform_data['b0']
transform_matrix[0:3, 1] = transform_data['b1']
transform_matrix[0:3, 2] = transform_data['b2']
# transform_matrix[0:3, 3] = transform_data['centroid'] if np.allclose(transform_data['origin'], [0,0,0]) else transform_data['origin']
transform_matrix[0:3, 3] = transform_data['origin'] 

# Load and transform LAS data
laz_path = f"/mnt/d/laz/{project_id}_01.laz"
las = laspy.read(laz_path)
points = np.vstack((las.x, las.y, las.z)).T
transformed_points = points @ transform_matrix[:3, :3].T + transform_matrix[:3, 3]

# Initialize labeling arrays (background=0 by default)
semantic_labels = np.zeros(len(points), dtype=np.uint32)
instance_labels = np.zeros(len(points), dtype=np.uint32)
distances = np.full(len(points), np.inf)

# Process object files
obj_dir = f"/mnt/c/revit2obj/{project_id}/"

for obj_type in ['wall', 'pipe']:
    obj_files = glob.glob(f"{obj_dir}{obj_type}_*.obj")
    for obj_file in obj_files:
        mesh = trimesh.load_mesh(obj_file)
        instance_id = int(obj_file.split('_')[-1].split('.')[0])
        semantic_id = 1 if obj_type == 'wall' else 2

        # Convert mesh vertices to feet if OBJ is in meters (uncomment if needed)
        # mesh.vertices *= FEET_PER_METER

        # Get mesh bounds and expand by search radius
        try:
            mesh_vertices = mesh.vertices
        except Exception as e:
            print(f"Error getting mesh vertices for {obj_file}: {e}")
            continue

        try:
            min_bounds = np.min(mesh_vertices, axis=0)
            max_bounds = np.max(mesh_vertices, axis=0)
            search_min = min_bounds - SEARCH_RADIUS
            search_max = max_bounds + SEARCH_RADIUS
        except Exception as e:
            print(f"Error getting mesh bounds for {obj_file}: {e}")
            continue

        # Find points within expanded bounds
        in_x = (transformed_points[:, 0] >= search_min[0]) & (transformed_points[:, 0] <= search_max[0])
        in_y = (transformed_points[:, 1] >= search_min[1]) & (transformed_points[:, 1] <= search_max[1])
        in_z = (transformed_points[:, 2] >= search_min[2]) & (transformed_points[:, 2] <= search_max[2])
        candidate_mask = in_x & in_y & in_z
        candidate_indices = np.where(candidate_mask)[0]

        if len(candidate_indices) == 0:
            print(f"No candidates for {obj_file}")
            continue

        if len(candidate_indices) > 2000000:
            print(f"Skipping {obj_file} due to too many candidates")
            continue

        # Calculate exact distances to mesh surface
        pts = transformed_points[candidate_indices]
        closest_points, dist, _ = mesh.nearest.on_surface(pts)
        valid_mask = dist < DISTANCE_THRESHOLD

        # Update only if closer than previous matches
        update_mask = (dist[valid_mask] < distances[candidate_indices[valid_mask]])
        update_indices = candidate_indices[valid_mask][update_mask]

        # Apply labels to matched points
        semantic_labels[update_indices] = semantic_id
        instance_labels[update_indices] = instance_id
        distances[update_indices] = dist[valid_mask][update_mask]

        print(f"{obj_type.capitalize()} {instance_id}: {len(update_indices)} points labeled")

print("Creating new LAS file with labeled points")
header = laspy.LasHeader(point_format=7, version="1.4")
new_las = laspy.LasData(header)

# Copy XYZ from transformed_points
new_las.x = transformed_points[:, 0]
new_las.y = transformed_points[:, 1]
new_las.z = transformed_points[:, 2]

# Copy other standard dimensions if they exist
for dim in ['intensity', 'return_number', 'number_of_returns',
            'classification', 'user_data',
            'scan_angle', 'point_source_id', 'gps_time']:
    if hasattr(las, dim):
        setattr(new_las, dim, getattr(las, dim))

# Add extra dimensions for semantic and instance IDs
if 'semantic_id' not in new_las.point_format.extra_dimension_names:
    new_las.add_extra_dim(
        laspy.ExtraBytesParams(
            name='semantic_id',
            type=np.uint32,
            description='Semantic ID'
        )
    )
if 'instance_id' not in new_las.point_format.extra_dimension_names:
    new_las.add_extra_dim(
        laspy.ExtraBytesParams(
            name='instance_id',
            type=np.uint32,
            description='Instance ID'
        )
    )

new_las.semantic_id = semantic_labels
new_las.instance_id = instance_labels

# Save results
output_dir = Path("/mnt/d/labeled_laz")
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / f"{project_id}_labeled.laz"

new_las.write(output_path, do_compress=True)
print(f"Saved labeled point cloud to {output_path} (LAS version 1.4)")