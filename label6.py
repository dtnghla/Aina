import os
import sys
import time
import numpy as np
import laspy
import trimesh
import glob
from pathlib import Path
import logging
from datetime import datetime
import traceback

# Set up logging
project_id = '488'  # Hardcoded project ID

# Create logs directory if it doesn't exist
logs_dir = Path('logs')
logs_dir.mkdir(exist_ok=True)

# Configure logging with timestamp in filename
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = logs_dir / f"labeling_{project_id}_{timestamp}.log"

# Set up logger
logger = logging.getLogger('point_cloud_labeler')
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"Starting point cloud labeling for project {project_id}")

# Configuration for feet-based point cloud
FEET_PER_METER = 3.28084
TARGET_DISTANCE_METERS = 0.0001 #threshold in meters
DISTANCE_THRESHOLD = TARGET_DISTANCE_METERS * FEET_PER_METER  # feet
SEARCH_RADIUS = DISTANCE_THRESHOLD  # Expanded search radius

logger.info(f"Distance threshold: {DISTANCE_THRESHOLD} feet, Search radius: {SEARCH_RADIUS} feet")

# Read transform matrix
transform_path = f"/mnt/c/transforms/{project_id}.txt"
try:
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
    transform_matrix[0:3, 3] = transform_data['origin'] 
    
    logger.info(f"Successfully loaded transformation matrix from {transform_path}")
except Exception as e:
    logger.error(f"Failed to load transformation matrix: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# Load and transform LAS data
laz_path = f"/mnt/d/laz/{project_id}_01.laz"
try:
    logger.info(f"Loading point cloud from {laz_path}")
    las = laspy.read(laz_path)
    points = np.vstack((las.x, las.y, las.z)).T
    transformed_points = points @ transform_matrix[:3, :3].T + transform_matrix[:3, 3]
    logger.info(f"Successfully loaded and transformed {len(points)} points")
except Exception as e:
    logger.error(f"Failed to load or transform point cloud: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# Initialize labeling arrays (background=0 by default)
semantic_labels = np.zeros(len(points), dtype=np.uint32)
instance_labels = np.zeros(len(points), dtype=np.uint32)
distances = np.full(len(points), np.inf)
logger.info("Initialized labeling arrays")

# Process object files
obj_dir = f"/mnt/c/revit2obj/{project_id}/"
logger.info(f"Processing object files from {obj_dir}")

total_labeled_points = 0

for obj_type in ['wall', 'pipe']:
    obj_files = glob.glob(f"{obj_dir}{obj_type}_*.obj")
    logger.info(f"Found {len(obj_files)} {obj_type} files to process")
    
    for obj_file in obj_files:
        try:
            logger.info(f"Processing {obj_file}")
            mesh = trimesh.load_mesh(obj_file)
            instance_id = int(obj_file.split('_')[-1].split('.')[0])
            semantic_id = 1 if obj_type == 'wall' else 2
            
            # Get mesh bounds and expand by search radius
            try:
                mesh_vertices = mesh.vertices
            except Exception as e:
                logger.error(f"Error getting mesh vertices for {obj_file}: {e}")
                continue
                
            min_bounds = np.min(mesh_vertices, axis=0)
            max_bounds = np.max(mesh_vertices, axis=0)
            search_min = min_bounds - SEARCH_RADIUS
            search_max = max_bounds + SEARCH_RADIUS
            
            # Find points within expanded bounds
            in_x = (transformed_points[:, 0] >= search_min[0]) & (transformed_points[:, 0] <= search_max[0])
            in_y = (transformed_points[:, 1] >= search_min[1]) & (transformed_points[:, 1] <= search_max[1])
            in_z = (transformed_points[:, 2] >= search_min[2]) & (transformed_points[:, 2] <= search_max[2])
            candidate_mask = in_x & in_y & in_z
            candidate_indices = np.where(candidate_mask)[0]
            logger.info(f"Found {len(candidate_indices)} candidate points for {obj_file}")
            
            if len(candidate_indices) == 0:
                logger.warning(f"No candidates for {obj_file}")
                continue
            
            # Calculate exact distances to mesh surface
            pts = transformed_points[candidate_indices]
            logger.info(f"Processing {len(pts)} points for {obj_file}")
            start_time = time.time()
            
            # Process in batches to reduce memory usage and improve performance
            batch_size = 10000
            closest_points = []
            dist = []
            
            for i in range(0, len(pts), batch_size):
                batch_pts = pts[i:i+batch_size]
                batch_closest, batch_dist, _ = mesh.nearest.on_surface(batch_pts)
                closest_points.append(batch_closest)
                dist.append(batch_dist)
                logger.debug(f"  Processed batch {i//batch_size + 1}/{(len(pts)-1)//batch_size + 1} ({len(batch_pts)} points)")
                
            if len(closest_points) > 1:
                closest_points = np.vstack(closest_points)
                dist = np.concatenate(dist)
            else:
                closest_points = closest_points[0] if closest_points else np.array([])
                dist = dist[0] if dist else np.array([])
                
            elapsed_time = time.time() - start_time
            logger.info(f"Distance calculation completed in {elapsed_time:.2f} seconds")
            valid_mask = dist < DISTANCE_THRESHOLD
            
            # Update only if closer than previous matches
            update_mask = (dist[valid_mask] < distances[candidate_indices[valid_mask]])
            update_indices = candidate_indices[valid_mask][update_mask]
            
            # Apply labels to matched points
            semantic_labels[update_indices] = semantic_id
            instance_labels[update_indices] = instance_id
            distances[update_indices] = dist[valid_mask][update_mask]
            
            total_labeled_points += len(update_indices)
            logger.info(f"{obj_type.capitalize()} {instance_id}: {len(update_indices)} points labeled")
            
        except Exception as e:
            logger.error(f"Error processing {obj_file}: {str(e)}")
            logger.error(traceback.format_exc())

logger.info(f"Labeling complete. Total labeled points: {total_labeled_points}")

# Create a new LAS file with version 1.4 to properly support extra dimensions
try:
    logger.info("Creating new LAS file with labeled points")
    header = laspy.LasHeader(point_format=7, version="1.4")
    new_las = laspy.LasData(header)
    
    # Copy point data from original LAS
    new_las.x = transformed_points[:, 0]
    new_las.y = transformed_points[:, 1]
    new_las.z = transformed_points[:, 2]
    
    # Copy other standard dimensions if they exist in the original file
    for dim in ['intensity', 'return_number', 'number_of_returns', 
                'classification', 'user_data',
                'scan_angle', 'point_source_id', 'gps_time']:
        if hasattr(las, dim):
            setattr(new_las, dim, getattr(las, dim))
    
    # Add dimensions for semantic and instance labels
    for name, dtype in [('semantic_id', np.uint32), ('instance_id', np.uint32)]:
        if name not in new_las.point_format.extra_dimension_names:
            new_las.add_extra_dim(laspy.ExtraBytesParams(
                name=name,
                type=dtype,
                description=f"{name.replace('_', ' ').title()}"
            ))
    
    new_las.semantic_id = semantic_labels
    new_las.instance_id = instance_labels
    
    # Save results
    output_dir = Path("/mnt/d/labeled_laz")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{project_id}_labeled.laz"
    new_las.write(output_path, do_compress=True)
    logger.info(f"Saved labeled point cloud to {output_path} (LAS version 1.4)")
except Exception as e:
    logger.error(f"Error creating or saving labeled point cloud: {str(e)}")
    logger.error(traceback.format_exc())

logger.info("Point cloud labeling process completed")