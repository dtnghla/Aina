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

# ============================================
# 1. Logging Setup
# ============================================
project_id = '509'  # Hardcoded project ID (modify as needed)

logs_dir = Path('logs')
logs_dir.mkdir(exist_ok=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = logs_dir / f"labeling_{project_id}_{timestamp}.log"

logger = logging.getLogger('point_cloud_labeler')
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"Starting point cloud labeling for project {project_id}")

# ============================================
# 2. Configuration
# ============================================
FEET_PER_METER = 3.28084
TARGET_DISTANCE_METERS = 0.0001  # threshold in meters
DISTANCE_THRESHOLD = TARGET_DISTANCE_METERS * FEET_PER_METER  # in feet

logger.info(f"Distance threshold (feet): {DISTANCE_THRESHOLD}")

# ============================================
# 3. Load Transformation Matrix
# ============================================
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

# ============================================
# 4. Load and Transform the Point Cloud
# ============================================
laz_path = f"/mnt/d/laz/{project_id}_01.laz"
try:
    logger.info(f"Loading point cloud from {laz_path}")
    las = laspy.read(laz_path)
    points = np.vstack((las.x, las.y, las.z)).T

    # Apply transformation
    transformed_points = points @ transform_matrix[:3, :3].T + transform_matrix[:3, 3]

    logger.info(f"Successfully loaded and transformed {len(points)} points")
except Exception as e:
    logger.error(f"Failed to load or transform point cloud: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

# Initialize labeling arrays (background = 0)
semantic_labels = np.zeros(len(points), dtype=np.uint32)
instance_labels = np.zeros(len(points), dtype=np.uint32)
distances = np.full(len(points), np.inf)

logger.info("Initialized labeling arrays")

# ============================================
# 5. Combine All OBJ Files into a Single Mesh
# ============================================
obj_dir = f"/mnt/c/revit2obj/{project_id}/"
all_obj_files = glob.glob(f"{obj_dir}*.obj")
logger.info(f"Found {len(all_obj_files)} OBJ files in {obj_dir}")

all_vertices = []
all_faces = []
face_to_semantic = []
face_to_instance = []

current_vertex_count = 0

for obj_file in all_obj_files:
    try:
        logger.info(f"Loading OBJ: {obj_file}")
        mesh = trimesh.load_mesh(obj_file, process=False)
        
        # Infer semantic_id and instance_id from filename convention "type_instance.obj"
        # e.g. "wall_123.obj" => semantic=1, instance=123
        base_name = os.path.splitext(os.path.basename(obj_file))[0]
        # Example split: "wall_123" -> ["wall", "123"]
        obj_type, instance_str = base_name.split('_', 1)
        instance_id = int(instance_str)  # e.g. 123
        semantic_id = 1 if obj_type.lower() == 'wall' else 2  # example logic

        # Extract geometry
        v = mesh.vertices
        f = mesh.faces

        # Append data
        all_vertices.append(v)
        # Adjust face indices by current offset
        all_faces.append(f + current_vertex_count)

        # Keep track of face -> semantic, instance
        face_count = len(f)
        face_to_semantic.append(np.full(face_count, semantic_id, dtype=np.uint32))
        face_to_instance.append(np.full(face_count, instance_id, dtype=np.uint32))

        # Increment vertex offset
        current_vertex_count += len(v)

    except Exception as e:
        logger.error(f"Error processing {obj_file}: {str(e)}")
        logger.error(traceback.format_exc())

if len(all_vertices) == 0:
    logger.warning("No valid OBJ geometry was loaded. Exiting...")
    sys.exit(0)

# Create combined arrays
all_vertices = np.vstack(all_vertices)
all_faces = np.vstack(all_faces)
face_to_semantic = np.concatenate(face_to_semantic)
face_to_instance = np.concatenate(face_to_instance)

# Build a single mesh
big_mesh = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)
logger.info(f"Combined mesh has {len(big_mesh.vertices)} vertices and {len(big_mesh.faces)} faces")

# ============================================
# 6. Nearest Surface Queries for All Points
# ============================================
logger.info("Building proximity structure for combined mesh")
proximity = trimesh.proximity.ProximityQuery(big_mesh)

logger.info("Starting distance queries")
total_labeled_points = 0
batch_size = 100000  # Tweak for memory vs. speed tradeoff

start_time_global = time.time()

for start_idx in range(0, len(transformed_points), batch_size):
    end_idx = min(start_idx + batch_size, len(transformed_points))
    batch_pts = transformed_points[start_idx:end_idx]

    # on_surface returns (distances, face_indices)
    closest, dists, face_ids = proximity.on_surface(batch_pts)

    # Filter by threshold
    valid_mask = (dists < DISTANCE_THRESHOLD)
    valid_local_indices = np.where(valid_mask)[0]
    if len(valid_local_indices) == 0:
        continue

    global_indices = start_idx + valid_local_indices  # map local -> global

    # Among those under threshold, check if closer than any existing match
    better_mask = dists[valid_mask] < distances[global_indices]
    final_global_indices = global_indices[better_mask]

    # Update distances
    distances[final_global_indices] = dists[valid_mask][better_mask]

    # Map face indices to semantic/instance
    chosen_face_ids = face_ids[valid_mask][better_mask]
    semantic_labels[final_global_indices] = face_to_semantic[chosen_face_ids]
    instance_labels[final_global_indices] = face_to_instance[chosen_face_ids]

    total_labeled_points += len(final_global_indices)

elapsed_global = time.time() - start_time_global
logger.info(f"Distance queries completed in {elapsed_global:.2f} seconds.")
logger.info(f"Total labeled points within threshold: {total_labeled_points}")

# ============================================
# 7. Create a New LAS File with Extra Dimensions
# ============================================
try:
    logger.info("Creating new LAS file with labeled points")
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
    logger.info(f"Saved labeled point cloud to {output_path} (LAS version 1.4)")

except Exception as e:
    logger.error(f"Error creating or saving labeled point cloud: {str(e)}")
    logger.error(traceback.format_exc())
    sys.exit(1)

logger.info("Point cloud labeling process completed")
