import laspy
import pandas as pd
import numpy as np
import trimesh
import os
from scipy.spatial import cKDTree

def process_laz(laz_f, transform_f):
    """Process LAZ file with corrected unit scaling (meters→feet)"""
    print('Processing point cloud data')
    
    # Read transform file
    with open(transform_f, "r") as f:
        b0 = [float(x) for x in f.readline().strip().split(",")[1:]]
        b1 = [float(x) for x in f.readline().strip().split(",")[1:]]
        b2 = [float(x) for x in f.readline().strip().split(",")[1:]]
        origin = [float(x) for x in f.readline().strip().split(",")[1:]]
        scale = float(f.readline().strip().split(",")[1])

        # Convert scale to feet (1 meter = 3.28084 feet)
        scale_factor = scale * 3.28084
        
        # Scale basis vectors and origin
        b0 = [x * scale_factor for x in b0]
        b1 = [x * scale_factor for x in b1]
        b2 = [x * scale_factor for x in b2]
        origin = [x * scale_factor for x in origin]

        # Construct transformation matrix
        t = np.array([
            [b0[0], b1[0], b2[0], origin[0]],
            [b0[1], b1[1], b2[1], origin[1]],
            [b0[2], b1[2], b2[2], origin[2]],
            [0, 0, 0, 1]
        ])
    
    # Open LAZ file
    las_file = laspy.open(laz_f)
    point_count = las_file.header.point_count
    
    # Check color info
    has_colors = all(hasattr(las_file, attr) for attr in ['red', 'green', 'blue'])
    if not has_colors:
        print("No color information found. Using grey color.")
    
    # Process in chunks
    chunk_size = 1000000
    points = np.zeros((point_count, 3), dtype=np.float32)
    colors = np.zeros((point_count, 3), dtype=np.float32)
    
    print(f'Processing {point_count} points...')
    
    points_processed = 0
    try:
        chunk_iterator = las_file.chunk_iterator(chunk_size)
        while points_processed < point_count:
            chunk = next(chunk_iterator)
            chunk_length = len(chunk.X)
            
            # Process coordinates
            X = chunk.X * las_file.header.scales[0] + las_file.header.offsets[0]
            Y = chunk.Y * las_file.header.scales[1] + las_file.header.offsets[1]
            Z = chunk.Z * las_file.header.scales[2] + las_file.header.offsets[2]
            
            chunk_points = np.column_stack([X, Y, Z])
            
            # Apply transform (now in Revit's feet-based coordinates)
            chunk_points = np.concatenate([chunk_points, np.ones([len(chunk_points), 1])], axis=1)
            chunk_points = chunk_points @ t
            chunk_points = chunk_points[:, :3]
            
            # Process colors
            if has_colors:
                chunk_colors = np.vstack([chunk.red, chunk.green, chunk.blue]).T / 65535.0
            else:
                chunk_colors = np.full((len(chunk_points), 3), 0.5, dtype=np.float32)
            
            # Store data
            end_idx = points_processed + chunk_length
            points[points_processed:end_idx] = chunk_points
            colors[points_processed:end_idx] = chunk_colors
            
            points_processed += chunk_length
            print(f'Processed {points_processed}/{point_count} points')
            
    except Exception as e:
        print(f"Error: {str(e)}")
    
    las_file.close()
    return points, colors

def get_wall_type(family_name):
    """Determine wall type based on family name"""
    if 'Curtain Wall' in family_name:
        return 'curtain_wall'
    elif 'LTS_' in family_name:
        thickness = int(family_name.split('mm')[0].split('LTS_')[1])
        if thickness <= 110:
            return 'partition_wall'
        else:
            return 'structural_wall'
    return 'other_wall'

def get_semantic_id(category, family_name):
    """Get semantic ID based on category and family name"""
    if category == 'Pipes':
        return 1
    elif category == 'Walls':
        wall_type = get_wall_type(family_name)
        if wall_type == 'curtain_wall':
            return 2
        elif wall_type == 'partition_wall':
            return 3
        elif wall_type == 'structural_wall':
            return 4
        return 5  # other walls
    return 0  # background

def label_points_with_mesh(points, mesh, tree, search_radius=0.1):
    """Label points that are close to or inside a mesh"""
    # Use KD-tree to find points near mesh vertices for efficiency
    nearby_points_idx = tree.query_ball_point(mesh.vertices, search_radius)
    nearby_points_idx = np.unique([idx for sublist in nearby_points_idx for idx in sublist])
    
    if len(nearby_points_idx) == 0:
        return np.array([], dtype=np.int64)
    
    # For nearby points, do more precise distance check
    candidate_points = points[nearby_points_idx]
    closest_points, distances, _ = trimesh.proximity.closest_point(mesh, candidate_points)
    
    # Points are labeled if they are within threshold distance of the mesh
    mask = distances < search_radius
    return nearby_points_idx[mask]

def main(points, colors, obj_dir, metadata_path, output_path):
    """Adjusted for foot-based search radii"""
    tree = cKDTree(points)
    metadata_df = pd.read_csv(metadata_path)
    
    semantic_ids = np.zeros(len(points), dtype=np.uint8)
    instance_ids = np.zeros(len(points), dtype=np.uint16)
    
    category_counts = {'Pipe':0, 'Curtain Wall':0, 'Partition Wall':0, 'Structural Wall':0, 'Other Wall':0}
    
    for idx, row in metadata_df.iterrows():
        element_id = row['ElementId']
        category = row['Category']
        family_name = row['FamilyName'] if pd.notna(row['FamilyName']) else row['ElementName']
        semantic_id = get_semantic_id(category, family_name)
        
        obj_path = os.path.join(obj_dir, f"{category}_{element_id}.obj")
        if not os.path.exists(obj_path):
            continue
        
        mesh = trimesh.load_mesh(obj_path)
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        
        # Set search radius in feet (converted to meters for calculation)
        search_radius = 0.3048  # 1 foot = 0.3048 meters
        if category == 'Pipes':
            search_radius = 0.1524  # 0.5 feet
        elif 'Curtain Wall' in family_name:
            search_radius = 0.09144  # 0.3 feet
        
        labeled_points = label_points_with_mesh(points, mesh, tree, search_radius)
        if len(labeled_points) > 0:
            semantic_ids[labeled_points] = semantic_id
            instance_ids[labeled_points] = idx + 1
            # Update category counts...
    
    # Save output LAZ file (unchanged)
    las = laspy.create(point_format=3, file_version="1.4")
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]
    las.red = (colors[:, 0] * 65535).astype(np.uint16)
    las.green = (colors[:, 1] * 65535).astype(np.uint16)
    las.blue = (colors[:, 2] * 65535).astype(np.uint16)
    las.add_extra_dim(laspy.ExtraBytesParams(name="semantic_id", type=np.uint8))
    las.add_extra_dim(laspy.ExtraBytesParams(name="instance_id", type=np.uint16))
    las.semantic_id = semantic_ids
    las.instance_id = instance_ids
    las.write(output_path)
    print(f"Saved labeled data to {output_path}")

if __name__ == "__main__":
    # Example usage
    project_path = "/mnt/c/revit2obj/511/"
    laz_path = "/mnt/d/511.LOC_WAITARA/E57/Point_511.LOC_WAITARA_01.laz"
    transform_path = os.path.join("/mnt/c/transforms/", "511.txt")
    obj_dir = project_path
    metadata_path = os.path.join(project_path, "metadata.csv")
    output_path = "/mnt/d/511.LOC_WAITARA/E57/Point_511.LOC_WAITARA_01_labeled.laz"
    
    # Update main call to include transform file
    points, colors = process_laz(laz_path, transform_path)
    main(points, colors, obj_dir, metadata_path, output_path)
