import laspy
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree  # More efficient than Open3D's KDTree

def process_laz(laz_f, transform_f):
    """Process LAZ file in a memory efficient way"""
    print('Processing point cloud data')

    # First pass: just get the transform matrix
    print('Reading transformation matrix...')
    with open(transform_f, "r") as f:
        b0 = [float(x) for x in f.readline().strip().split(",")[1:]]
        b1 = [float(x) for x in f.readline().strip().split(",")[1:]]
        b2 = [float(x) for x in f.readline().strip().split(",")[1:]]
        origin = [float(x) for x in f.readline().strip().split(",")[1:]]
        scale = float(f.readline().strip().split(",")[1])
        t = np.array([b0, b1, b2, origin]) / scale

    # Open LAZ file to get header info
    las_file = laspy.open(laz_f, laz_backend=laspy.LazBackend.Laszip)
    point_count = las_file.header.point_count
    
    # Check if the file has color information
    has_colors = all(hasattr(las_file, attr) for attr in ['red', 'green', 'blue'])
    if not has_colors:
        print("No color information found in LAZ file. Using grey color.")
    
    # Process in chunks to reduce memory usage
    chunk_size = 1000000  # Reduce chunk size to avoid buffer issues
    
    # Pre-allocate the final arrays
    points = np.zeros((point_count, 3), dtype=np.float32)
    colors = np.zeros((point_count, 3), dtype=np.float32)
    
    print(f'Processing {point_count} points...')
    
    # Use chunk_iterator for memory efficient reading
    points_processed = 0
    try:
        # Create a new iterator for each attempt
        chunk_iterator = las_file.chunk_iterator(chunk_size)
        while points_processed < point_count:
            try:
                chunk = next(chunk_iterator)
            except StopIteration:
                print(f"Reached end of file after {points_processed} points")
                break
            
            # Get actual number of points in this chunk
            chunk_length = len(chunk.X)
            if chunk_length == 0:
                break
                
            # Process coordinates
            X = chunk.X * las_file.header.scales[0] + las_file.header.offsets[0]
            Y = chunk.Y * las_file.header.scales[1] + las_file.header.offsets[1]
            Z = chunk.Z * las_file.header.scales[2] + las_file.header.offsets[2]
            
            # Stack and transform points
            chunk_points = np.column_stack([X, Y, Z, np.ones_like(X)])
            chunk_points = chunk_points @ t
            chunk_points = chunk_points[:, :3] / 0.3048  # Convert to feet
            
            # Process colors
            if has_colors:
                # Convert from uint16 to float32 [0,1] range
                chunk_colors = np.vstack([
                    chunk.red.astype(np.float32),
                    chunk.green.astype(np.float32),
                    chunk.blue.astype(np.float32)
                ]).T
                chunk_colors /= 65535.0  # uint16 max value
            else:
                # Use grey color (0.5, 0.5, 0.5)
                chunk_colors = np.full((len(chunk_points), 3), 0.5, dtype=np.float32)
            
            # Store in pre-allocated arrays
            end_idx = points_processed + chunk_length
            points[points_processed:end_idx] = chunk_points
            colors[points_processed:end_idx] = chunk_colors
            
            points_processed += chunk_length
            print(f'Processed {points_processed}/{point_count} points')
            
    except Exception as e:
        print(f"Warning: Encountered error while processing points: {str(e)}")
        print(f"Successfully processed {points_processed} points out of {point_count}")

    las_file.close()
    
    return points, colors

def find_points_in_radius(tree, center, radius):
    """Find points within radius using KD-tree"""
    return tree.query_ball_point(center, radius)

def process_geometry_batch(points, candidate_idx, bbox, endpoint1, endpoint2, radius=None, height=None, is_pipe=True):
    """Process a batch of points for either pipe or wall geometry"""
    # Convert candidate_idx to numpy array if it's not already
    candidate_idx = np.asarray(candidate_idx)
    
    # Filter points within bounding box (vectorized)
    bbox_min, bbox_max = bbox
    in_bbox = np.all((points >= bbox_min) & (points <= bbox_max), axis=1)
    
    if not np.any(in_bbox):
        return np.array([], dtype=np.int64), np.array([], dtype=bool)
    
    candidate_points = points[in_bbox]
    valid_idx = candidate_idx[in_bbox]
    
    if is_pipe:
        # Pipe distance calculation (vectorized)
        line_vector = endpoint2 - endpoint1
        point_vectors = candidate_points - endpoint1
        projections = np.dot(point_vectors, line_vector) / np.dot(line_vector, line_vector)
        closest_points = endpoint1 + projections[:, np.newaxis] * line_vector
        distances = np.linalg.norm(candidate_points - closest_points, axis=1)
        mask = (distances <= radius) & (projections >= 0) & (projections <= 1)
    else:
        # Wall distance calculation (vectorized)
        normal_vector = np.cross(endpoint2 - endpoint1, [0, 0, height])
        normal_norm = np.linalg.norm(normal_vector)
        d = -np.dot(normal_vector, endpoint1)
        distances = np.abs(np.dot(candidate_points, normal_vector) + d) / normal_norm
        mask = distances < 1e-2
    
    return valid_idx, mask

# Create bounding box for pipe (unchanged)
def create_pipe_bbox(endpoint1, endpoint2, radius):
    min_coords = np.minimum(endpoint1, endpoint2) - radius
    max_coords = np.maximum(endpoint1, endpoint2) + radius
    return np.stack([min_coords, max_coords])

# Create bounding box for wall (vectorized)
def create_wall_bbox(endpoint1, endpoint2, height):
    min_coords = np.minimum(endpoint1, endpoint2)
    max_coords = np.maximum(endpoint1, endpoint2)
    min_coords[2] = min(endpoint1[2], endpoint2[2])
    max_coords[2] = min_coords[2] + height
    return np.stack([min_coords, max_coords])

#"D:\687.LDA_Boardroom\E57\Point_687.LDA_Boardroom_01.laz"
project_path = "/mnt/d/687.LDA_Boardroom/"
project_name = "687"
path_laz = project_path + "E57/Point_687.LDA_Boardroom_01.laz"
path_transforms = project_path + "Model/transforms.txt"
path_csv = f"/mnt/c/walls_and_pipes_geometry_info/{project_name}.csv"
path_output = project_path + "E57/Point_687.LDA_Boardroom_01_labeled.laz"
#687
# Process point cloud
points, colors = process_laz(path_laz, path_transforms)

# Build KD-tree for efficient spatial queries
print("Building KD-tree...")
tree = cKDTree(points)

# Load the CSV file
csv_file = pd.read_csv(path_csv)

# Initialize arrays to store semantic and instance IDs (using uint8/uint16 for memory efficiency)
semantic_ids = np.zeros(len(points), dtype=np.uint8)
instance_ids = np.zeros(len(points), dtype=np.uint16)

# Process geometries in batches
batch_size = 100  # Process geometries in batches
num_batches = (len(csv_file) + batch_size - 1) // batch_size

for batch_idx in range(num_batches):
    start_idx = batch_idx * batch_size
    end_idx = min((batch_idx + 1) * batch_size, len(csv_file))
    batch_df = csv_file.iloc[start_idx:end_idx]
    
    print(f"Processing batch {batch_idx + 1}/{num_batches}")
    
    for _, row in batch_df.iterrows():
        type_ = row['type']
        endpoint1 = np.array([row['endpoint1_x'], row['endpoint1_y'], row['endpoint1_z']])
        endpoint2 = np.array([row['endpoint2_x'], row['endpoint2_y'], row['endpoint2_z']])
        
        # Calculate center and search radius
        center = (endpoint1 + endpoint2) / 2
        length = np.linalg.norm(endpoint2 - endpoint1)
        
        if type_ == 'pipe':
            radius = row['radius']
            bbox = create_pipe_bbox(endpoint1, endpoint2, radius)
            search_radius = length/2 + radius
        else:  # wall
            height = row['height']
            bbox = create_wall_bbox(endpoint1, endpoint2, height)
            center[2] += height/2
            search_radius = np.sqrt((length/2)**2 + (height/2)**2)
        
        # KD-tree search
        idx = find_points_in_radius(tree, center, search_radius)
        
        if len(idx) > 0:
            # Process candidate points
            valid_idx, mask = process_geometry_batch(
                points[idx], 
                idx,
                bbox, 
                endpoint1, 
                endpoint2,
                radius if type_ == 'pipe' else None,
                height if type_ == 'wall' else None,
                is_pipe=(type_ == 'pipe')
            )
            
            if len(valid_idx) > 0:
                semantic_ids[valid_idx[mask]] = 1 if type_ == 'pipe' else 2
                instance_ids[valid_idx[mask]] = start_idx + _ + 1

print("Creating output LAZ file...")
# Create a LasData object with the point data
las = laspy.create(point_format=3, file_version="1.4")
las.x = points[:, 0]
las.y = points[:, 1]
las.z = points[:, 2]
las.red = (colors[:, 0] * 255).astype(np.uint16)
las.green = (colors[:, 1] * 255).astype(np.uint16)
las.blue = (colors[:, 2] * 255).astype(np.uint16)

# Add extra dimensions
las.add_extra_dim(laspy.ExtraBytesParams(name="semantic_id", type=np.uint8))
las.add_extra_dim(laspy.ExtraBytesParams(name="instance_id", type=np.uint16))

# Set the semantic and instance IDs
las.semantic_id = semantic_ids
las.instance_id = instance_ids

# Write to file
las.write(path_output)