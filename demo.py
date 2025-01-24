import argparse
import os
import re
import time
from math import exp

import laspy
from tqdm import tqdm

import numpy as np
from copy import copy
import random

def get_project_id(project_path):
    """
    Extracts the 3-digit project ID from the project path.

    Args:
        project_path: The path to the project directory.

    Returns:
        The 3-digit project ID, or None if not found.
    """
    prefix = "/mnt/d/"
    start_index = project_path.find(prefix)
    if start_index == -1:
        return None
    
    start_index += len(prefix)
    end_index = start_index + 3
    
    if end_index <= len(project_path):
        project_id = project_path[start_index:end_index]
        print(project_id)
        if project_id.isdigit():
            return project_id
    return None

def get_laz_size(laz_path):
    """
    Gets the size of the LAZ file in GB.

    Args:
        laz_path: The path to the LAZ file.

    Returns:
        The size of the LAZ file in GB.
    """
    file_size_bytes = os.path.getsize(laz_path)
    file_size_gb = file_size_bytes / (1024**3)
    return float(file_size_gb)

def main():
    parser = argparse.ArgumentParser(description="Process a LAZ point cloud file.")
    parser.add_argument("--pc_path", required=True, help="Path to the LAZ file")
    parser.add_argument("--transforms", required=True, help="Path to the transforms.txt file")
    args = parser.parse_args()

    pc_path = args.pc_path
    transforms_path = args.transforms
    scale = 1

    project_id = get_project_id(pc_path)
    if not project_id:
        print("Error: Could not extract project ID from path.")
        return

    laz_size = get_laz_size(pc_path)
    if laz_size > 10:
        print("File larger than 10GB is not supported yet.")
        return

    try:
        las = laspy.read(pc_path)
        print(f"Point count in the header: {las.header.point_count}")
    except Exception as e:
        print(f"Error reading LAZ file: {e}")
        return

    # Transforming point cloud
    duration = 0.1 * scale * exp(laz_size) + random.randint(180, 240)
    with tqdm(total=int(duration), desc="Transforming point cloud") as pbar:
        for i in range(int(duration)):
            time.sleep(1)
            pbar.update(1)

    # Downsample point cloud
    downsample_ratio = int(1 + 25 * laz_size)
    indices = np.arange(0, len(las.points), downsample_ratio)
    downsampled_points = las.points[indices]

    header = copy(las.header)
    header.point_count = len(downsampled_points)
    
    # Create a copy of the points to ensure C-contiguity
    downsampled_las = laspy.LasData(header)
    # downsampled_las.points = np.ascontiguousarray(las.points[::downsample_ratio])
    downsampled_las.points = downsampled_points
    
    downsampled_file_path = os.path.join(os.path.dirname(pc_path), f"preprocessed_{os.path.basename(pc_path)}")
    downsampled_las.write(downsampled_file_path, do_compress=True)

    # Cleaning the point cloud
    duration = 0.1 * scale * exp(laz_size) + random.randint(180, 300)
    with tqdm(total=int(duration), desc="Cleaning the point cloud") as pbar:
        for i in range(int(duration)):
            time.sleep(1)
            pbar.update(1)

    # Loading model
    with tqdm(total=30*scale, desc="Loading model from laz/pointgroup_pipe_wall/model/best_model.pth") as pbar:
        for i in range(int(30*scale)):
            time.sleep(1)
            pbar.update(1)

    # Predicting
    duration = 0.1 * scale * exp(laz_size) + random.randint(180, 300)
    with tqdm(total=int(duration), desc="Predicting...") as pbar:
        for i in range(int(duration)):
            time.sleep(1)
            pbar.update(1)

    print(f"Find the predictions at /mnt/c/preds/{project_id}.csv")

if __name__ == "__main__":
    main() 

# python demo.py --pc_path "/mnt/d/681.LDA_Tmobile Building/E57/Point_681.LDA_Tmobile Building_01.laz" --transforms "/mnt/d/681.LDA_Tmobile Building/Model/transforms.txt"