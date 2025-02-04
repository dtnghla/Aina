import os
import math
import laspy
import numpy as np
from copy import copy

def downsample_point_cloud(las, ratio):
    # Create downsampled point array
    indices = np.arange(0, len(las.points), ratio)
    downsampled_points = las.points[indices]
    
    # Create new LAS header with updated point count
    header = copy(las.header)
    header.point_count = len(downsampled_points)
    
    # Build new LAS object and copy ALL dimensions
    downsampled_las = laspy.LasData(header)
    downsampled_las.points = downsampled_points
    
    # Explicitly copy over all non-standard fields (if any)
    if hasattr(las.header, 'extra_dim_params') and len(las.header.extra_dim_params) > 0:
        for extra_dim in las.header.extra_dim_params:
            downsampled_las[extra_dim.name] = las[extra_dim.name][indices]
    
    return downsampled_las