# Point Cloud Labeling Workflow

### 1. Initialization & Configuration
```python
# Libraries: os, sys, time, numpy, laspy, trimesh, glob, pathlib
SET project_id = '488'
CALCULATE conversion constants (feet/meters)
SET distance thresholds (DISTANCE_THRESHOLD, SEARCH_RADIUS)
```

### 2. Transform Matrix Loading
```python
# Libraries: numpy
READ transform.txt file
PARSE matrix components (b0, b1, b2, origin)
CONSTRUCT 4x4 transformation matrix using numpy
```

### 3. Point Cloud Processing
```python
# Libraries: laspy, numpy
LOAD LAZ/LAS file using laspy
EXTRACT x,y,z coordinates
APPLY transformation matrix to points using numpy matrix operations
```

### 4. Mesh Processing Loop
```python
# Libraries: trimesh, glob, numpy
FOR EACH object type (wall, pipe):
    LIST OBJ files using glob
    FOR EACH OBJ file:
        LOAD mesh with trimesh
        CALCULATE expanded search bounds using numpy
        FIND candidate points within bounds using numpy array operations
```

### 5. Distance Calculation
```python
# Libraries: trimesh, numpy
PROCESS in batches (memory optimization):
    CALCULATE point-to-mesh distances using trimesh.nearest.on_surface()
    FILTER points within threshold using numpy boolean indexing
    UPDATE labels only for closer matches using numpy array masking
```

### 6. Label Assignment
```python
# Libraries: numpy
MAINTAIN three parallel arrays:
- semantic_labels (1=wall, 2=pipe)
- instance_labels (object-specific ID)
- distances (closest found distance)
```

### 7. Output Generation
```python
# Libraries: laspy, pathlib
CREATE new LAS 1.4 file with laspy
COPY original point attributes
ADD extra dimensions for:
- semantic_id (uint32)
- instance_id (uint32)
WRITE compressed LAZ output with do_compress=True
```

### Key Optimization Techniques
1. **Spatial Filtering**: Bounding box culling before precise distance calculation
2. **Batch Processing**: Memory-friendly chunk processing of large point sets
3. **Lazy Updates**: Only overwrite labels when closer match is found
4. **Compressed Output**: LASzip compression for efficient storage 