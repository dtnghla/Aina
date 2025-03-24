# Point Cloud Labeling Project

## Overview
This project is dedicated to processing and labeling 3D point cloud data. The main focus is on identifying and categorizing elements within point cloud data (such as walls, pipes, and other structures) by computing distances between points and mesh objects. Additionally, the project includes training and using a PointGroup model to automatically detect and segment walls and pipes from point cloud data.

## Core Components

### Point Cloud Processing
- Processing of LAZ/LAS files containing point cloud data
- Transformation of coordinates using transformation matrices
- Distance calculations between points and mesh objects

### Labeling System
- Semantic labeling (categorizing points by type, e.g., wall=1, pipe=2)
- Instance labeling (assigning object-specific IDs)
- Distance tracking (storing closest distances for optimization)

### Deep Learning with PointGroup
- Training of PointGroup models for automatic segmentation
- Instance and semantic segmentation of point clouds
- Model optimization and evaluation
- Experiment tracking and analysis

### Output Generation
- Creation of labeled point cloud data in LAZ/LAS format
- Storage of semantic and instance labels as additional dimensions
- Visualization of segmentation results

## Technologies Used
- Python for core processing logic
- Libraries: laspy, numpy, trimesh for traditional point cloud processing
- PyTorch for deep learning model training
- TensorBoard for experiment visualization
- Memory optimization techniques for handling large datasets

