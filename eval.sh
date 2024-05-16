#!/bin/bash\

# sample surface point cloud 
python sample_points.py --in_dir path/to/your/generated/samples/folder --out_dir sampled_pointcloud

# Evaluate MMD/COV/JSD
python pc_metric.py  --fake sampled_pointcloud --real deepcad_test_pcd