#!/usr/bin/env python3
"""
Test script to demonstrate the transform_from_object_frame methods.
Shows how to use the new methods to undo transformations.
"""

import numpy as np
import sys
import os

# Add the current directory to path to import from bitsi_slicer
sys.path.append('/home/khiem/bitsi')
from bitsi_slicer.bitsi_slicer.point_cloud import PointCloud

def test_transform_undo():
    """Test the transform_from_object_frame methods."""
    
    print("🧪 Testing transform_from_object_frame methods...")
    print("=" * 50)
    
    # Create a simple point cloud for testing
    # Create a simple cube with 8 vertices
    cube_points = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
    ])
    
    print(f"📦 Original cube points:")
    print(cube_points)
    
    # Create PointCloud object
    cloud = PointCloud()
    
    # Set up some dummy transformation matrices for testing
    # In real usage, these would be computed by transform_to_object_frame()
    cloud.R_base = np.identity(3)
    cloud.R_bounding_box = np.array([
        [0.707, -0.707, 0],
        [0.707, 0.707, 0],
        [0, 0, 1]
    ])  # 45-degree rotation around Z-axis
    cloud.p_bounding_box = np.array([[2.0], [3.0], [1.0]])  # Translation
    
    # Compute R_object
    cloud.R_object = np.matmul(cloud.R_base, cloud.R_bounding_box)
    
    print(f"\n🔄 Transformation matrices:")
    print(f"R_object:\n{cloud.R_object}")
    print(f"p_bounding_box:\n{cloud.p_bounding_box.flatten()}")
    
    # Test single point transformation
    print(f"\n🧪 Testing single point transformation...")
    test_point = np.array([0.5, 0.5, 0.5])
    print(f"Original point: {test_point}")
    
    # Transform to object frame (simulate what transform_to_object_frame does)
    point_obj_frame = np.dot(cloud.R_object.T, (test_point.reshape(3, 1) - cloud.p_bounding_box)).flatten()
    print(f"Point in object frame: {point_obj_frame}")
    
    # Transform back to world frame using our new method
    point_world_frame = cloud.transform_from_object_frame(point_obj_frame)
    print(f"Point back to world frame: {point_world_frame}")
    print(f"Difference from original: {np.abs(test_point - point_world_frame).max()}")
    
    # Test multiple points
    print(f"\n🧪 Testing multiple points transformation...")
    test_points = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    print(f"Original points:\n{test_points}")
    
    # Transform to object frame
    points_obj_frame = np.zeros_like(test_points)
    for i, point in enumerate(test_points):
        points_obj_frame[i] = np.dot(cloud.R_object.T, (point.reshape(3, 1) - cloud.p_bounding_box)).flatten()
    print(f"Points in object frame:\n{points_obj_frame}")
    
    # Transform back to world frame
    points_world_frame = cloud.transform_from_object_frame(points_obj_frame)
    print(f"Points back to world frame:\n{points_world_frame}")
    print(f"Max difference from original: {np.abs(test_points - points_world_frame).max()}")
    
    # Test batch transformation
    print(f"\n🧪 Testing batch transformation...")
    points_list = [
        np.array([[0, 0, 0], [1, 0, 0]]),
        np.array([[0, 1, 0], [0, 0, 1]])
    ]
    print(f"Original points list:")
    for i, pts in enumerate(points_list):
        print(f"  Set {i}: {pts}")
    
    # Transform to object frame
    points_list_obj_frame = []
    for points in points_list:
        points_obj_frame = np.zeros_like(points)
        for i, point in enumerate(points):
            points_obj_frame[i] = np.dot(cloud.R_object.T, (point.reshape(3, 1) - cloud.p_bounding_box)).flatten()
        points_list_obj_frame.append(points_obj_frame)
    
    print(f"Points in object frame:")
    for i, pts in enumerate(points_list_obj_frame):
        print(f"  Set {i}: {pts}")
    
    # Transform back using batch method
    points_list_world_frame = cloud.transform_from_object_frame_batch(points_list_obj_frame)
    print(f"Points back to world frame:")
    for i, pts in enumerate(points_list_world_frame):
        print(f"  Set {i}: {pts}")
    
    # Check accuracy
    max_error = 0
    for orig, world in zip(points_list, points_list_world_frame):
        error = np.abs(orig - world).max()
        max_error = max(max_error, error)
    print(f"Max error across all transformations: {max_error}")
    
    if max_error < 1e-10:
        print("✅ All transformations are accurate!")
    else:
        print("❌ Some transformations have errors!")
    
    print("\n🎉 Transform undo test completed!")

if __name__ == "__main__":
    test_transform_undo()
