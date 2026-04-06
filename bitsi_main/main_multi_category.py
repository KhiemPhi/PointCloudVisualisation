#!/usr/bin/env python3
"""
Multi-category scene generation script.
Loads objects from different categories and places them in a scene without segmentation.
Works with all datasets specified in the original main.py.
"""

import argparse
import open3d as o3d 
import sys
import os
import numpy as np 
import matplotlib.pyplot as plt
import random
import math
from scipy.ndimage import label
from scipy.spatial import cKDTree
from bitsi_slicer.bitsi_slicer.slicing import get_overall_ibr_ratio

# Import functions from the original main.py
from main import (
    adjust_handal, adjust_kitti, get_random_object, get_category_mapping, 
    get_available_categories, load_point_cloud_shapenetpart,
    load_point_cloud_input, segment_single_point_cloud, SegmentNode, rename_segments
)

class SceneObject:
    """Represents an object in the scene with its position and orientation."""
    
    def __init__(self, pcd, name, position=(0, 0, 0), rotation=0, scale=1.0):
        self.pcd = pcd
        self.name = name
        self.position = np.array(position)
        self.rotation = rotation
        self.scale = scale
        self.transformed_pcd = None
        
    def apply_transformation(self):
        """Apply position, rotation, and scale transformations to the point cloud."""
        # Create a copy of the point cloud
        self.transformed_pcd = o3d.geometry.PointCloud(self.pcd)
        
        # Apply scale
        if self.scale != 1.0:
            self.transformed_pcd.scale(self.scale, center=self.transformed_pcd.get_center())
        
        # Apply rotation around Z-axis
        if self.rotation != 0:
            rotation_matrix = np.array([
                [np.cos(self.rotation), -np.sin(self.rotation), 0],
                [np.sin(self.rotation), np.cos(self.rotation), 0],
                [0, 0, 1]
            ])
            self.transformed_pcd.rotate(rotation_matrix, center=self.transformed_pcd.get_center())
        
        # Apply translation
        self.transformed_pcd.translate(self.position)
        
        return self.transformed_pcd

def load_random_object_from_dataset(parent_dir, object_name, obj_num=None):
    """
    Load a random object from the specified dataset.
    
    Parameters
    ----------
    parent_dir : str
        Dataset directory (HANDAL, YCBV, KITTI, ShapeNetPart)
    object_name : str
        Object name/category
    obj_num : int, optional
        Object number for HANDAL dataset
        
    Returns
    -------
    tuple
        (pcd, object_info) where object_info contains metadata
    """
    if parent_dir == 'HANDAL':
        if obj_num is None:
            obj_num = random.randint(1, 100)  # Random object number
        
        model_num = f"{obj_num:0{6}}"
        parent_dir_path = os.path.join("/home/khiem/Robotics/obj-decomposition/", parent_dir)
        
        pcd = o3d.io.read_point_cloud(f"{parent_dir_path}/{object_name}/models/obj_{model_num}.ply")
        pcd = adjust_handal(pcd)
        pcd.paint_uniform_color([0, 0, 1])
        
        object_info = {
            'type': 'HANDAL',
            'object_name': object_name,
            'obj_num': obj_num,
            'model_num': model_num
        }
        
    elif parent_dir == "YCBV" or parent_dir == 'YCBV-Partial':
        parent_dir_path = os.path.join("/home/khiem/Point-Cloud-Decomposition-for-Grasping", parent_dir)
        pcd = o3d.io.read_point_cloud(f"{parent_dir_path}/{object_name}/nontextured.ply")
        pcd.paint_uniform_color([0, 0, 1])
        
        object_info = {
            'type': 'YCBV',
            'object_name': object_name
        }
        
    elif parent_dir == 'KITTI':
        parent_dir_path = os.path.join("/home/khiem/Robotics/obj-decomposition/", parent_dir)
        pcd = o3d.io.read_point_cloud(f"{parent_dir_path}/{object_name}.ply")
        pcd = adjust_kitti(pcd)
        
        object_info = {
            'type': 'KITTI',
            'object_name': object_name
        }
        
    elif parent_dir == 'ShapeNetPart':
        points, normals, part_ids, file_path, category_id = get_random_object(object_name)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.paint_uniform_color([0, 0, 1])
        
        object_info = {
            'type': 'ShapeNetPart',
            'object_name': object_name,
            'category_id': category_id,
            'file_path': file_path,
            'points': points,
            'normals': normals,
            'part_ids': part_ids
        }
    
    else:
        raise ValueError(f"Unknown dataset: {parent_dir}")
    
    if pcd.is_empty():
        return None, None
        
    return pcd, object_info

def get_available_objects_for_dataset(parent_dir, root_dir="/home/khiem/Point-Cloud-Decomposition-for-Grasping"):
    """
    Get available objects for a specific dataset.
    
    Parameters
    ----------
    parent_dir : str
        Dataset directory
        
    Returns
    -------
    list
        List of available object names
    """
    if parent_dir == 'HANDAL':
        # Get available objects from HANDAL dataset
        handal_path = os.path.join(root_dir, "HANDAL")
        if os.path.exists(handal_path):
            return [d for d in os.listdir(handal_path) if os.path.isdir(os.path.join(handal_path, d))]
        return []
    
    elif parent_dir == "YCBV" or parent_dir == 'YCBV-Partial':
        # Get available objects from YCBV dataset
        ycbv_path = os.path.join(root_dir, parent_dir)
        if os.path.exists(ycbv_path):
            return [d for d in os.listdir(ycbv_path) if os.path.isdir(os.path.join(ycbv_path, d))]
        return []
    
    elif parent_dir == 'KITTI':
        # Get available objects from KITTI dataset
        kitti_path = os.path.join(root_dir, "KITTI")
        if os.path.exists(kitti_path):
            return [f.replace('.ply', '') for f in os.listdir(kitti_path) if f.endswith('.ply')]
        return []
    
    elif parent_dir == 'ShapeNetPart':
        # Get available categories from ShapeNetPart
        available_categories = get_available_categories()
        return [cat[0] for cat in available_categories]  # Return category names
    
    return []

def generate_scene_layout(num_objects, scene_size=2.0, adversarial=False):
    """
    Generate random positions for objects in the scene.
    
    Parameters
    ----------
    num_objects : int
        Number of objects to place
    scene_size : float
        Size of the scene (radius)
    adversarial : bool
        If True, place objects close together in pairs/triplets along x-axis
        
    Returns
    -------
    list
        List of (position, rotation, scale) tuples
    """
    layouts = []
    
    if adversarial:
        # Adversarial mode: place objects close together in pairs or triplets
        # Objects will be touching or nearly touching in the x-direction
        print(f"⚔️  Adversarial mode: placing {num_objects} objects in close pairs/triplets")
        
        # Small spacing between objects in a group (simulating touching)
        touch_spacing = 0.01  # Very small gap to simulate touching
        
        # Group objects into pairs and triplets
        object_idx = 0
        while object_idx < num_objects:
            # Decide group size: prefer pairs, but use triplets if needed
            remaining = num_objects - object_idx
            if remaining >= 3 and random.random() < 0.4:  # 40% chance for triplet
                group_size = 3
            elif remaining >= 2:
                group_size = 2
            else:
                group_size = 1
            
            # Base x position for this group
            base_x = random.uniform(-scene_size * 0.7, scene_size * 0.7)
            base_y = random.uniform(-scene_size * 0.7, scene_size * 0.7)
            
            # Place objects in the group close together along x-axis
            for j in range(group_size):
                if object_idx >= num_objects:
                    break
                
                # X position: base position + small offset for each object in group
                x = base_x + j * touch_spacing
                
                # Y position: random within scene bounds
                y = base_y + j * touch_spacing
                z = 0  # fixed height (no random z)
                
                rotation = 0.0  # fixed rotation
                scale = 1.0     # fixed scale
                
                layouts.append(((x, y, z), rotation, scale))
                object_idx += 1
            
            print(f"  📦 Group of {group_size} objects at x ≈ {base_x:.3f}")
    else:
        # Normal mode: random positions
        for i in range(num_objects):
            # Random x, y position within scene bounds
            x = random.uniform(-scene_size, scene_size)
            y = random.uniform(-scene_size, scene_size)
            z = 0  # fixed height (no random z)

            rotation = 0.0  # fixed rotation
            scale = 1.0     # fixed scale

            layouts.append(((x, y, z), rotation, scale))
            
    return layouts

def octree_based_segmentation(pcd, max_depth=8, min_points_per_voxel=1):
    """
    Octree-based segmentation using voxel adjacency.
    
    Steps:
    1. Build an octree from the point cloud
    2. Look for empty space boundaries between objects
    3. Group leaves connected through occupied nodes
    
    Parameters
    ----------
    pcd : o3d.geometry.PointCloud
        Input point cloud
    max_depth : int
        Maximum depth of the octree
    min_points_per_voxel : int
        Minimum number of points required for a voxel to be considered occupied
        
    Returns
    -------
    list of o3d.geometry.PointCloud
        List of segmented point clouds, one for each connected component
    """
    print(f"\n🌳 Applying octree-based segmentation (max_depth={max_depth})...")
    
    if pcd.is_empty() or len(pcd.points) == 0:
        print("⚠️ Empty point cloud, returning empty list")
        return []
    
    # Build octree (for reference, though we use voxel grid for actual segmentation)
    octree = o3d.geometry.Octree(max_depth=max_depth)
    octree.convert_from_point_cloud(pcd, size_expand=0.01)
    
    # Use voxel grid to identify occupied voxels (more straightforward than octree traversal)
    # The voxel grid approach uses the same underlying concept as octree but is easier to work with
    points = np.asarray(pcd.points)
    
    # Calculate appropriate voxel size based on point cloud extent
    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    # Use a voxel size that gives us roughly 2^max_depth voxels along the longest dimension
    max_extent = np.max(extent)
    voxel_size = max_extent / (2 ** max_depth)
    
    print(f"  📏 Voxel size: {voxel_size:.6f}")
    
    # Create voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    
    # Get all occupied voxels
    occupied_voxels = voxel_grid.get_voxels()
    
    if len(occupied_voxels) == 0:
        print("⚠️ No occupied voxels found")
        return []
    
    print(f"  📦 Found {len(occupied_voxels)} occupied voxels")
    
    # Create a 3D grid to represent occupied voxels
    # Get the bounding box of voxels
    voxel_indices = np.array([voxel.grid_index for voxel in occupied_voxels])
    min_idx = voxel_indices.min(axis=0)
    max_idx = voxel_indices.max(axis=0)
    
    # Normalize indices to start from 0
    normalized_indices = voxel_indices - min_idx
    grid_shape = (max_idx - min_idx + 1).astype(int)
    
    # Create 3D occupancy grid and map voxels to points
    occupancy_grid = np.zeros(grid_shape, dtype=bool)
    voxel_to_points = {}  # Map voxel index to point indices
    
    # Get voxel grid origin and voxel size for coordinate conversion
    voxel_grid_origin = voxel_grid.get_min_bound()
    voxel_size_actual = voxel_grid.voxel_size
    
    # Map points to voxels using the voxel grid's coordinate system
    points_array = np.asarray(pcd.points)
    for i, point in enumerate(points_array):
        # Convert point to voxel coordinates
        voxel_coords = ((point - voxel_grid_origin) / voxel_size_actual).astype(int)
        voxel_idx = tuple(voxel_coords)
        
        # Normalize to our grid coordinate system
        normalized_voxel = tuple(voxel_idx - min_idx)
        if all(0 <= idx < grid_shape[j] for j, idx in enumerate(normalized_voxel)):
            occupancy_grid[normalized_voxel] = True
            if normalized_voxel not in voxel_to_points:
                voxel_to_points[normalized_voxel] = []
            voxel_to_points[normalized_voxel].append(i)
    
    # Also mark voxels that are in the occupied_voxels list
    for voxel in occupied_voxels:
        voxel_idx = tuple(voxel.grid_index)
        normalized_voxel = tuple(np.array(voxel_idx) - min_idx)
        if all(0 <= idx < grid_shape[j] for j, idx in enumerate(normalized_voxel)):
            occupancy_grid[normalized_voxel] = True
    
    # Find connected components in 3D grid (26-connectivity for 3D)
    structure = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity
    labeled_grid, num_components = label(occupancy_grid, structure=structure)
    
    print(f"  🔗 Found {num_components} connected components")
    
    # Group points by connected component
    segmented_pcds = []
    component_point_indices = {}
    
    for voxel_idx_tuple, point_indices in voxel_to_points.items():
        voxel_idx = np.array(voxel_idx_tuple)
        if all(0 <= idx < grid_shape[j] for j, idx in enumerate(voxel_idx)):
            component_id = labeled_grid[tuple(voxel_idx)]
            if component_id > 0:  # Skip background (0)
                if component_id not in component_point_indices:
                    component_point_indices[component_id] = []
                component_point_indices[component_id].extend(point_indices)
    
    # Create point clouds for each component
    for component_id, point_indices in component_point_indices.items():
        if len(point_indices) >= min_points_per_voxel:
            component_points = points_array[point_indices]
            component_pcd = o3d.geometry.PointCloud()
            component_pcd.points = o3d.utility.Vector3dVector(component_points)
            
            # Copy colors if available
            if pcd.has_colors():
                component_colors = np.asarray(pcd.colors)[point_indices]
                component_pcd.colors = o3d.utility.Vector3dVector(component_colors)
            
            # Copy normals if available
            if pcd.has_normals():
                component_normals = np.asarray(pcd.normals)[point_indices]
                component_pcd.normals = o3d.utility.Vector3dVector(component_normals)
            
            segmented_pcds.append(component_pcd)
    
    # Sort by number of points (largest first)
    segmented_pcds.sort(key=lambda pcd: len(pcd.points), reverse=True)
    
    print(f"  ✅ Created {len(segmented_pcds)} segments")
    for i, seg_pcd in enumerate(segmented_pcds):
        print(f"     Segment {i+1}: {len(seg_pcd.points)} points")
    
    return segmented_pcds

def create_multi_category_scene(parent_dir, object_categories, scene_size=2.0, 
                               gripper_width=0.05, gripper_height=0.07, epsilon=5e-3,
                               strength_threshold=0.001, ibr_tolerance=0.02, adversarial=False):
    """
    Create a multi-category scene with objects from different categories.
    
    Parameters
    ----------
    parent_dir : str
        Dataset directory
    object_categories : list
        List of object categories to load
    scene_size : float
        Size of the scene
    gripper_width, gripper_height : float
        Gripper parameters
    epsilon : float
        BITSI epsilon parameter
    strength_threshold : float
        Inflection detection threshold
    ibr_tolerance : float
        Segment fusion tolerance
    adversarial : bool
        If True, use adversarial layout (objects close together)
        
    Returns
    -------
    tuple
        (scene_pcd, scene_objects, segmentation_results)
    """
    print(f"🎬 Creating multi-category scene with {len(object_categories)} different objects from {parent_dir}")
    if adversarial:
        print("⚔️  Adversarial mode enabled: objects will be placed close together")
    print("=" * 60)
    
    # Generate scene layout
    layouts = generate_scene_layout(len(object_categories), scene_size, adversarial=adversarial)
    
    # Load objects
    scene_objects = []
    all_point_clouds = []
    
    for i, (object_name, (position, rotation, scale)) in enumerate(zip(object_categories, layouts)):
        print(f"📦 Loading object {i+1}/{len(object_categories)}: {object_name}...")
        
        # Load object
        pcd, object_info = load_random_object_from_dataset(parent_dir, object_name)
        
        if pcd is None:
            print(f"⚠️ Failed to load {object_name}, skipping...")
            continue
        
        # Create scene object
        scene_obj = SceneObject(pcd, f"{object_name}_{i+1}", position, rotation, scale)
        scene_objects.append((scene_obj, object_info))
        
        # Apply transformations
        transformed_pcd = scene_obj.apply_transformation()
        
        # Assign unique color to each object
        color = plt.cm.get_cmap("tab20", len(object_categories))(i)[:3]
        transformed_pcd.paint_uniform_color(color)
        
        all_point_clouds.append(transformed_pcd)
        print(f"  ✅ {object_name}: {len(transformed_pcd.points)} points at {position}")
    
    if not all_point_clouds:
        print("❌ No objects loaded successfully!")
        return None, None, None
    
    # Combine all point clouds into scene
    scene_pcd = all_point_clouds[0]
    for pcd in all_point_clouds[1:]:
        scene_pcd += pcd
    
    print(f"\n🎯 Scene created: {len(scene_pcd.points)} total points")
    print(f"📏 Scene bounds: {scene_pcd.get_axis_aligned_bounding_box().get_extent()}")
    
    # Step 1: Apply octree-based segmentation to separate objects
    print("\n" + "=" * 60)
    print("Step 1: Octree-based Segmentation")
    print("=" * 60)
    segmented_pcds = octree_based_segmentation(scene_pcd, max_depth=8, min_points_per_voxel=10)
    
    if not segmented_pcds:
        print("⚠️ Octree segmentation produced no segments, using original point cloud")
        segmented_pcds = [scene_pcd]
    
    # Step 2: Apply BITSI segmentation to each segment
    print("\n" + "=" * 60)
    print("Step 2: BITSI Segmentation on Octree Segments")
    print("=" * 60)
    
    try:
        # Import BITSI functions
        from main import (
            build_cloud_object, bitsi_metric, build_segmentation_tree, 
            fuse_consecutive_segments, auto_thickness, pretty_print_bitsi, SegmentNode
        )
        
        all_roots = []
        all_segmentation_info = []
        
        for seg_idx, seg_pcd in enumerate(segmented_pcds):
            print(f"\n📦 Processing octree segment {seg_idx + 1}/{len(segmented_pcds)} ({len(seg_pcd.points)} points)...")
            
            # Build cloud object for this segment
            seg_cloud_object = build_cloud_object(seg_pcd, gripper_width, gripper_height)
            overall_ibr_ratio = get_overall_ibr_ratio(seg_cloud_object, epsilon=epsilon, thickness=0.001, gripper_length=gripper_width, base_ibr=0.7)
            
            # Check if segment exceeds gripper width in x or y dimensions
            if (seg_cloud_object.x_dim <= gripper_width and seg_cloud_object.y_dim <= gripper_width) or overall_ibr_ratio >= 0.7:
                print(f"  ⚠️ Segment {seg_idx + 1} does not exceed gripper width (x={seg_cloud_object.x_dim:.4f}, y={seg_cloud_object.y_dim:.4f}), skipping BITSI segmentation")
                # Create a simple root node with all points
                all_points = np.asarray(seg_pcd.points).tolist()
                simple_root = SegmentNode(
                    name=f"octree_segment_{seg_idx + 1}", 
                    points=all_points, 
                    slice_idx=0
                )
                all_roots.append(simple_root)
                all_segmentation_info.append({
                    'octree_segment_idx': seg_idx + 1,
                    'root': simple_root,
                    'bitsi_x': None,
                    'bitsi_y': None,
                    'bitsi_z': None,
                    'slice_idx': 0,
                    'thickness': None,
                    'cloud_object': seg_cloud_object,
                    'point_cloud': seg_pcd,
                    'skipped': True
                })
                continue
            
            # Determine slice_idx based on largest bounding box dimension
            dimensions = np.array([seg_cloud_object.x_dim, seg_cloud_object.y_dim, seg_cloud_object.z_dim])
            seg_slice_idx = int(np.argmax(dimensions))
            print(f"  📐 Bounding box dimensions: x={seg_cloud_object.x_dim:.4f}, y={seg_cloud_object.y_dim:.4f}, z={seg_cloud_object.z_dim:.4f}")
            print(f"  🎯 Using slice_idx={seg_slice_idx} (largest dimension: {['X', 'Y', 'Z'][seg_slice_idx]})")
            
            seg_thickness = auto_thickness(seg_cloud_object, scale=0.001)
            print(f"  📏 Adaptive thickness: {seg_thickness:.6f}")
            
            # Compute BITSI metrics for this segment
            seg_bitsi_x, seg_bitsi_y, seg_bitsi_z, _, seg_points_per_slice_x, seg_points_per_slice_y, seg_points_per_slice_z = bitsi_metric(
                seg_cloud_object, epsilon=epsilon, thickness=seg_thickness
            )
            
            seg_points_per_slice = [seg_points_per_slice_x, seg_points_per_slice_y, seg_points_per_slice_z]
            seg_points_per_slice_to_use = seg_points_per_slice[seg_slice_idx]
            
            # Print BITSI metrics for this segment
            print(f"  📊 BITSI metrics for segment {seg_idx + 1}:")
            pretty_print_bitsi(seg_bitsi_x, seg_bitsi_y, seg_bitsi_z, seg_slice_idx)
            
            # Build segmentation tree for this segment
            try:
                seg_root = build_segmentation_tree(
                    seg_points_per_slice_to_use, 
                    seg_bitsi_x, seg_bitsi_y, seg_bitsi_z, 
                    seg_slice_idx, 
                    strength_threshold, 
                    max_deg=10, 
                    visualize=False
                )
                seg_root.name = f"octree_segment_{seg_idx + 1}"
                all_roots.append(seg_root)
                
                all_segmentation_info.append({
                    'octree_segment_idx': seg_idx + 1,
                    'root': seg_root,
                    'bitsi_x': seg_bitsi_x,
                    'bitsi_y': seg_bitsi_y,
                    'bitsi_z': seg_bitsi_z,
                    'slice_idx': seg_slice_idx,
                    'thickness': seg_thickness,
                    'cloud_object': seg_cloud_object,
                    'point_cloud': seg_pcd,
                    'skipped': False
                })
                
                print(f"  ✅ Created segmentation tree with {len(seg_root.children)} children")
            except Exception as e:
                print(f"  ⚠️ Failed to build segmentation tree for segment {seg_idx + 1}: {e}")
                # Create a simple root node with all points
                all_points = []
                for slice_points in seg_points_per_slice_to_use:
                    all_points.extend(slice_points)
                simple_root = SegmentNode(
                    name=f"octree_segment_{seg_idx + 1}", 
                    points=all_points, 
                    slice_idx=seg_slice_idx
                )
                all_roots.append(simple_root)
                all_segmentation_info.append({
                    'octree_segment_idx': seg_idx + 1,
                    'root': simple_root,
                    'bitsi_x': seg_bitsi_x,
                    'bitsi_y': seg_bitsi_y,
                    'bitsi_z': seg_bitsi_z,
                    'slice_idx': seg_slice_idx,
                    'thickness': seg_thickness,
                    'cloud_object': seg_cloud_object,
                    'point_cloud': seg_pcd,
                    'skipped': False
                })
                print(f"  ✅ Created simple root node with {len(all_points)} points")
        
        # Create a combined root that contains all segment roots as children
        if len(all_roots) > 1:
            # Combine all points from all roots
            combined_points = []
            for root in all_roots:
                def gather_all_points(node):
                    points = list(node.points) if hasattr(node, 'points') else []
                    for child in node.children:
                        points.extend(gather_all_points(child))
                    return points
                combined_points.extend(gather_all_points(root))
            
            # Create a parent root
            combined_root = SegmentNode(
                name="combined_octree_segments",
                points=combined_points,
                slice_idx=0  # Default slice index
            )
            combined_root.children = all_roots
            root = combined_root
        elif len(all_roots) == 1:
            root = all_roots[0]
        else:
            print("⚠️ No roots created, creating empty root")
            root = SegmentNode(name="empty_root", points=[], slice_idx=0)
        
        print(f"\n✅ BITSI segmentation completed on {len(segmented_pcds)} octree segments!")
        
        # Create segmentation results dictionary
        segmentation_results = {
            'root': root,
            'octree_segments': segmented_pcds,
            'segment_roots': all_roots,
            'segmentation_info': all_segmentation_info,
            'num_octree_segments': len(segmented_pcds)
        }
        
    except Exception as e:
        print(f"❌ BITSI segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        segmentation_results = None
    
    return scene_pcd, scene_objects, segmentation_results

def plot_segment_on_axis(ax, points, title, colors=None):
    """Helper function to plot points on a 3D axis."""
    if points is None or len(points) == 0:
        ax.text(0.5, 0.5, 0.5, "No points", ha='center', va='center')
        ax.set_title(title)
        return
    
    points = np.asarray(points)
    
    if colors is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  s=5, c=colors, alpha=0.8)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                  s=5, color=[0.2, 0.6, 0.8], alpha=0.8)
    
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    # Fix axis scaling
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    
    x_range = x_max - x_min
    y_range = y_max - y_min
    z_range = z_max - z_min
    
    max_range = max(x_range, y_range, z_range) if max(x_range, y_range, z_range) > 0 else 1.0
    
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    z_center = (z_max + z_min) / 2
    
    ax.set_xlim(x_center - max_range/2, x_center + max_range/2)
    ax.set_ylim(y_center - max_range/2, y_center + max_range/2)
    ax.set_zlim(z_center - max_range/2, z_center + max_range/2)
    ax.set_box_aspect([1, 1, 1])

def visualize_octree_segments_side_by_side(segmentation_info_list, combined_root):
    """
    Visualize octree segments side by side.
    - Non-subdivided segments: show as whole segments
    - Subdivided segments: show all children (not the root)
    - Also create separate figures for each combined octree segment
    
    Parameters
    ----------
    segmentation_info_list : list of dict
        List of segmentation info dictionaries from create_multi_category_scene
    combined_root : SegmentNode
        The combined root containing all octree segments
    """
    if not segmentation_info_list:
        print("⚠️ No segmentation info to visualize")
        return
    
    def has_children(root):
        """Check if root has any children (has been subdivided)."""
        return root is not None and len(root.children) > 0
    
    def gather_all_children(node):
        """Recursively gather all leaf children nodes."""
        children = []
        if node is None:
            return children
        for child in node.children:
            if len(child.children) == 0:
                # Leaf node
                children.append(child)
            else:
                # Recursively get children of this child
                children.extend(gather_all_children(child))
        return children
    
    # Collect all segments (flattened list: non-subdivided segments + children of subdivided segments)
    all_segments_flat = []
    
    for seg_info in segmentation_info_list:
        if seg_info.get('skipped', False):
            continue
        root = seg_info.get('root')
        if root is None:
            continue
        
        seg_idx = seg_info['octree_segment_idx']
        slice_idx = seg_info['slice_idx']
        axis_name = ['X', 'Y', 'Z'][slice_idx]
        point_cloud = seg_info.get('point_cloud')
        
        if has_children(root):
            # Subdivided segment: add all children to the flat list
            children = gather_all_children(root)
            valid_children = [c for c in children if len(c.points) > 0]
            if valid_children:
                print(f"  📊 Octree segment {seg_idx} is subdivided: adding {len(valid_children)} children")
                for child in valid_children:
                    all_segments_flat.append({
                        'type': 'child',
                        'parent_seg_idx': seg_idx,
                        'points': np.asarray(child.points),
                        'name': getattr(child, 'display_name', child.name)
                    })
        else:
            # Non-subdivided segment: add to flat list
            if point_cloud is not None and len(point_cloud.points) > 0:
                print(f"  📊 Octree segment {seg_idx} is not subdivided: adding whole segment")
                all_segments_flat.append({
                    'type': 'whole',
                    'parent_seg_idx': seg_idx,
                    'points': np.asarray(point_cloud.points),
                    'name': f"octree_segment_{seg_idx}"
                })
    
    if not all_segments_flat:
        print("⚠️ No segments to visualize")
        return
    
    total_segments = len(all_segments_flat)
    print(f"\n🎨 Visualizing {total_segments} total segments in a single image...")
    
    # Create colormap based on total number of segments
    if total_segments <= 20:
        cmap = plt.cm.get_cmap("tab20", total_segments)
    else:
        cmap = plt.cm.get_cmap("gist_rainbow", total_segments)
    
    # Create single figure with all segments
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    
    
    
    # Create separate figures for each combined octree segment (child of combined_root)
    # Also collect all points and colors for combined point cloud
    all_combined_points = []
    all_combined_colors = []
    
    if combined_root is not None and has_children(combined_root):
        # Count total number of segments (subdivided children + non-subdivided segments)
        # to create a colormap for unique segment colors
        total_segments = 0
        for octree_seg_root in combined_root.children:
            if has_children(octree_seg_root):
                children = gather_all_children(octree_seg_root)
                valid_children = [c for c in children if len(c.points) > 0]
                total_segments += len(valid_children)
            else:
                if hasattr(octree_seg_root, 'points') and len(octree_seg_root.points) > 0:
                    total_segments += 1
        
        # Create colormap for all segments
        if total_segments <= 20:
            global_cmap = plt.cm.get_cmap("tab20", total_segments)
        else:
            global_cmap = plt.cm.get_cmap("gist_rainbow", total_segments)
        
        global_segment_idx = 0
        
        print(f"\n🎨 Creating separate figures for each combined octree segment...")
        for child_idx, octree_seg_root in enumerate(combined_root.children):
            seg_name = getattr(octree_seg_root, 'name', f'segment_{child_idx + 1}')
            print(f"  📊 Creating figure for {seg_name}...")
            
            # Check if this segment has children (is subdivided)
            if has_children(octree_seg_root):
                # Show children
                children = gather_all_children(octree_seg_root)
                valid_children = [c for c in children if len(c.points) > 0]
                
                if valid_children:
                    fig_seg = plt.figure(figsize=(12, 10))
                    ax_seg = fig_seg.add_subplot(111, projection="3d")
                    
                    num_valid = len(valid_children)
                    if num_valid <= 20:
                        cmap = plt.cm.get_cmap("tab20", num_valid)
                    else:
                        cmap = plt.cm.get_cmap("gist_rainbow", num_valid)
                    
                    all_pts = []
                    for i, child in enumerate(valid_children):
                        pts = np.asarray(child.points)
                        all_pts.append(pts)
                        color = cmap(i)[:3]
                        label = getattr(child, 'display_name', child.name)
                        ax_seg.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5, color=color, alpha=0.8, label=label)
                        
                        # Collect points and colors for combined point cloud
                        # Use global colormap for consistent coloring across all segments
                        global_color = global_cmap(global_segment_idx)[:3]
                        all_combined_points.append(pts)
                        # Create color array for all points in this child
                        num_points = len(pts)
                        child_colors = np.tile(global_color, (num_points, 1))
                        all_combined_colors.append(child_colors)
                        global_segment_idx += 1
                    
                    ax_seg.set_title(f"{seg_name}\n{len(valid_children)} sub-segments")
                    ax_seg.set_xlabel("X")
                    ax_seg.set_ylabel("Y")
                    ax_seg.set_zlabel("Z")
                    
                    # Fix axis scaling
                    if all_pts:
                        all_points_array = np.vstack(all_pts)
                        x_min, x_max = all_points_array[:, 0].min(), all_points_array[:, 0].max()
                        y_min, y_max = all_points_array[:, 1].min(), all_points_array[:, 1].max()
                        z_min, z_max = all_points_array[:, 2].min(), all_points_array[:, 2].max()
                        
                        x_range = x_max - x_min
                        y_range = y_max - y_min
                        z_range = z_max - z_min
                        
                        max_range = max(x_range, y_range, z_range) if max(x_range, y_range, z_range) > 0 else 1.0
                        
                        x_center = (x_max + x_min) / 2
                        y_center = (y_max + y_min) / 2
                        z_center = (z_max + z_min) / 2
                        
                        ax_seg.set_xlim(x_center - max_range/2, x_center + max_range/2)
                        ax_seg.set_ylim(y_center - max_range/2, y_center + max_range/2)
                        ax_seg.set_zlim(z_center - max_range/2, z_center + max_range/2)
                        ax_seg.set_box_aspect([1, 1, 1])
                    
                    ax_seg.legend(loc="upper right", fontsize=8)
                    plt.tight_layout()
                    plt.show()
                    
                    # Save figure
                    safe_name = seg_name.replace(" ", "_").replace("/", "_")
                    output_filename = f"{safe_name}_visualization.png"
                    fig_seg.savefig(output_filename, dpi=150, bbox_inches='tight')
                    print(f"    💾 Saved to {output_filename}")
            else:
                # Show whole segment
                if hasattr(octree_seg_root, 'points') and len(octree_seg_root.points) > 0:
                    fig_seg = plt.figure(figsize=(12, 10))
                    ax_seg = fig_seg.add_subplot(111, projection="3d")
                    
                    points = np.asarray(octree_seg_root.points)
                    plot_segment_on_axis(ax_seg, points, seg_name)
                    
                    # Collect points and colors for combined point cloud
                    # Use global colormap for consistent coloring across all segments
                    global_color = global_cmap(global_segment_idx)[:3]
                    all_combined_points.append(points)
                    num_points = len(points)
                    segment_colors = np.tile(global_color, (num_points, 1))
                    all_combined_colors.append(segment_colors)
                    global_segment_idx += 1
                    
                    plt.tight_layout()
                    plt.show()
                    
                    # Save figure
                    safe_name = seg_name.replace(" ", "_").replace("/", "_")
                    output_filename = f"{safe_name}_visualization.png"
                    fig_seg.savefig(output_filename, dpi=150, bbox_inches='tight')
                    print(f"    💾 Saved to {output_filename}")
        
        # Create and save combined point cloud
        if all_combined_points:
            print(f"\n💾 Creating combined point cloud from all segments...")
            combined_points_array = np.vstack(all_combined_points)
            combined_colors_array = np.vstack(all_combined_colors)
            
            # Create Open3D point cloud
            combined_pcd = o3d.geometry.PointCloud()
            combined_pcd.points = o3d.utility.Vector3dVector(combined_points_array)
            combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors_array)
            
            # Save as PLY file
            output_filename = "combined_segments_pointcloud.ply"
            o3d.io.write_point_cloud(output_filename, combined_pcd)
            print(f"    ✅ Saved combined point cloud with {len(combined_points_array)} points to {output_filename}")
            
            # Display combined point cloud in matplotlib
            print(f"    🎨 Displaying combined point cloud visualization...")
            fig_combined = plt.figure(figsize=(12, 10))
            ax_combined = fig_combined.add_subplot(111, projection="3d")
            
            # Plot all points with their colors
            ax_combined.scatter(
                combined_points_array[:, 0], 
                combined_points_array[:, 1], 
                combined_points_array[:, 2], 
                s=5, 
                c=combined_colors_array, 
                alpha=0.8
            )
            
            ax_combined.set_title(f"Combined Segments Point Cloud\n{len(combined_points_array)} points, {len(all_combined_points)} segments")
            ax_combined.set_xlabel("X")
            ax_combined.set_ylabel("Y")
            ax_combined.set_zlabel("Z")
            
            # Fix axis scaling
            x_min, x_max = combined_points_array[:, 0].min(), combined_points_array[:, 0].max()
            y_min, y_max = combined_points_array[:, 1].min(), combined_points_array[:, 1].max()
            z_min, z_max = combined_points_array[:, 2].min(), combined_points_array[:, 2].max()
            
            x_range = x_max - x_min
            y_range = y_max - y_min
            z_range = z_max - z_min
            
            max_range = max(x_range, y_range, z_range) if max(x_range, y_range, z_range) > 0 else 1.0
            
            x_center = (x_max + x_min) / 2
            y_center = (y_max + y_min) / 2
            z_center = (z_max + z_min) / 2
            
            ax_combined.set_xlim(x_center - max_range/2, x_center + max_range/2)
            ax_combined.set_ylim(y_center - max_range/2, y_center + max_range/2)
            ax_combined.set_zlim(z_center - max_range/2, z_center + max_range/2)
            ax_combined.set_box_aspect([1, 1, 1])
            
            plt.tight_layout()
            plt.show()
            
            # Save combined visualization figure
            combined_viz_filename = "combined_segments_visualization.png"
            fig_combined.savefig(combined_viz_filename, dpi=150, bbox_inches='tight')
            print(f"    💾 Saved combined visualization to {combined_viz_filename}")

def visualize_multi_category_scene(scene_pcd, scene_objects, segmentation_results=None, 
                                 show_individual_objects=True, show_segmentation=True):
    """
    Visualize the multi-category scene.
    
    Parameters
    ----------
    scene_pcd : o3d.geometry.PointCloud
        Combined scene point cloud
    scene_objects : list
        List of (SceneObject, object_info) tuples
    segmentation_results : dict, optional
        Segmentation results
    show_individual_objects : bool
        Whether to show individual objects
    show_segmentation : bool
        Whether to show segmentation results
    """
    
    if show_segmentation and segmentation_results is not None:
        # Show segmentation results
        print("\n🔍 Visualizing BITSI segmentation results...")
        root = segmentation_results['root']
        root.print_tree()
        
        # Visualize octree segments side by side
        if 'segmentation_info' in segmentation_results:
            visualize_octree_segments_side_by_side(
                segmentation_results['segmentation_info'],
                segmentation_results.get('root')
            )
        
       

# ===============================================================
# ADDED FOR VISUALIZATION TOOL INTEGRATION
# Reusable multi-object BITSI API that callers can invoke directly
# ===============================================================
def segment_multi_object_point_cloud(input_data, gripper_width=0.13, gripper_height=0.07,
                                     epsilon=5e-3, ibr_tolerance=0.003,
                                     strength_threshold=0.01, octree_max_depth=8,
                                     min_points_per_voxel=10, output_dir=None):
    """Split a multi-object scene, then run BITSI on each object cluster."""
    # ADDED: no automatic single/multi detection here; callers explicitly choose this mode
    scene_pcd = load_point_cloud_input(input_data)
    scene_points = np.asarray(scene_pcd.points)

    segmented_pcds = octree_based_segmentation(
        scene_pcd, max_depth=octree_max_depth, min_points_per_voxel=min_points_per_voxel
    )
    if not segmented_pcds:
        segmented_pcds = [scene_pcd]

    scene_tree = cKDTree(scene_points)
    combined_labels = np.full(len(scene_points), -1, dtype=int)
    object_labels = np.full(len(scene_points), -1, dtype=int)
    combined_root = SegmentNode(name='scene', points=scene_points, slice_idx=0)
    combined_root.children = []

    per_object_results = []
    label_offset = 0
    for obj_idx, seg_pcd in enumerate(segmented_pcds):
        result = segment_single_point_cloud(
            seg_pcd,
            gripper_width=gripper_width,
            gripper_height=gripper_height,
            epsilon=epsilon,
            ibr_tolerance=ibr_tolerance,
            strength_threshold=strength_threshold,
            output_dir=output_dir,
            visualize=False,
            segment_name=f'object_{obj_idx + 1}_segment'
        )

        object_root = rename_segments(result['root'], prefix=f'object_{obj_idx + 1}_segment')
        object_root.parent = combined_root
        combined_root.children.append(object_root)

        seg_points = np.asarray(seg_pcd.points)
        _, scene_idx = scene_tree.query(seg_points, k=1)

        local_labels = np.asarray(result['labels'], dtype=int)
        combined_labels[scene_idx] = local_labels + label_offset
        object_labels[scene_idx] = obj_idx
        label_offset += int(local_labels.max()) + 1 if len(local_labels) else 1

        per_object_results.append({
            'object_index': obj_idx,
            'scene_indices': scene_idx,
            'result': result,
        })

    # Fill any leftover points conservatively from nearest labeled point
    unlabeled = np.where(combined_labels < 0)[0]
    labeled = np.where(combined_labels >= 0)[0]
    if len(unlabeled) and len(labeled):
        backfill_tree = cKDTree(scene_points[labeled])
        _, nn_idx = backfill_tree.query(scene_points[unlabeled], k=1)
        combined_labels[unlabeled] = combined_labels[labeled][nn_idx]
        object_labels[unlabeled] = object_labels[labeled][nn_idx]

    if np.any(combined_labels < 0):
        combined_labels[combined_labels < 0] = 0
    if np.any(object_labels < 0):
        object_labels[object_labels < 0] = 0

    return {
        'pcd': scene_pcd,
        'root': combined_root,
        'labels': combined_labels,
        'object_labels': object_labels,
        'object_clouds': segmented_pcds,
        'per_object_results': per_object_results,
    }


def main():
    """Main function for multi-category scene generation."""
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Multi-Category Scene Generation (No Segmentation)')
    
    # Dataset parameters
    parser.add_argument('--parent_dir', type=str, default='YCBV', 
                       choices=['HANDAL', 'YCBV', 'YCBV-Partial', 'KITTI', 'ShapeNetPart'],
                       help='Dataset directory')
    parser.add_argument('--objects', type=str, nargs='+', default=None,
                       help='List of object categories to load')
    parser.add_argument('--num_objects', type=int, default=None,
                       help='Number of random objects to load (if not specified, uses --objects list)')
    
    # Scene parameters
    parser.add_argument('--scene_size', type=float, default=0.5,
                       help='Size of the scene (radius)')
    
    # Visualization parameters
    parser.add_argument('--no_individual', action='store_true',
                       help='Skip individual object visualization')
    
    parser.add_argument('--adversarial', action='store_true', help='use adversarial or not adversarial data')
    # ADDED: allow direct multi-object segmentation from a scene file
    parser.add_argument('--input_path', type=str, default=None, help='Direct path to a multi-object point cloud file')
    
    args = parser.parse_args()
    
    print("🎬 Multi-Category Scene Generation (No Segmentation)")
    print("=" * 60)
    print(f"📦 Dataset: {args.parent_dir}")
    
    # Determine object categories
    if args.num_objects is not None and args.objects is None:
        # Load random objects
        available_objects = get_available_objects_for_dataset(args.parent_dir) 
        
        if not available_objects:
            print(f"❌ No objects found for dataset {args.parent_dir}")
            return
        
        if args.num_objects > len(available_objects):
            print(f"⚠️ Requested {args.num_objects} objects, but only {len(available_objects)} available")
            args.num_objects = len(available_objects)
        
        object_categories = random.sample(available_objects, args.num_objects)
        print(f"🔢 Loading {args.num_objects} random objects: {object_categories}")
    else:
        # Use specified objects
        object_categories = args.objects
        print(f"🏷️  Loading specified objects: {object_categories}")
    
    print(f"📏 Scene size: {args.scene_size}")
    print("=" * 60)

    if args.input_path:
        print(f"📥 Segmenting custom multi-object scene: {args.input_path}")
        segmentation_results = segment_multi_object_point_cloud(args.input_path)
        root = segmentation_results['root']
        root.print_tree()
        print(f"✅ Produced {len(np.unique(segmentation_results['labels']))} part labels across {len(np.unique(segmentation_results['object_labels']))} objects")
        return
    
    # Create multi-category scene
    scene_pcd, scene_objects, segmentation_results = create_multi_category_scene(
        parent_dir=args.parent_dir,
        object_categories=object_categories,
        scene_size=args.scene_size,
        adversarial=False
    )
    
    if scene_pcd is None:
        print("❌ Failed to create scene!")
        return
    
    # Visualize results
    visualize_multi_category_scene(
        scene_pcd, scene_objects, segmentation_results,
        show_individual_objects=not args.no_individual
    )
    
    print("\n🎉 Multi-category scene generation completed!")
    

if __name__ == "__main__":
    main()
