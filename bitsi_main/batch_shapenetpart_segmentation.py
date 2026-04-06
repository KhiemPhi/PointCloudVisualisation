#!/usr/bin/env python3
"""
Batch processing script for ShapeNetPart dataset segmentation.
Processes all point clouds in the ShapeNetPart dataset, performs BITSI segmentation,
and saves the segmented part IDs in the same format as the original part_ids.
"""

import os
import sys
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import json
from datetime import datetime

# Add the current directory to path to import from main.py
sys.path.append('/home/khiem/bitsi')
from main import (
    load_point_cloud_shapenetpart, get_category_mapping, get_available_categories,
    build_cloud_object, bitsi_metric, build_segmentation_tree, fuse_consecutive_segments,
    SegmentNode, auto_thickness, visualize_shapepart_comparison, calculate_iou_metrics
)

def process_single_pointcloud(file_path, category_id, file_name, output_dir, 
                             gripper_width=0.13, gripper_height=0.07, epsilon=5e-3,
                             strength_threshold=0.01, ibr_tolerance=0.02, visualize=False):
    """
    Process a single ShapeNetPart point cloud file.
    
    Parameters
    ----------
    file_path : str
        Path to the point cloud file
    category_id : str
        Category ID of the object
    file_name : str
        Name of the file
    output_dir : str
        Directory to save results
    gripper_width, gripper_height : float
        Gripper parameters
    epsilon : float
        Epsilon parameter for BITSI
    strength_threshold : float
        Threshold for inflection detection
    ibr_tolerance : float
        Tolerance for fusing consecutive segments
    visualize : bool
        Whether to create visualization plots
        
    Returns
    -------
    dict
        Results dictionary with segmentation info and metrics
    """
    try:
        # Load point cloud
        points, normals, part_ids = load_point_cloud_shapenetpart(file_path)
        
        if points is None or len(points) < 100:
            return {
                'success': False,
                'error': 'Failed to load or insufficient points',
                'file_path': file_path
            }
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.paint_uniform_color([0, 0, 1])
        
        # Build cloud object and compute BITSI metrics
        cloud_object = build_cloud_object(pcd, gripper_width, gripper_height)
        thickness = auto_thickness(cloud_object, scale=0.03)
        
        bitsi_x, bitsi_y, bitsi_z, slice_idx, points_per_slice_x, points_per_slice_y, points_per_slice_z = bitsi_metric(
            cloud_object, epsilon=epsilon, thickness=thickness
        )
        
        points_per_slice = [points_per_slice_x, points_per_slice_y, points_per_slice_z]
        points_per_slice_to_use = points_per_slice[slice_idx]
        
        # Perform segmentation
        root = build_segmentation_tree(points_per_slice_to_use, bitsi_x, bitsi_y, bitsi_z, slice_idx, strength_threshold)
        root = fuse_consecutive_segments(root, ibr_tolerance=ibr_tolerance)
        
        # Get leaf segments (final segmented parts)
        def gather_leaves(node):
            leaves = []
            if not node.children:
                leaves.append(node)
            else:
                for child in node.children:
                    leaves += gather_leaves(child)
            return leaves
        
        segmented_leaves = gather_leaves(root)
        
        # Create segmented part IDs (same format as original part_ids)
        segmented_part_ids = np.zeros(len(points), dtype=int)
        
        # Map each point to its segmented part
        for i, leaf in enumerate(segmented_leaves):
            if len(leaf.points) == 0:
                continue
                
            # Find closest points in original point cloud for each segmented point
            for seg_point in leaf.points:
                distances = np.linalg.norm(points - seg_point, axis=1)
                closest_idx = np.argmin(distances)
                segmented_part_ids[closest_idx] = i + 1  # Start from 1, not 0
        
        # Calculate IoU metrics if ground truth is available
        iou_results = None
        if part_ids is not None:
            try:
                iou_results, mean_iou = calculate_iou_metrics(points, part_ids, segmented_leaves)
            except Exception as e:
                print(f"Warning: IoU calculation failed for {file_name}: {e}")
                iou_results = None
                mean_iou = 0.0
        else:
            mean_iou = 0.0
        
        # Save results
        result_data = {
            'file_path': file_path,
            'category_id': category_id,
            'file_name': file_name,
            'num_points': len(points),
            'num_gt_parts': len(np.unique(part_ids)) if part_ids is not None else 0,
            'num_segmented_parts': len(segmented_leaves),
            'segmented_part_ids': segmented_part_ids,
            'mean_iou': mean_iou,
            'slice_idx': slice_idx,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save segmented part IDs
        output_file = os.path.join(output_dir, f"{category_id}_{file_name.replace('.txt', '_segmented.txt')}")
        np.savetxt(output_file, segmented_part_ids, fmt='%d')
        
        # Save detailed results
        results_file = os.path.join(output_dir, f"{category_id}_{file_name.replace('.txt', '_results.json')}")
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_data = result_data.copy()
            json_data['segmented_part_ids'] = segmented_part_ids.tolist()
            json.dump(json_data, f, indent=2)
        
        # Create visualization if requested
        if visualize and part_ids is not None:
            try:
                visualize_shapepart_comparison(root, points, part_ids, 'ShapeNetPart')
                plt.savefig(os.path.join(output_dir, f"{category_id}_{file_name.replace('.txt', '_comparison.png')}"))
                plt.close()
            except Exception as e:
                print(f"Warning: Visualization failed for {file_name}: {e}")
        
        return {
            'success': True,
            'file_path': file_path,
            'num_segmented_parts': len(segmented_leaves),
            'mean_iou': mean_iou,
            'output_file': output_file
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'file_path': file_path
        }

def process_category(category_id, category_name, dataset_path, output_dir, 
                    max_files=None, visualize_samples=5, **kwargs):
    """
    Process all files in a category.
    
    Parameters
    ----------
    category_id : str
        Category ID to process
    category_name : str
        Category name
    dataset_path : str
        Path to ShapeNetPart dataset
    output_dir : str
        Output directory
    max_files : int, optional
        Maximum number of files to process (for testing)
    visualize_samples : int
        Number of random samples to visualize
    **kwargs
        Additional parameters for processing
    """
    category_path = os.path.join(dataset_path, category_id)
    if not os.path.exists(category_path):
        print(f"Category directory not found: {category_path}")
        return []
    
    files = [f for f in os.listdir(category_path) if f.endswith('.txt')]
    if not files:
        print(f"No .txt files found in {category_path}")
        return []
    
    if max_files:
        files = files[:max_files]
    
    print(f"\n🔄 Processing {category_name} ({category_id}): {len(files)} files")
    
    # Create category output directory
    category_output_dir = os.path.join(output_dir, category_id)
    os.makedirs(category_output_dir, exist_ok=True)
    
    results = []
    successful = 0
    failed = 0
    mean_ious = []
    
    # Select random samples for visualization
    visualize_files = np.random.choice(files, min(visualize_samples, len(files)), replace=False)
    
    for i, file_name in enumerate(tqdm(files, desc=f"Processing {category_name}")):
        file_path = os.path.join(category_path, file_name)
        
        # Determine if this file should be visualized
        should_visualize = file_name in visualize_files
        
        result = process_single_pointcloud(
            file_path, category_id, file_name, category_output_dir,
            visualize=should_visualize, **kwargs
        )
        
        results.append(result)
        
        if result['success']:
            successful += 1
            if 'mean_iou' in result and result['mean_iou'] > 0:
                mean_ious.append(result['mean_iou'])
        else:
            failed += 1
            print(f"❌ Failed: {file_name} - {result.get('error', 'Unknown error')}")
    
    # Print category summary
    print(f"\n📊 {category_name} Summary:")
    print(f"  ✅ Successful: {successful}/{len(files)}")
    print(f"  ❌ Failed: {failed}/{len(files)}")
    if mean_ious:
        print(f"  📈 Mean IoU: {np.mean(mean_ious):.3f} ± {np.std(mean_ious):.3f}")
        print(f"  🎯 Best IoU: {np.max(mean_ious):.3f}")
        print(f"  📉 Worst IoU: {np.min(mean_ious):.3f}")
    
    return results

def main():
    """Main function to process all ShapeNetPart categories."""
    
    # Configuration
    dataset_path = "/home/khiem/data/ShapeNetPart"
    output_dir = "/home/khiem/bitsi/shapenetpart_segmented_results"
    
    # Processing parameters
    processing_params = {
        'gripper_width': 0.13,
        'gripper_height': 0.07,
        'epsilon': 5e-3,
        'strength_threshold': 0.01,
        'ibr_tolerance': 0.02
    }
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get available categories
    available_categories = get_available_categories()
    print(f"📦 Found {len(available_categories)} categories in ShapeNetPart dataset")
    
    # Process each category
    all_results = {}
    overall_stats = {
        'total_files': 0,
        'successful_files': 0,
        'failed_files': 0,
        'all_mean_ious': []
    }
    
    for category_name, category_id, file_count in available_categories:
        print(f"\n{'='*60}")
        print(f"Processing {category_name} ({category_id}) - {file_count} files")
        print(f"{'='*60}")
        
        # Process category (limit to first 10 files for testing)
        category_results = process_category(
            category_id, category_name, dataset_path, output_dir,
            max_files=10,  # Remove this limit for full processing
            visualize_samples=3,
            **processing_params
        )
        
        all_results[category_id] = category_results
        
        # Update overall stats
        for result in category_results:
            overall_stats['total_files'] += 1
            if result['success']:
                overall_stats['successful_files'] += 1
                if 'mean_iou' in result and result['mean_iou'] > 0:
                    overall_stats['all_mean_ious'].append(result['mean_iou'])
            else:
                overall_stats['failed_files'] += 1
    
    # Print overall summary
    print(f"\n{'='*60}")
    print("🎉 BATCH PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"📊 Overall Statistics:")
    print(f"  📁 Total files processed: {overall_stats['total_files']}")
    print(f"  ✅ Successful: {overall_stats['successful_files']}")
    print(f"  ❌ Failed: {overall_stats['failed_files']}")
    print(f"  📈 Success rate: {overall_stats['successful_files']/overall_stats['total_files']*100:.1f}%")
    
    if overall_stats['all_mean_ious']:
        print(f"  📊 Overall Mean IoU: {np.mean(overall_stats['all_mean_ious']):.3f} ± {np.std(overall_stats['all_mean_ious']):.3f}")
        print(f"  🎯 Best IoU: {np.max(overall_stats['all_mean_ious']):.3f}")
        print(f"  📉 Worst IoU: {np.min(overall_stats['all_mean_ious']):.3f}")
    
    # Save overall results
    results_file = os.path.join(output_dir, "batch_processing_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'overall_stats': overall_stats,
            'category_results': {k: [r for r in v if r['success']] for k, v in all_results.items()},
            'processing_params': processing_params,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_dir}")
    print(f"📄 Summary saved to: {results_file}")

if __name__ == "__main__":
    main()
