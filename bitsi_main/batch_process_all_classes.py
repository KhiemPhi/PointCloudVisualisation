#!/usr/bin/env python3
"""
Batch processing script for all ShapeNetPart classes.
Processes all categories with progress bars and error handling with breakpoints.
"""

import os
import sys
import json
import argparse
import traceback
from tqdm import tqdm
from datetime import datetime

# Add the current directory to path to import from main.py
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from main import (
    get_category_mapping,
    get_available_categories,
    get_random_object,
    load_point_cloud_shapenetpart,
    build_cloud_object,
    bitsi_metric,
    build_segmentation_tree,
    fuse_consecutive_segments,
    apply_slicing_to_children,
    rename_segments,
    auto_thickness,
    calculate_iou_metrics,
    calculate_unsupervised_clustering,
    visualize_shapepart_comparison
)
import open3d as o3d
import numpy as np


def process_single_object(file_path, category_id, object_id, output_base_dir, 
                         gripper_width=0.13, gripper_height=0.07, epsilon=5e-3,
                         strength_threshold=0.01, ibr_tolerance=0.07, split='train'):
    """
    Process a single ShapeNetPart object.
    
    Parameters
    ----------
    file_path : str
        Path to the point cloud file
    category_id : str
        Category ID
    object_id : str
        Object ID (filename without extension)
    output_base_dir : str
        Base output directory
    gripper_width, gripper_height : float
        Gripper parameters
    epsilon : float
        Epsilon parameter for BITSI
    strength_threshold : float
        Threshold for inflection detection
    ibr_tolerance : float
        Tolerance for fusing consecutive segments
    split : str
        Dataset split ('train', 'val', 'test')
    
    Returns
    -------
    dict
        Results dictionary with metrics and status
    """
    try:
        # Load point cloud
        points, normals, part_ids = load_point_cloud_shapenetpart(file_path)
        
        if points is None or len(points) < 100:
            return {
                'success': False,
                'error': 'Failed to load or insufficient points',
                'file_path': file_path,
                'num_points': len(points) if points is not None else 0
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
        
        # Create output directory for this object
        output_dir = os.path.join(output_base_dir, category_id, object_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Perform segmentation
        root = build_segmentation_tree(
            points_per_slice_to_use, bitsi_x, bitsi_y, bitsi_z, 
            slice_idx, strength_threshold, output_dir=output_dir
        )
        root = fuse_consecutive_segments(root, ibr_tolerance=ibr_tolerance)
        #root = apply_slicing_to_children(root, bitsi_x, bitsi_y, bitsi_z, epsilon=epsilon, thickness=thickness)
        root = rename_segments(root)
        
        # Gather leaf segments
        def gather_leaves(node):
            leaves = []
            if not node.children:
                leaves.append(node)
            else:
                for child in node.children:
                    leaves += gather_leaves(child)
            return leaves
        
        segmented_leaves = gather_leaves(root)
        
        # Calculate IoU metrics
        alignment, mean_iou = calculate_iou_metrics(
            points, part_ids, segmented_leaves, output_dir=output_dir
        )
        
        # Calculate unsupervised clustering
        clusters, cluster_ious = calculate_unsupervised_clustering(
            points, part_ids, segmented_leaves, output_dir=output_dir
        )
        
        # Create comparison visualization
        visualize_shapepart_comparison(root, points, part_ids, 'ShapeNetPart', output_dir=output_dir)
        
        # Calculate statistics
        num_segments = len(segmented_leaves)
        num_gt_parts = len(np.unique(part_ids))
        good_matches = sum(1 for _, (_, iou) in alignment.items() if iou > 0.5) if alignment else 0
        
        return {
            'success': True,
            'file_path': file_path,
            'category_id': category_id,
            'object_id': object_id,
            'num_points': len(points),
            'num_gt_parts': num_gt_parts,
            'num_segments': num_segments,
            'mean_iou': mean_iou,
            'good_matches': good_matches,
            'output_dir': output_dir
        }
        
    except Exception as e:
        # Breakpoint on error for debugging
        import pdb
        print(f"\n❌ ERROR processing {file_path}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("\n🔴 Entering debugger (breakpoint)...")
        pdb.set_trace()
        
        return {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__,
            'file_path': file_path,
            'traceback': traceback.format_exc()
        }


def process_category(category_name, category_id, split='train', output_base_dir='batch_output',
                    gripper_width=0.13, gripper_height=0.07, epsilon=5e-3,
                    strength_threshold=0.01, ibr_tolerance=0.07, max_objects=None):
    """
    Process all objects in a category with progress bar.
    
    Parameters
    ----------
    category_name : str
        Category name (e.g., 'airplane')
    category_id : str
        Category ID (e.g., '02691156')
    split : str
        Dataset split to process
    output_base_dir : str
        Base output directory
    max_objects : int, optional
        Maximum number of objects to process (None for all)
    Other parameters same as process_single_object
    
    Returns
    -------
    dict
        Summary statistics for the category
    """
    # Get script directory and construct paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_path = os.path.join(project_root, 'data', 'ShapeNetPart')
    
    # Load split file list
    split_file = f'shuffled_{split}_file_list.json'
    split_file_path = os.path.join(dataset_path, 'train_test_split', split_file)
    
    if not os.path.exists(split_file_path):
        print(f"❌ Split file not found: {split_file_path}")
        return {'success': False, 'error': 'Split file not found'}
    
    with open(split_file_path, 'r') as f:
        file_list = json.load(f)
    
    # Filter files for this category
    category_files = [f for f in file_list if f'shape_data/{category_id}/' in f]
    
    if not category_files:
        print(f"⚠️ No files found for category {category_name} ({category_id}) in {split} set")
        return {'success': False, 'error': 'No files found'}
    
    # Limit number of objects if specified
    if max_objects:
        category_files = category_files[:max_objects]
    
    # Parse file paths and extract object IDs
    objects_to_process = []
    for file_path_json in category_files:
        parts = file_path_json.split('/')
        if len(parts) >= 3:
            object_id = parts[2]  # object_id without .txt extension
            full_file_path = os.path.join(dataset_path, category_id, f"{object_id}.txt")
            if os.path.exists(full_file_path):
                objects_to_process.append((full_file_path, object_id))
    
    if not objects_to_process:
        print(f"⚠️ No valid files found for category {category_name} ({category_id})")
        return {'success': False, 'error': 'No valid files'}
    
    print(f"\n📦 Processing category: {category_name} ({category_id})")
    print(f"   Found {len(objects_to_process)} objects in {split} set")
    
    # Process with progress bar
    results = []
    successful = 0
    failed = 0
    
    with tqdm(total=len(objects_to_process), desc=f"{category_name:12s}", 
              unit="obj", ncols=100, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
        for file_path, object_id in objects_to_process:
            result = process_single_object(
                file_path, category_id, object_id, output_base_dir,
                gripper_width, gripper_height, epsilon,
                strength_threshold, ibr_tolerance, split
            )
            results.append(result)
            
            if result['success']:
                successful += 1
                pbar.set_postfix({'✓': successful, '✗': failed, 'IoU': f"{result.get('mean_iou', 0):.3f}"})
            else:
                failed += 1
                pbar.set_postfix({'✓': successful, '✗': failed, 'err': result.get('error_type', 'Unknown')[:10]})
            
            pbar.update(1)
    
    # Calculate summary statistics
    successful_results = [r for r in results if r['success']]
    mean_ious = [r['mean_iou'] for r in successful_results if 'mean_iou' in r]
    avg_iou = np.mean(mean_ious) if mean_ious else 0.0
    
    summary = {
        'category_name': category_name,
        'category_id': category_id,
        'total_objects': len(objects_to_process),
        'successful': successful,
        'failed': failed,
        'success_rate': successful / len(objects_to_process) if objects_to_process else 0.0,
        'average_iou': avg_iou,
        'results': results
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Batch process all ShapeNetPart classes')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'],
                       help='Dataset split to process')
    parser.add_argument('--output_dir', type=str, default='/data/khiem/bitsi/batch_output',
                       help='Base output directory')
    parser.add_argument('--width', type=float, default=0.13, help='Gripper width')
    parser.add_argument('--height', type=float, default=0.07, help='Gripper height')
    parser.add_argument('--epsilon', type=float, default=5e-3, help='Epsilon parameter')
    parser.add_argument('--strength_threshold', type=float, default=0.01,
                       help='Strength threshold for inflection detection')
    parser.add_argument('--ibr_tolerance', type=float, default=0.07,
                       help='IBR tolerance for fusing segments')
    parser.add_argument('--max_objects_per_category', type=int, default=None,
                       help='Maximum objects to process per category (None for all)')
    parser.add_argument('--categories', type=str, nargs='+', default=None,
                       help='Specific categories to process (default: all)')
    parser.add_argument('--skip_categories', type=str, nargs='+', default=None,
                       help='Categories to skip')
    
    args = parser.parse_args()
    
    # Get all available categories
    category_map, reverse_map = get_category_mapping()
    available = get_available_categories()
    
    # Filter categories if specified
    categories_to_process = []
    if args.categories:
        # Process specified categories
        for cat_name in args.categories:
            cat_name_lower = cat_name.lower()
            if cat_name_lower in reverse_map:
                cat_id = reverse_map[cat_name_lower]
                categories_to_process.append((cat_name, cat_id))
            else:
                print(f"⚠️ Category '{cat_name}' not found, skipping")
    else:
        # Process all available categories
        for cat_name, cat_id, count in available:
            categories_to_process.append((cat_name, cat_id))
    
    # Remove skipped categories
    if args.skip_categories:
        skip_lower = [c.lower() for c in args.skip_categories]
        categories_to_process = [(name, cid) for name, cid in categories_to_process 
                                if name.lower() not in skip_lower]
    
    if not categories_to_process:
        print("❌ No categories to process")
        return
    
    print(f"\n🚀 Starting batch processing")
    print(f"   Split: {args.split}")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Categories to process: {len(categories_to_process)}")
    print(f"   Max objects per category: {args.max_objects_per_category or 'all'}")
    print("=" * 80)
    
    # Process each category
    all_summaries = []
    total_successful = 0
    total_failed = 0
    total_objects = 0
    
    for category_name, category_id in categories_to_process:
        summary = process_category(
            category_name, category_id, args.split, args.output_dir,
            args.width, args.height, args.epsilon,
            args.strength_threshold, args.ibr_tolerance,
            args.max_objects_per_category
        )
        
        all_summaries.append(summary)
        total_successful += summary.get('successful', 0)
        total_failed += summary.get('failed', 0)
        total_objects += summary.get('total_objects', 0)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("📊 FINAL SUMMARY")
    print("=" * 80)
    print(f"Total categories processed: {len(categories_to_process)}")
    print(f"Total objects processed: {total_objects}")
    print(f"Successful: {total_successful} ({100*total_successful/total_objects:.1f}%)" if total_objects > 0 else "Successful: 0")
    print(f"Failed: {total_failed} ({100*total_failed/total_objects:.1f}%)" if total_objects > 0 else "Failed: 0")
    
    # Calculate overall average IoU
    all_ious = []
    for summary in all_summaries:
        if summary.get('average_iou', 0) > 0:
            all_ious.append(summary['average_iou'])
    
    if all_ious:
        overall_avg_iou = np.mean(all_ious)
        print(f"Overall average IoU: {overall_avg_iou:.3f}")
    
    # Save summary to JSON
    summary_file = os.path.join(args.output_dir, f'summary_{args.split}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    os.makedirs(args.output_dir, exist_ok=True)
    
    summary_data = {
        'timestamp': datetime.now().isoformat(),
        'split': args.split,
        'parameters': {
            'gripper_width': args.width,
            'gripper_height': args.height,
            'epsilon': args.epsilon,
            'strength_threshold': args.strength_threshold,
            'ibr_tolerance': args.ibr_tolerance
        },
        'total_statistics': {
            'categories_processed': len(categories_to_process),
            'total_objects': total_objects,
            'successful': total_successful,
            'failed': total_failed,
            'overall_avg_iou': float(np.mean(all_ious)) if all_ious else 0.0
        },
        'category_summaries': all_summaries
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n💾 Summary saved to: {summary_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()

