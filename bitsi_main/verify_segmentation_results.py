#!/usr/bin/env python3
"""
Verification script for ShapeNetPart segmentation results.
Loads and visualizes random samples to verify the segmentation quality.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import random
from pathlib import Path

# Add the current directory to path to import from main.py
sys.path.append('/home/khiem/bitsi')
from main import load_point_cloud_shapenetpart, visualize_shapepart_comparison

def load_segmentation_results(results_dir, category_id=None, max_samples=5):
    """
    Load segmentation results and select random samples for verification.
    
    Parameters
    ----------
    results_dir : str
        Directory containing segmentation results
    category_id : str, optional
        Specific category to load (if None, loads from all categories)
    max_samples : int
        Maximum number of samples to load
        
    Returns
    -------
    list
        List of sample dictionaries with original and segmented data
    """
    samples = []
    
    if category_id:
        # Load from specific category
        category_dir = os.path.join(results_dir, category_id)
        if not os.path.exists(category_dir):
            print(f"Category directory not found: {category_dir}")
            return samples
        
        # Find all result files
        result_files = [f for f in os.listdir(category_dir) if f.endswith('_results.json')]
        selected_files = random.sample(result_files, min(max_samples, len(result_files)))
        
        for result_file in selected_files:
            result_path = os.path.join(category_dir, result_file)
            with open(result_path, 'r') as f:
                result_data = json.load(f)
            
            # Load original point cloud
            original_file = result_data['file_path']
            points, normals, part_ids = load_point_cloud_shapenetpart(original_file)
            
            if points is not None:
                samples.append({
                    'category_id': category_id,
                    'file_name': result_data['file_name'],
                    'points': points,
                    'normals': normals,
                    'part_ids': part_ids,
                    'segmented_part_ids': np.array(result_data['segmented_part_ids']),
                    'mean_iou': result_data.get('mean_iou', 0.0),
                    'num_gt_parts': result_data.get('num_gt_parts', 0),
                    'num_segmented_parts': result_data.get('num_segmented_parts', 0)
                })
    else:
        # Load from all categories
        for category_dir in os.listdir(results_dir):
            category_path = os.path.join(results_dir, category_dir)
            if os.path.isdir(category_path):
                category_samples = load_segmentation_results(results_dir, category_dir, max_samples//len(os.listdir(results_dir)))
                samples.extend(category_samples)
    
    return samples

def visualize_sample_comparison(sample, save_path=None):
    """
    Visualize a single sample comparison between ground truth and segmented parts.
    
    Parameters
    ----------
    sample : dict
        Sample data containing points, part_ids, and segmented_part_ids
    save_path : str, optional
        Path to save the visualization
    """
    points = sample['points']
    part_ids = sample['part_ids']
    segmented_part_ids = sample['segmented_part_ids']
    
    # Create figure with side-by-side subplots
    fig = plt.figure(figsize=(16, 8))
    
    # Left subplot: Ground truth parts
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_title(f"Ground Truth Parts\n{sample['category_id']}/{sample['file_name']}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    
    # Right subplot: Segmented parts
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_title(f"Segmented Parts (IoU: {sample['mean_iou']:.3f})")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    
    # Plot ground truth parts
    if part_ids is not None:
        unique_parts = np.unique(part_ids)
        gt_colors = plt.cm.get_cmap("tab20", len(unique_parts))
        
        for i, part_id in enumerate(unique_parts):
            mask = part_ids == part_id
            part_points = points[mask]
            if len(part_points) > 0:
                color = gt_colors(i)[:3]
                ax1.scatter(part_points[:, 0], part_points[:, 1], part_points[:, 2], 
                           s=5, color=color, alpha=0.8, label=f"Part {int(part_id)}")
    
    # Plot segmented parts
    unique_seg_parts = np.unique(segmented_part_ids)
    seg_colors = plt.cm.get_cmap("tab20", len(unique_seg_parts))
    
    for i, part_id in enumerate(unique_seg_parts):
        if part_id == 0:  # Skip unassigned points
            continue
        mask = segmented_part_ids == part_id
        part_points = points[mask]
        if len(part_points) > 0:
            color = seg_colors(i)[:3]
            ax2.scatter(part_points[:, 0], part_points[:, 1], part_points[:, 2], 
                       s=5, color=color, alpha=0.8, label=f"Seg {int(part_id)}")
    
    # Fix scaling for both subplots
    all_pts = points
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0
    mid_x = (all_pts[:, 0].max() + all_pts[:, 0].min()) / 2.0
    mid_y = (all_pts[:, 1].max() + all_pts[:, 1].min()) / 2.0
    mid_z = (all_pts[:, 2].max() + all_pts[:, 2].min()) / 2.0
    
    for ax in [ax1, ax2]:
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        ax.legend(loc="upper right", fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Visualization saved to: {save_path}")
    
    plt.show()

def create_summary_visualization(samples, save_path=None):
    """
    Create a summary visualization showing multiple samples in a grid.
    
    Parameters
    ----------
    samples : list
        List of sample dictionaries
    save_path : str, optional
        Path to save the visualization
    """
    n_samples = len(samples)
    n_cols = 3
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows), 
                            subplot_kw={'projection': '3d'})
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, sample in enumerate(samples):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        points = sample['points']
        segmented_part_ids = sample['segmented_part_ids']
        
        # Plot segmented parts
        unique_seg_parts = np.unique(segmented_part_ids)
        seg_colors = plt.cm.get_cmap("tab20", len(unique_seg_parts))
        
        for j, part_id in enumerate(unique_seg_parts):
            if part_id == 0:  # Skip unassigned points
                continue
            mask = segmented_part_ids == part_id
            part_points = points[mask]
            if len(part_points) > 0:
                color = seg_colors(j)[:3]
                ax.scatter(part_points[:, 0], part_points[:, 1], part_points[:, 2], 
                           s=3, color=color, alpha=0.8)
        
        # Fix scaling
        max_range = (points.max(axis=0) - points.min(axis=0)).max() / 2.0
        mid_x = (points[:, 0].max() + points[:, 0].min()) / 2.0
        mid_y = (points[:, 1].max() + points[:, 1].min()) / 2.0
        mid_z = (points[:, 2].max() + points[:, 2].min()) / 2.0
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.set_title(f"{sample['category_id']}\nIoU: {sample['mean_iou']:.3f}\n"
                    f"GT: {sample['num_gt_parts']} | Seg: {sample['num_segmented_parts']}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
    
    # Hide empty subplots
    for i in range(n_samples, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"💾 Summary visualization saved to: {save_path}")
    
    plt.show()

def main():
    """Main function to verify segmentation results."""
    
    # Configuration
    results_dir = "/home/khiem/bitsi/shapenetpart_segmented_results"
    output_dir = "/home/khiem/bitsi/verification_plots"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("🔍 Loading segmentation results for verification...")
    
    # Load random samples from all categories
    samples = load_segmentation_results(results_dir, max_samples=15)
    
    if not samples:
        print("❌ No samples found. Make sure the batch processing has been completed.")
        return
    
    print(f"📊 Loaded {len(samples)} samples for verification")
    
    # Print sample statistics
    mean_ious = [s['mean_iou'] for s in samples if s['mean_iou'] > 0]
    if mean_ious:
        print(f"📈 Mean IoU: {np.mean(mean_ious):.3f} ± {np.std(mean_ious):.3f}")
        print(f"🎯 Best IoU: {np.max(mean_ious):.3f}")
        print(f"📉 Worst IoU: {np.min(mean_ious):.3f}")
    
    # Create detailed comparison for a few samples
    print("\n🔍 Creating detailed comparisons...")
    for i, sample in enumerate(samples[:3]):  # Show first 3 samples in detail
        print(f"\n📊 Sample {i+1}: {sample['category_id']}/{sample['file_name']}")
        print(f"   Points: {len(sample['points'])}")
        print(f"   GT Parts: {sample['num_gt_parts']}")
        print(f"   Segmented Parts: {sample['num_segmented_parts']}")
        print(f"   IoU: {sample['mean_iou']:.3f}")
        
        save_path = os.path.join(output_dir, f"detailed_comparison_{i+1}.png")
        visualize_sample_comparison(sample, save_path)
    
    # Create summary visualization
    print("\n📊 Creating summary visualization...")
    summary_path = os.path.join(output_dir, "summary_visualization.png")
    create_summary_visualization(samples, summary_path)
    
    # Print category-wise statistics
    print("\n📊 Category-wise Statistics:")
    categories = {}
    for sample in samples:
        cat = sample['category_id']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(sample['mean_iou'])
    
    for cat, ious in categories.items():
        valid_ious = [iou for iou in ious if iou > 0]
        if valid_ious:
            print(f"  {cat}: {np.mean(valid_ious):.3f} ± {np.std(valid_ious):.3f} ({len(valid_ious)} samples)")
    
    print(f"\n💾 Verification plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
