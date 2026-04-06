import argparse
import open3d as o3d 
import sys
import os
import numpy as np 
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless servers
import matplotlib.pyplot as plt
from bitsi_slicer.bitsi_slicer import slice_object_bbox, bitsi_metric, build_cloud_object, get_overall_ibr_ratio
from numpy.polynomial import Chebyshev
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from numpy.polynomial import chebyshev as C
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
from scipy.cluster.hierarchy import linkage, fcluster

import random
import json
from itertools import combinations



def fit_best_chebyshev(x, y, min_deg=3, max_deg=None, criterion="bic", strength_threshold=0.2):
    """
    Fit Chebyshev polynomials of increasing degree and pick the best one
    based on the chosen information criterion ('bic', 'aic', 'aicc', or 'adj_r2').

    Returns:
        best_coeffs, best_degree, best_yfit, best_metrics
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        raise ValueError("Need at least 3 points for fitting.")

    if max_deg is None:
        max_deg = min(8, n - 2)
    max_deg = max(max_deg, min_deg)

    # scale x → [-1, 1] for Chebyshev stability
    xmin, xmax = np.min(x), np.max(x)
    if xmax == xmin:
        x_scaled = np.zeros_like(x)
    else:
        x_scaled = 2 * (x - xmin) / (xmax - xmin) - 1.0

    def metrics(y_true, y_pred, k):
        rss = np.sum((y_true - y_pred) ** 2)
        tss = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-12
        r2 = 1 - rss / tss
        adj_r2 = 1 - (1 - r2) * (n - 1) / max(n - k, 1)
        sigma2 = rss / max(n, 1)
        eps = 1e-18
        aic = n * np.log(sigma2 + eps) + 2 * k
        aicc = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
        bic = n * np.log(sigma2 + eps) + k * np.log(max(n, 1))
        return dict(r2=r2, adj_r2=adj_r2, aic=aic, aicc=aicc, bic=bic, rss=rss)

    best = None
    all_models = []

    for deg in range(min_deg, max_deg + 1):
        try:
            ch = Chebyshev.fit(x_scaled, y, deg)
            y_fit = ch(x_scaled)
            m = metrics(y, y_fit, k=deg + 1)
            m["degree"] = deg
            m["model"] = ch
            m["y_fit"] = y_fit
            all_models.append(m)
        except np.linalg.LinAlgError:
            continue

    if not all_models:
        raise RuntimeError("Chebyshev fitting failed for all degrees.")

    if criterion.lower() in ("bic", "aic", "aicc"):
        key = criterion.lower()
        best = min(all_models, key=lambda m: m[key])
    elif criterion.lower() == "adj_r2":
        best = max(all_models, key=lambda m: m["adj_r2"])
    else:
        raise ValueError("criterion must be one of 'bic','aic','aicc','adj_r2'")

    best_ch = best["model"]
    coeffs = best_ch.coef
    degree = best["degree"]
    y_fit = best["y_fit"]
    

    
     # --- Compute inflection points directly on given x
    dcoeffs = C.chebder(coeffs)
    ddcoeffs = C.chebder(dcoeffs)
    ddy = C.chebval(x_scaled, ddcoeffs)

        # Sign change in second derivative → inflection point
    sign_change = np.diff(np.sign(ddy))
    inflection_idx = np.where(sign_change != 0)[0]
    inflections = np.int32(inflection_idx)

    # If no inflections found, try best model from another criterion
    if len(inflections) == 0:
        tried_criteria = [criterion.lower()]
        possible_criteria = ["bic", "aic", "aicc", "adj_r2"]
        for alt_criterion in possible_criteria:
            if alt_criterion not in tried_criteria and any(m for m in all_models):
                if alt_criterion in ("bic", "aic", "aicc"):
                    best_alt = min(all_models, key=lambda m: m[alt_criterion])
                elif alt_criterion == "adj_r2":
                    best_alt = max(all_models, key=lambda m: m["adj_r2"])
                else:
                    continue
                alt_ch = best_alt["model"]
                alt_coeffs = alt_ch.coef
                alt_degree = best_alt["degree"]
                alt_y_fit = best_alt["y_fit"]
                # --- Compute inflection points directly on given x for alt model
                d_alt_coeffs = C.chebder(alt_coeffs)
                dd_alt_coeffs = C.chebder(d_alt_coeffs)
                ddy_alt = C.chebval(x_scaled, dd_alt_coeffs)
                sign_change_alt = np.diff(np.sign(ddy_alt))
                inflection_idx_alt = np.where(sign_change_alt != 0)[0]
                inflections = np.int32(inflection_idx_alt)
                if len(inflections) > 0:
                    coeffs = alt_coeffs
                    degree = alt_degree
                    y_fit = alt_y_fit
                    best = best_alt
                    ddy = ddy_alt
                    inflection_idx = inflection_idx_alt
                    break

    # --- Compute strength of each inflection ---
    strengths = []
    for i in inflection_idx:
        # measure magnitude of curvature change around the inflection
        left = np.abs(ddy[i - 1]) if i > 0 else 0
        right = np.abs(ddy[i + 1]) if i < len(ddy) - 1 else 0
        strength = np.abs(right - left)
        strengths.append(strength)

    # normalize (optional, for easier comparison)
    strengths = np.array(strengths)
    if len(strengths) > 0:
        strengths = strengths / np.max(strengths)
    
    
    strong_mask = strengths >= strength_threshold

    inflections = inflections[strong_mask]
    strengths = strengths[strong_mask]
   
    # Print results
    print(f"📍 Found {len(inflections)} inflection point(s) with strength threshold {strength_threshold}:")
    for idx, s in zip(inflections, strengths):
        print(f"   ↳ slice {idx:3d} | strength = {s:.3f}")
    
    return coeffs, degree, y_fit, best, inflections

def visualize_tree_segments(root_node, output_dir=None, filename=None):
    """
    Visualize all child segments of a tree (excluding the root node).

    Parameters
    ----------
    root_node : SegmentNode
        The root node of the segmentation tree.
    output_dir : str, optional
        Directory to save the visualization. If None, displays the plot.
    filename : str, optional
        Name of the file to save the visualization. If None, the visualization is not saved.
    """
    # Gather all children recursively, excluding root
    def gather_children(node):
        nodes = []
        for c in node.children:
            nodes.append(c)
            nodes += gather_children(c)
        return nodes

    children = gather_children(root_node)
    if not children:
        print("⚠️ No child segments to visualize.")
        return

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D Segmented Parts (excluding root)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    cmap = plt.cm.get_cmap("tab20", len(children))

    for i, node in enumerate(children):
        if len(node.points) == 0:
            continue
        pts = np.asarray(node.points)
        color = cmap(i)[:3]
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=5, color=color, alpha=0.8, label=node.name)

    # --- Fix scaling (make axes equal) ---
    all_pts = np.vstack([np.asarray(n.points) for n in children if len(n.points) > 0])
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0
    mid_x = (all_pts[:, 0].max() + all_pts[:, 0].min()) / 2.0
    mid_y = (all_pts[:, 1].max() + all_pts[:, 1].min()) / 2.0
    mid_z = (all_pts[:, 2].max() + all_pts[:, 2].min()) / 2.0
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved visualization to {output_path}")
        plt.close()
    else:
        plt.show()
    
def bitsi_with_extrema(bitsi_x, bitsi_y, bitsi_z, dominant_idx, visualize=True, strength_threshold=0.2, max_deg=10, output_dir=None):
    # Axis conventions
    axis_names = ["X", "Y", "Z"]
    axis_colors = {"X": "r", "Y": "g", "Z": "b"}
    all_bitsi = [bitsi_x, bitsi_y, bitsi_z]


    dominant_axis = axis_names[dominant_idx]
    y = np.array(all_bitsi[dominant_idx], dtype=float)
    x = np.arange(len(y))
    # --- Fit best Chebyshev polynomial
    coeffs, degree, y_fit, metrics, inflections = fit_best_chebyshev(x, y, min_deg=2, max_deg=max_deg, criterion="bic", strength_threshold=strength_threshold)
    print(f"🎯 Chosen Chebyshev degree {degree}, adj R²={metrics['adj_r2']:.3f}, BIC={metrics['bic']:.2f}")


    # --- Plot
    if visualize:
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, "o", color=axis_colors[dominant_axis], label=f"BITSI ({dominant_axis}-axis)")
        plt.plot(x, y_fit, "--", color="k", alpha=0.7, label=f"Chebyshev deg {degree} fit")

        for i in inflections:
            plt.scatter(i, y_fit[i], color="orange", s=60, zorder=3)
            plt.text(i, y_fit[i] + 0.05, f"inflection (slice {i})", ha="center", fontsize=10, color="orange")

        plt.title(f"BITSI Curve Along Dominant Axis ({dominant_axis}-axis)")
        plt.xlabel("Slice Index")
        plt.ylabel("BITSI Value")
        plt.legend()
        plt.grid(True)
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "bitsi_curve.png")
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"💾 Saved BITSI curve to {output_path}")
            plt.close()
        else:
            plt.show()
    
    return x, y_fit, inflections


def pretty_print_bitsi(bitsi_x, bitsi_y, bitsi_z, slice_idx):
    axis_names = ["X", "Y", "Z"]
    axis_colors = {"X": "\033[91m", "Y": "\033[92m", "Z": "\033[94m"}  # red, green, blue
    reset = "\033[0m"

    # Compute mean and variance IBR values for each axis
    mean_vals = [np.mean(bitsi_x), np.mean(bitsi_y), np.mean(bitsi_z)]
    var_vals = [np.var(bitsi_x), np.var(bitsi_y), np.var(bitsi_z)]

    print("\n✅ BITSI Metrics Summary:")
    print("-" * 40)
    for i, (axis, mean_val, var_val) in enumerate(zip(axis_names, mean_vals, var_vals)):
        color = axis_colors[axis]
        print(f"{color}{axis}-axis:{reset} mean IBR = {mean_val:.3f}, variance = {var_val:.3f}")

    # Determine the dominant (smallest mean) axis
    dominant_axis_idx = int(np.argmin(mean_vals))
    dominant_axis = axis_names[dominant_axis_idx]

    print("-" * 40)
    print(f"🧭 Dominant slicing direction: {axis_colors[dominant_axis]}{dominant_axis}-axis{reset}")
    print(f"   (largest mean IBR = {mean_vals[dominant_axis_idx]:.3f})")

def auto_thickness(cloud_object, scale=0.01):
    bbox_min = np.min(cloud_object.points, axis=0)
    bbox_max = np.max(cloud_object.points, axis=0)
    graspable_dim = np.array([cloud_object.x_dim, cloud_object.y_dim, cloud_object.z_dim])
    search_idx = np.argmax(graspable_dim)   
    
    return scale * graspable_dim[search_idx]

def adjust_handal(pcd): 
    points_array = np.asarray(pcd.points)
    points_array /= 1000.0 
    pcd.points = o3d.utility.Vector3dVector(points_array)
    pcd.paint_uniform_color([0, 0, 1])
    return pcd

def adjust_kitti(pcd): 
    points_array = np.asarray(pcd.points)
    points_array /= 1000.0 
    pcd.points = o3d.utility.Vector3dVector(points_array)
    pcd.paint_uniform_color([0, 0, 1])
    return pcd

class SegmentNode:
    """Tree node representing a segmented region of the point cloud."""
    def __init__(self, name, points, slice_idx, parent=None, gripper_width=0.13, gripper_height=0.07, epsilon=5e-3):
        self.name = name
        self.points = points
        self.parent = parent
        self.children = []
        self.gripper_width = gripper_width
        self.gripper_height = gripper_height
        self.slice_idx = slice_idx
        # Create an Open3D point cloud for this segment
        self.pcd = o3d.geometry.PointCloud()
        if len(points) > 0:
            self.pcd.points = o3d.utility.Vector3dVector(points)
            # Optional: random color per node
            color = np.random.rand(3)
            self.pcd.paint_uniform_color(color)
        else:
            self.pcd.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
        
        self.cloud_object = build_cloud_object(self.pcd, gripper_width, gripper_height)
        thickness = auto_thickness(self.cloud_object, scale=0.03)         
        self.ibr_ratio = get_overall_ibr_ratio(self.cloud_object, epsilon=epsilon, thickness=thickness)
    
    def detect_and_break(self, eps=0.02, min_points=50, curvature_threshold=0.03):
        """
        Drop-in method to detect if objects are touching and split them.
        1. Tries simple distance clustering.
        2. If it's one blob, it removes high-curvature 'seams' to find hidden splits.
        """
        # --- STEP 1: Fast Distance-Based Split ---
        # Check if they are already physically separated
        labels = np.array(self.pcd.cluster_dbscan(eps=eps, min_points=min_points))
        
        if labels.max() > 0:
            return self._spawn_children(labels)

        # --- STEP 2: Touching Object Detection (Curvature/Normal Analysis) ---
        # If we are here, it's one 'blob'. We look for sharp 'valleys' between objects.
        if not self.pcd.has_normals():
            self.pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=eps*2, max_nn=30)
            )
        
        # Compute the curvature (variation of normals) for each point
        # In Open3D, we can estimate this by looking at how much normals deviate in a neighborhood
        kdtree = o3d.geometry.KDTreeFlann(self.pcd)
        curvatures = []
        
        points = np.asarray(self.pcd.points)
        normals = np.asarray(self.pcd.normals)
        
        for i in range(len(points)):
            [_, idx, _] = kdtree.search_knn_vector_3d(points[i], 20)
            # Variance of normals in the neighborhood serves as a speed-proxy for curvature
            neighbor_normals = normals[idx, :]
            mean_normal = np.mean(neighbor_normals, axis=0)
            curvature = 1 - np.linalg.norm(mean_normal) # High value = sharp edge/seam
            curvatures.append(curvature)
        
        curvatures = np.array(curvatures)

        # --- STEP 3: The "Break" (Removing the Seams) ---
        # Identify 'smooth' points (surface centers) and 'sharp' points (the touching seams)
        smooth_indices = np.where(curvatures < curvature_threshold)[0]
        
        if len(smooth_indices) < min_points:
            return [self] # Too small to break safely

        # Create a temporary 'cut' cloud without the edges
        smooth_pcd = self.pcd.select_by_index(smooth_indices)
        
        # Cluster the smooth parts
        sub_labels = np.array(smooth_pcd.cluster_dbscan(eps=eps*1.5, min_points=min_points))
        
        # If the smooth areas form multiple clusters, the objects are 'touching'
        if sub_labels.max() > 0:
            # We project the labels back to the original full cloud for a clean split
            full_labels = np.full(len(self.points), -1)
            full_labels[smooth_indices] = sub_labels
            
            # Simple nearest-neighbor to fix the 'holes' we cut out
            return self._spawn_children(full_labels)

        return [self]

    def _spawn_children(self, labels):
        """Internal helper to instantiate child nodes from labels."""
        new_segments = []
        for i in range(labels.max() + 1):
            indices = np.where(labels == i)[0]
            if len(indices) < 10: continue # Final noise check
            
            child = self.__class__(
                name=f"{self.name}_part_{i}",
                points=self.points[indices],
                slice_idx=self.slice_idx,
                parent=self,
                gripper_width=self.gripper_width,
                gripper_height=self.gripper_height
            )
            new_segments.append(child)
        
        self.children = new_segments
        return new_segments
        
    
    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f"SegmentNode(name={self.name}, points={len(self.points)}, children={len(self.children)})"

    def print_tree(self, level=0):
        indent = "  " * level
        parent_idx = self.parent.slice_idx if self.parent is not None else None
        print(f"{indent}📦 {self.name}: {len(self.points)} points (slice_idx={self.slice_idx}, parent={parent_idx}, ibr_ratio={self.ibr_ratio})")
        for child in self.children:
            child.print_tree(level + 1)
    
    

def fuse_consecutive_segments(node, ibr_tolerance=0.2):
    """
    Merge only consecutive child segments with similar IBR ratios.

    Parameters
    ----------
    node : SegmentNode
        Parent node whose consecutive children will be checked for fusion.
    ibr_tolerance : float
        Maximum allowed absolute difference in IBR ratio for merging.

    Returns
    -------
    node : SegmentNode
        The same node with updated children (merged if necessary).
    """
    if not node.children or len(node.children) < 2:
        return node

    fused_children = []
    i = 0
    while i < len(node.children):
        current = node.children[i]

        # Check next segment if exists
        if i + 1 < len(node.children):
            nxt = node.children[i + 1]
            diff = abs(current.ibr_ratio - nxt.ibr_ratio)
            print(diff)

            if diff < ibr_tolerance:
                # Merge these two consecutive children
                merged_points = np.vstack([current.points, nxt.points])
                merged_name = f"{current.name}_{nxt.name}"
                merged_ibr = (current.ibr_ratio + nxt.ibr_ratio) / 2.0

                merged_child = SegmentNode(
                    name=merged_name,
                    points=merged_points,
                    slice_idx=current.slice_idx,
                    parent=node,
                    gripper_width=current.gripper_width,
                    gripper_height=current.gripper_height,
                )
                merged_child.ibr_ratio = merged_ibr
                fused_children.append(merged_child)

                # skip next since it’s merged
                i += 2
                continue

        # otherwise, keep current as-is
        fused_children.append(current)
        i += 1

    node.children = fused_children
    return node

def build_segmentation_tree(points_per_slice_to_use, bitsi_x, bitsi_y, bitsi_z, slice_idx, strength_threshold, max_deg=12, visualize=False, output_dir=None):
    x, y_fit, inflections = bitsi_with_extrema(bitsi_x, bitsi_y, bitsi_z, slice_idx, visualize=visualize, strength_threshold=strength_threshold, max_deg=max_deg, output_dir=output_dir)
    x_min, x_max = np.min(x), np.max(x)
    boundaries = [x_min] + inflections.tolist() + [x_max]

    # Convert inflection ranges to slice objects
    segments = [
        slice(boundaries[i] + (1 if i > 0 else 0), boundaries[i + 1] + 1)
        for i in range(len(boundaries) - 1)
    ]

    # Collect segmented point groups
    point_segments = [points_per_slice_to_use[s] for s in segments]

    # --- Create tree
    root = SegmentNode("root_pcd", [p for sublist in points_per_slice_to_use for p in sublist], slice_idx)

    for i, pts in enumerate(point_segments):
        total_points = sum(len(point) for point in pts)
        if total_points == 0:
            continue
        child = SegmentNode(name=f"segment_{i+1}", points=[p for sublist in pts for p in sublist], parent=root, slice_idx=None)
        root.add_child(child)
        print(f"Segment {i+1}: {total_points} points")

    return root

def get_category_mapping():
    """Read the category mapping from synsetoffset2category.txt"""
    # Get the script directory and construct path relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from bitsi/ to project root
    mapping_path = os.path.join(project_root, 'data', 'ShapeNetPart', 'synsetoffset2category.txt')
    
    category_map = {}
    reverse_map = {}
    
    with open(mapping_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) == 2:
                    category_name, category_id = parts
                    category_map[category_id] = category_name
                    reverse_map[category_name.lower()] = category_id
    
    return category_map, reverse_map

def get_available_categories():
    """Get list of all available categories with their IDs from the training set"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    train_file_path = os.path.join(project_root, 'data', 'ShapeNetPart', 'train_test_split', 'shuffled_train_file_list.json')
    
    category_map, _ = get_category_mapping()
    available = {}
    
    # Load training file list
    try:
        with open(train_file_path, 'r') as f:
            train_files = json.load(f)
        
        # Count files per category
        for file_path in train_files:
            # Format: "shape_data/{category_id}/{object_id}"
            parts = file_path.split('/')
            if len(parts) >= 3:
                category_id = parts[1]
                if category_id in category_map:
                    if category_id not in available:
                        available[category_id] = 0
                    available[category_id] += 1
        
        # Convert to list format
        result = [(category_map[cat_id], cat_id, count) 
                 for cat_id, count in available.items()]
        return result
    except Exception as e:
        print(f"Error loading training file list: {e}")
        return []

def load_point_cloud_shapenetpart(file_path):
    """
    Load point cloud data from a .txt file
    Expected format (line-by-line): x y z nx ny nz part_id
    - Columns 0-2: x, y, z coordinates
    - Columns 3-5: nx, ny, nz normals
    - Column 6: part_id (integer, may be stored as float)
    """
    try:
        # Load as space-separated values (handles line-by-line format automatically)
        data = np.loadtxt(file_path)
        
        if data.ndim == 1:
            # Single point, reshape to 2D
            data = data.reshape(1, -1)
        
        if data.shape[1] < 3:
            print(f"Warning: File {file_path} has unexpected format (only {data.shape[1]} columns, need at least 3)")
            return None, None, None
        
        # Extract coordinates (first 3 columns: x, y, z)
        points = data[:, :3]
        
        # Extract normals if available (columns 3-5: nx, ny, nz)
        normals = data[:, 3:6] if data.shape[1] >= 6 else None
        
        # Extract part annotations if available (column 6: part_id)
        # Note: part_id may be stored as float (e.g., 5.000000) but should be integer
        part_ids = data[:, 6].astype(int) if data.shape[1] >= 7 else None
        
        return points, normals, part_ids
            
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None, None

def get_random_object(category_name=None, split='train'):
    """
    Get a random object from the specified category in the training set
    
    Args:
        category_name (str, optional): Name of the category (e.g., "airplane", "chair", "table")
                                      If None, selects from all categories
        split (str): Dataset split to use ('train', 'val', or 'test'). Defaults to 'train'
    
    Returns:
        tuple: (points, normals, part_ids, file_path, category_id) or (None, None, None, None, None) if not found
    """
    # Get the script directory and construct paths relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_path = os.path.join(project_root, 'data', 'ShapeNetPart')
    
    # Load the appropriate split file
    split_file = f'shuffled_{split}_file_list.json'
    split_file_path = os.path.join(dataset_path, 'train_test_split', split_file)
    
    if not os.path.exists(split_file_path):
        print(f"Split file not found: {split_file_path}")
        return None, None, None, None, None
    
    # Load file list
    try:
        with open(split_file_path, 'r') as f:
            file_list = json.load(f)
    except Exception as e:
        print(f"Error loading {split_file_path}: {e}")
        return None, None, None, None, None
    
    # Filter by category if specified
    _, reverse_map = get_category_mapping()
    category_id = None
    
    
    if category_name:
        category_name_lower = category_name.lower().strip()
        if category_name_lower in reverse_map:
            category_id = reverse_map[category_name_lower]
        else:
            print(f"Category '{category_name}' not found.")
            print("Available categories:")
            available = get_available_categories()
            for name, cat_id, count in available:
                print(f"  - {name} ({cat_id}): {count} files")
            return None, None, None, None, None
    
    # Filter files by category if specified
    # Note: JSON format uses "shape_data/{category_id}/{object_id}"
    if category_id:
        filtered_files = [item.split('/')[-1] for item in file_list if f'/{category_id}/' in item]
    else:
        filtered_files = file_list

    if not filtered_files:
        print("No files found for requested ShapeNetPart category")
        return None, None, None, None, None

    random_file_path_json = random.choice(filtered_files)
    if '/' in random_file_path_json:
        parts = random_file_path_json.split('/')
        category_id = parts[1]
        object_id = parts[-1]
    else:
        object_id = random_file_path_json

    # Construct actual file path: data/ShapeNetPart/{category_id}/{object_id}
    full_file_path = os.path.join(dataset_path, category_id, object_id)
    if not full_file_path.endswith('.txt'):
        full_file_path += '.txt'
    
    
    if not os.path.exists(full_file_path):
        print(f"File not found: {full_file_path}")
        return None, None, None, None, None
    
    # Load the point cloud
    points, normals, part_ids = load_point_cloud_shapenetpart(full_file_path)
    
    if points is not None:
        category_map, _ = get_category_mapping()
        
        category_display_name = category_map.get(category_id)
        print(f"Loaded random {category_display_name} object from {split} set:")
        #print(f"  File: {file_category_id}/{object_id}.txt")
        print(f"  Points: {len(points)}")
        print(f"  Has normals: {normals is not None}")
        print(f"  Has part annotations: {part_ids is not None}")
        if part_ids is not None:
            unique_parts = np.unique(part_ids)
            print(f"  Parts: {sorted(unique_parts)}")
        print(f"  Bounding box: [{points.min(axis=0)}] to [{points.max(axis=0)}]")
        
        return points, normals, part_ids, full_file_path, category_id
    else:
        return None, None, None, None, None



def apply_slicing_to_children(root, bitsi_x, bitsi_y, bitsi_z, epsilon=5e-3, thickness=None):
    """
    Apply slicing to all children using the 2nd slice index.
    
    Parameters
    ----------
    root : SegmentNode
        Root node containing children to be sliced
    bitsi_x, bitsi_y, bitsi_z : array-like
        BITSI metrics for each axis
    epsilon : float
        Epsilon parameter for slicing
    thickness : float, optional
        Thickness for slicing. If None, will be auto-calculated.
    
    Returns
    -------
    root : SegmentNode
        Root node with updated children (sliced if applicable)
    """
    # Determine the 2nd slice index (non-dominant axis with highest variance)  
    # Process each child
    new_children = []
    for child in root.children:
        if len(child.points) == 0:
            new_children.append(child)
            continue
            
        # Get BITSI metrics for this child
        child_bitsi_x, child_bitsi_y, child_bitsi_z, child_slice_idx, child_points_per_slice_x, child_points_per_slice_y, child_points_per_slice_z = bitsi_metric(
            child.cloud_object, epsilon=epsilon, thickness=thickness or auto_thickness(child.cloud_object, scale=0.03)
        )
        if root.slice_idx == 0:
            child_slice_idx = 1
        elif root.slice_idx == 1:
            child_slice_idx = 0
        print(child_slice_idx)
        
        # Use the 2nd slice index for this child
        child_points_per_slice = [child_points_per_slice_x, child_points_per_slice_y, child_points_per_slice_z]
        child_points_per_slice_to_use = child_points_per_slice[child_slice_idx]
        
        # Apply segmentation to this child
        
        if len(child_points_per_slice_to_use) > 200:
            child_segments = build_segmentation_tree(
                child_points_per_slice_to_use, 
                child_bitsi_x, child_bitsi_y, child_bitsi_z, 
                child_slice_idx, 
                strength_threshold=0.01,
                visualize=False
            )
        
        # Add the child's children as new children of root
        for grandchild in child_segments.children:
            grandchild.parent = root
            grandchild.slice_idx = child_slice_idx
            new_children.append(grandchild)
            print(f"  ↳ Created sub-segment: {grandchild.name} with {len(grandchild.points)} points")
    
    root.children = new_children
    return root

def visualize_shapepart_comparison(root_node, points, part_ids, parent_dir, output_dir=None):
    """
    Visualize ShapeNetPart ground truth vs segmented parts side by side and calculate IoU.
    
    Parameters
    ----------
    root_node : SegmentNode
        Root node of the segmentation tree
    points : numpy array
        Original point cloud points
    part_ids : numpy array
        Ground truth part IDs for each point
    parent_dir : str
        Parent directory to determine if this is ShapeNetPart
    output_dir : str, optional
        Directory to save the visualization. If None, displays the plot.
    """
    if parent_dir != 'ShapeNetPart':
        print("⚠️ This comparison is only available for ShapeNetPart dataset")
        return
    
    # Gather all leaf segments
    def gather_leaves(node):
        leaves = []
        if not node.children:  # This is a leaf node
            leaves.append(node)
        else:
            for child in node.children:
                leaves += gather_leaves(child)
        return leaves

    segmented_leaves = gather_leaves(root_node)
    
    if not segmented_leaves:
        print("⚠️ No segmented parts to compare.")
        return
    
    # Create figure with side-by-side subplots
    fig = plt.figure(figsize=(16, 8))
    
    # Left subplot: Ground truth parts
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_title("Ground Truth Parts (ShapeNetPart)")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y") 
    ax1.set_zlabel("Z")
    
    # Right subplot: Segmented parts
    ax2 = fig.add_subplot(122, projection="3d")
    ax2.set_title("Segmented Parts (BITSI)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    
    # Plot ground truth parts
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
    seg_colors = plt.cm.get_cmap("tab20", len(segmented_leaves))
    
    for i, node in enumerate(segmented_leaves):
        if len(node.points) == 0:
            continue
        pts = np.asarray(node.points)
        color = seg_colors(i)[:3]
        ax2.scatter(pts[:, 0], pts[:, 1], pts[:, 2], 
                   s=5, color=color, alpha=0.8, label=node.name)
    
    # Fix scaling for both subplots
    all_pts = np.vstack([points] + [np.asarray(n.points) for n in segmented_leaves if len(n.points) > 0])
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
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "shapepart_comparison.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"💾 Saved ShapeNetPart comparison to {output_path}")
        plt.close()
    else:
        plt.show()
    
    # Calculate IoU between ground truth and segmented parts
    calculate_iou_metrics(points, part_ids, segmented_leaves, output_dir=output_dir)
    
    # Unsupervised clustering-based fusion
    calculate_unsupervised_clustering(points, part_ids, segmented_leaves, output_dir=output_dir)

def calculate_iou_metrics(points, part_ids, segmented_leaves, output_dir=None):
    """
    Calculate IoU metrics between ground truth and segmented parts.
    Finds best non-overlapping alignment by fusing segmented parts to match GT part count.
    
    Parameters
    ----------
    points : numpy array
        Original point cloud points
    part_ids : numpy array  
        Ground truth part IDs
    segmented_leaves : list
        List of segmented leaf nodes
    output_dir : str, optional
        Directory to save the alignment visualization
    """
    print("\n📊 IoU Analysis (optimal non-overlapping alignment):")
    print("=" * 50)
    
    # Count unique parts in ground truth
    unique_gt_parts = np.unique(part_ids)
    num_gt_parts = len(unique_gt_parts)
    print(f"🔢 Number of unique GT parts: {num_gt_parts}")
    print(f"🔢 Number of segmented parts: {len(segmented_leaves)}")
    
    # Create point-to-segment mapping for segmented parts
    seg_point_to_part = {}
    seg_part_points = {}  # Map segment index to set of point indices
    for i, leaf in enumerate(segmented_leaves):
        if len(leaf.points) == 0:
            continue
        seg_part_points[i] = set()
        for point in leaf.points:
            # Find closest point in original point cloud
            distances = np.linalg.norm(points - point, axis=1)
            closest_idx = np.argmin(distances)
            seg_point_to_part[closest_idx] = i
            seg_part_points[i].add(closest_idx)
    
    # Calculate IoU matrix: GT parts x Segmented parts (or combinations)
    def calculate_iou_for_combination(gt_part_id, seg_combination):
        """Calculate IoU between a GT part and a combination of segmented parts"""
        gt_mask = part_ids == gt_part_id
        gt_indices = set(np.where(gt_mask)[0])
        
        # Get all points in the segmented combination
        seg_indices = set()
        for seg_idx in seg_combination:
            if seg_idx in seg_part_points:
                seg_indices.update(seg_part_points[seg_idx])
        
        intersection = len(gt_indices & seg_indices)
        union = len(gt_indices | seg_indices)
        iou = intersection / union if union > 0 else 0.0
        return iou, intersection, union
    
    # Find best non-overlapping alignment using greedy matching
    # We want to match each GT part to a non-overlapping combination of segmented parts
    used_segments = set()
    alignment = {}  # Maps GT part ID to (seg_combination, iou)
    
    # For each GT part, find the best matching combination that doesn't overlap with already used segments
    for gt_part_id in unique_gt_parts:
        gt_mask = part_ids == gt_part_id
        gt_indices = set(np.where(gt_mask)[0])
        
        # Find which segmented parts overlap with this GT part
        overlapping_segments = set()
        for gt_idx in gt_indices:
            if gt_idx in seg_point_to_part:
                overlapping_segments.add(seg_point_to_part[gt_idx])
        
        if not overlapping_segments:
            alignment[gt_part_id] = (None, 0.0)
            continue
        
        # Try all possible combinations of overlapping segments (excluding already used ones)
        available_segments = [s for s in overlapping_segments if s not in used_segments]
        
        if not available_segments:
            # All overlapping segments are already used, skip this GT part
            alignment[gt_part_id] = (None, 0.0)
            continue
        
        best_iou = 0.0
        best_combination = None
        
        # Try all combinations of available segments (up to reasonable size)
        max_combination_size = min(4, len(available_segments))
        for combo_size in range(1, max_combination_size + 1):
            for combo in combinations(available_segments, combo_size):
                # Check if this combination overlaps with used segments
                combo_set = set(combo)
                if combo_set & used_segments:
                    continue
                
                iou, _, _ = calculate_iou_for_combination(gt_part_id, combo)
                if iou > best_iou:
                    best_iou = iou
                    best_combination = list(combo)
        
        if best_combination:
            alignment[gt_part_id] = (best_combination, best_iou)
            used_segments.update(best_combination)
            print(f"  ✅ GT Part {int(gt_part_id)} → Segments {best_combination} (IoU: {best_iou:.3f})")
        else:
            alignment[gt_part_id] = (None, 0.0)
            print(f"  ⚠️ GT Part {int(gt_part_id)} → No match found")
    
    # Calculate overall metrics
    valid_ious = [iou for _, (_, iou) in alignment.items() if iou > 0]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0
    
    print(f"\n📈 Overall Mean IoU: {mean_iou:.3f}")
    print(f"🎯 Matched parts: {len(valid_ious)}/{num_gt_parts}")
    print(f"🎯 Good matches (IoU > 0.5): {sum(1 for _, (_, iou) in alignment.items() if iou > 0.5)}/{num_gt_parts}")
    
    # Create visualization of the alignment
    if output_dir:
        visualize_alignment(points, part_ids, segmented_leaves, alignment, output_dir)
    
    return alignment, mean_iou

def visualize_alignment(points, part_ids, segmented_leaves, alignment, output_dir):
    """
    Visualize the optimal alignment between GT parts and fused segmented parts.
    
    Parameters
    ----------
    points : numpy array
        Original point cloud points
    part_ids : numpy array
        Ground truth part IDs
    segmented_leaves : list
        List of segmented leaf nodes
    alignment : dict
        Mapping from GT part ID to (seg_combination, iou)
    output_dir : str
        Directory to save the visualization
    """
    unique_gt_parts = np.unique(part_ids)
    
    # Create figure with side-by-side comparison
    fig = plt.figure(figsize=(20, 10))
    
    # Left: Ground Truth
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.set_title("Ground Truth Parts", fontsize=14, fontweight='bold')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    
    # Right: Aligned Segmented Parts (fused)
    ax2 = fig.add_subplot(132, projection="3d")
    ax2.set_title("Aligned Segmented Parts (Fused)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    
    # Bottom: IoU Scores
    ax3 = fig.add_subplot(133)
    ax3.set_title("IoU Scores per Part", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Ground Truth Part ID")
    ax3.set_ylabel("IoU Score")
    ax3.grid(True, alpha=0.3)
    
    # Plot ground truth parts
    gt_colors = plt.cm.get_cmap("tab20", len(unique_gt_parts))
    for i, gt_part_id in enumerate(unique_gt_parts):
        mask = part_ids == gt_part_id
        part_points = points[mask]
        if len(part_points) > 0:
            color = gt_colors(i)[:3]
            ax1.scatter(part_points[:, 0], part_points[:, 1], part_points[:, 2], 
                       s=5, color=color, alpha=0.8, label=f"Part {int(gt_part_id)}")
    
    # Plot aligned segmented parts (fused)
    seg_colors = plt.cm.get_cmap("tab20", len(unique_gt_parts))
    iou_scores = []
    part_labels = []
    
    for i, gt_part_id in enumerate(unique_gt_parts):
        seg_combination, iou = alignment.get(gt_part_id, (None, 0.0))
        iou_scores.append(iou)
        part_labels.append(f"P{int(gt_part_id)}")
        
        if seg_combination:
            # Fuse the segmented parts
            fused_points = []
            for seg_idx in seg_combination:
                if seg_idx < len(segmented_leaves) and len(segmented_leaves[seg_idx].points) > 0:
                    fused_points.append(np.asarray(segmented_leaves[seg_idx].points))
            
            if fused_points:
                fused_points = np.vstack(fused_points)
                color = seg_colors(i)[:3]
                ax2.scatter(fused_points[:, 0], fused_points[:, 1], fused_points[:, 2], 
                           s=5, color=color, alpha=0.8, 
                           label=f"P{int(gt_part_id)} (IoU: {iou:.3f})")
    
    # Plot IoU scores
    bars = ax3.bar(range(len(unique_gt_parts)), iou_scores, 
                   color=[seg_colors(i)[:3] for i in range(len(unique_gt_parts))], alpha=0.7)
    ax3.set_xticks(range(len(unique_gt_parts)))
    ax3.set_xticklabels(part_labels, rotation=45, ha='right')
    ax3.set_ylim([0, 1.1])
    ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='IoU = 0.5')
    ax3.legend()
    
    # Add IoU values on bars
    for i, (bar, iou) in enumerate(zip(bars, iou_scores)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{iou:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Fix scaling for 3D plots
    all_pts = np.vstack([points] + [np.asarray(n.points) for n in segmented_leaves if len(n.points) > 0])
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
    
    # Save the visualization
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "iou_alignment.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved IoU alignment visualization to {output_path}")
    plt.close()

def calculate_unsupervised_clustering(points, part_ids, segmented_leaves, output_dir=None):
    """
    Unsupervised clustering-based fusion of segmented parts using hierarchical clustering.
    Uses rich geometric features and Ward linkage for optimal clustering.
    K is set to the number of unique parts in ground truth.
    
    Features extracted per segment:
    - Centroid (3D position)
    - Bounding box size and volume
    - Point count
    - IBR ratio (graspability metric)
    - Principal direction (orientation)
    - Compactness (density)
    
    Parameters
    ----------
    points : numpy array
        Original point cloud points
    part_ids : numpy array
        Ground truth part IDs (used only to determine K)
    segmented_leaves : list
        List of segmented leaf nodes
    output_dir : str, optional
        Directory to save the clustering visualization
    """
    print("\n🔬 Unsupervised Clustering Analysis (Hierarchical Clustering):")
    print("=" * 50)
    
    # Count unique parts in ground truth to determine K
    unique_gt_parts = np.unique(part_ids)
    num_gt_parts = len(unique_gt_parts)
    K = num_gt_parts
    
    print(f"🔢 Number of unique GT parts (K): {K}")
    print(f"🔢 Number of segmented parts to cluster: {len(segmented_leaves)}")
    
    # Filter out empty segments
    valid_segments = [(i, leaf) for i, leaf in enumerate(segmented_leaves) if len(leaf.points) > 0]
    
    if len(valid_segments) == 0:
        print("⚠️ No valid segments to cluster")
        return None, None
    
    if len(valid_segments) < K:
        print(f"⚠️ Warning: Only {len(valid_segments)} segments available, but K={K}. Using K={len(valid_segments)}")
        K = len(valid_segments)
    
    # Extract rich features for each segment
    segment_features = []
    segment_indices = []
    segment_centroids = []
    segment_bboxes = []
    
    for seg_idx, leaf in valid_segments:
        seg_points = np.asarray(leaf.points)
        
        # 1. Centroid (3D position)
        centroid = np.mean(seg_points, axis=0)
        segment_centroids.append(centroid)
        
        # 2. Size features
        num_points = len(seg_points)
        bbox_min = np.min(seg_points, axis=0)
        bbox_max = np.max(seg_points, axis=0)
        bbox_size = bbox_max - bbox_min
        bbox_volume = np.prod(bbox_size)
        segment_bboxes.append((bbox_min, bbox_max))
        
        # 3. IBR ratio (graspability metric)
        ibr_ratio = leaf.ibr_ratio if hasattr(leaf, 'ibr_ratio') else 0.0
        
        # 4. Principal direction (first principal component)
        if len(seg_points) > 3:
            centered = seg_points - centroid
            cov = np.cov(centered.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            principal_dir = eigenvecs[:, -1]  # Largest eigenvector
        else:
            principal_dir = np.array([0, 0, 1])
        
        # 5. Compactness (ratio of volume to bounding box volume)
        # Approximate as: points per unit volume
        compactness = num_points / (bbox_volume + 1e-10)
        
        # Combine all features
        features = np.concatenate([
            centroid,                    # 3D: position
            bbox_size,                   # 3D: size
            [bbox_volume],               # 1D: volume
            [num_points],                # 1D: point count
            [ibr_ratio],                 # 1D: graspability
            principal_dir,               # 3D: orientation
            [compactness]                # 1D: density
        ])
        
        segment_features.append(features)
        segment_indices.append(seg_idx)
    
    segment_features = np.array(segment_features)
    segment_centroids = np.array(segment_centroids)
    
    # Normalize features for better clustering
    scaler = StandardScaler()
    segment_features_normalized = scaler.fit_transform(segment_features)
    
    # Compute spatial adjacency matrix (distance between segment centroids)
    centroid_distances = cdist(segment_centroids, segment_centroids)
    median_distance = np.median(centroid_distances[centroid_distances > 0])
    
    # Adaptive clustering method selection
    print(f"📐 Median inter-segment distance: {median_distance:.4f}")
    
    # Clustering Method: Hierarchical Clustering with Ward Linkage
    # This method works well for spatial data because:
    # 1. Ward linkage minimizes within-cluster variance
    # 2. It naturally handles clusters of different sizes
    # 3. It respects the feature space structure
    # 
    # Alternative methods that could be tried:
    # - DBSCAN: Good for density-based clustering, but requires tuning eps parameter
    #   dbscan = DBSCAN(eps=median_distance * 0.5, min_samples=1)
    #   cluster_labels = dbscan.fit_predict(segment_features_normalized)
    #
    # - K-means: Simple but assumes spherical clusters
    #   kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    #   cluster_labels = kmeans.fit_predict(segment_features_normalized)
    #
    # - Agglomerative with different linkage: 'complete' or 'average'
    #   agg = AgglomerativeClustering(n_clusters=K, linkage='complete')
    #   cluster_labels = agg.fit_predict(segment_features_normalized)
    
    print(f"🔬 Using Hierarchical Clustering (Ward linkage)...")
    
    # Compute linkage matrix
    linkage_matrix = linkage(segment_features_normalized, method='ward')
    cluster_labels = fcluster(linkage_matrix, K, criterion='maxclust')
    
    # Adjust cluster IDs to start from 0
    cluster_labels = cluster_labels - 1
    
    # If hierarchical clustering produces fewer clusters, use K-means as fallback
    num_clusters_found = len(np.unique(cluster_labels))
    if num_clusters_found < K:
        print(f"⚠️ Hierarchical clustering found {num_clusters_found} clusters, using K-means with K={K}")
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(segment_features_normalized)
    
    # Group segments by cluster
    clusters = {}
    for i, (seg_idx, leaf) in enumerate(valid_segments):
        cluster_id = cluster_labels[i]
        if cluster_id not in clusters:
            clusters[cluster_id] = []
        clusters[cluster_id].append(seg_idx)
    
    print(f"\n📊 Clustering Results:")
    for cluster_id in sorted(clusters.keys()):
        print(f"  Cluster {cluster_id}: {len(clusters[cluster_id])} segments {clusters[cluster_id]}")
    
    # Create point-to-cluster mapping
    seg_point_to_cluster = {}
    seg_idx_to_cluster = {}  # Map original segment index to cluster ID
    for i, (seg_idx, leaf) in enumerate(valid_segments):
        cluster_id = cluster_labels[i]
        seg_idx_to_cluster[seg_idx] = cluster_id
        
        for point in leaf.points:
            # Find closest point in original point cloud
            distances = np.linalg.norm(points - point, axis=1)
            closest_idx = np.argmin(distances)
            seg_point_to_cluster[closest_idx] = cluster_id
    
    # Calculate IoU for each cluster against GT parts
    cluster_ious = {}
    num_clusters_found = len(np.unique(cluster_labels))
    for cluster_id in range(num_clusters_found):
        cluster_mask = np.array([i in seg_point_to_cluster and seg_point_to_cluster[i] == cluster_id 
                                for i in range(len(points))])
        cluster_indices = set(np.where(cluster_mask)[0])
        
        best_iou = 0.0
        best_gt_part = None
        
        for gt_part_id in unique_gt_parts:
            gt_mask = part_ids == gt_part_id
            gt_indices = set(np.where(gt_mask)[0])
            
            intersection = len(gt_indices & cluster_indices)
            union = len(gt_indices | cluster_indices)
            iou = intersection / union if union > 0 else 0.0
            
            if iou > best_iou:
                best_iou = iou
                best_gt_part = gt_part_id
        
        cluster_ious[cluster_id] = (best_gt_part, best_iou)
        print(f"  Cluster {cluster_id} → GT Part {int(best_gt_part) if best_gt_part is not None else 'None'} (IoU: {best_iou:.3f})")
    
    # Calculate overall metrics
    valid_ious = [iou for _, (_, iou) in cluster_ious.items() if iou > 0]
    mean_iou = np.mean(valid_ious) if valid_ious else 0.0
    
    num_clusters_found = len(clusters)
    print(f"\n📈 Overall Mean IoU: {mean_iou:.3f}")
    print(f"🎯 Good matches (IoU > 0.5): {sum(1 for _, (_, iou) in cluster_ious.items() if iou > 0.5)}/{num_clusters_found}")
    
    # Create visualization
    if output_dir:
        visualize_clustering(points, part_ids, segmented_leaves, clusters, cluster_ious, output_dir)
    
    
    return clusters, cluster_ious

def visualize_clustering(points, part_ids, segmented_leaves, clusters, cluster_ious, output_dir):
    """
    Visualize the K-means clustering results.
    
    Parameters
    ----------
    points : numpy array
        Original point cloud points
    part_ids : numpy array
        Ground truth part IDs
    segmented_leaves : list
        List of segmented leaf nodes
    clusters : dict
        Mapping from cluster_id to list of segment indices
    cluster_ious : dict
        Mapping from cluster_id to (best_gt_part, iou)
    output_dir : str
        Directory to save the visualization
    """
    unique_gt_parts = np.unique(part_ids)
    K = len(clusters)
    
    # Create figure with side-by-side comparison
    fig = plt.figure(figsize=(20, 10))
    
    # Left: Ground Truth
    ax1 = fig.add_subplot(131, projection="3d")
    ax1.set_title("Ground Truth Parts", fontsize=14, fontweight='bold')
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    
    # Middle: Clustered Segmented Parts (fused)
    ax2 = fig.add_subplot(132, projection="3d")
    ax2.set_title(f"Hierarchical Clustered Parts (K={K})", fontsize=14, fontweight='bold')
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    
    # Right: IoU Scores
    ax3 = fig.add_subplot(133)
    ax3.set_title("IoU Scores per Cluster", fontsize=14, fontweight='bold')
    ax3.set_xlabel("Cluster ID")
    ax3.set_ylabel("IoU Score")
    ax3.grid(True, alpha=0.3)
    
    # Plot ground truth parts
    gt_colors = plt.cm.get_cmap("tab20", len(unique_gt_parts))
    for i, gt_part_id in enumerate(unique_gt_parts):
        mask = part_ids == gt_part_id
        part_points = points[mask]
        if len(part_points) > 0:
            color = gt_colors(i)[:3]
            ax1.scatter(part_points[:, 0], part_points[:, 1], part_points[:, 2], 
                       s=5, color=color, alpha=0.8, label=f"Part {int(gt_part_id)}")
    
    # Plot clustered segmented parts (fused)
    cluster_colors = plt.cm.get_cmap("tab20", K)
    iou_scores = []
    cluster_labels = []
    
    for cluster_id in sorted(clusters.keys()):
        best_gt_part, iou = cluster_ious.get(cluster_id, (None, 0.0))
        iou_scores.append(iou)
        cluster_labels.append(f"C{cluster_id}")
        
        # Fuse all segments in this cluster
        fused_points = []
        for seg_idx in clusters[cluster_id]:
            if seg_idx < len(segmented_leaves) and len(segmented_leaves[seg_idx].points) > 0:
                fused_points.append(np.asarray(segmented_leaves[seg_idx].points))
        
        if fused_points:
            fused_points = np.vstack(fused_points)
            color = cluster_colors(cluster_id)[:3]
            label = f"C{cluster_id}"
            if best_gt_part is not None:
                label += f"→P{int(best_gt_part)}"
            label += f" (IoU: {iou:.3f})"
            ax2.scatter(fused_points[:, 0], fused_points[:, 1], fused_points[:, 2], 
                       s=5, color=color, alpha=0.8, label=label)
    
    # Plot IoU scores
    bars = ax3.bar(range(K), iou_scores, 
                   color=[cluster_colors(i)[:3] for i in range(K)], alpha=0.7)
    ax3.set_xticks(range(K))
    ax3.set_xticklabels(cluster_labels, rotation=45, ha='right')
    ax3.set_ylim([0, 1.1])
    ax3.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='IoU = 0.5')
    ax3.legend()
    
    # Add IoU values on bars
    for i, (bar, iou) in enumerate(zip(bars, iou_scores)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{iou:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Fix scaling for 3D plots
    all_pts = np.vstack([points] + [np.asarray(n.points) for n in segmented_leaves if len(n.points) > 0])
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
    
    # Save the visualization
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "unsupervised_clustering.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"💾 Saved unsupervised clustering visualization to {output_path}")
    plt.close()

def rename_segments(root, prefix="segment"):
    """
    Rename all segments in the tree with sequential numbers.
    
    Parameters
    ----------
    root : SegmentNode
        Root node of the segmentation tree
    prefix : str, optional
        Prefix to use for segment names, defaults to "segment"
        
    Returns
    -------
    root : SegmentNode
        Root node with renamed segments
    """
    def _rename_recursive(node, counter=[0]):
        # Skip renaming the root node
        if node.parent is not None:
            node.name = f"{prefix}_{counter[0]}"
            counter[0] += 1
            
        # Recursively rename children
        for child in node.children:
            _rename_recursive(child, counter)
            
    _rename_recursive(root)
    return root


# ===============================================================
# ADDED FOR VISUALIZATION TOOL INTEGRATION
# Generic loaders + callable single-object segmentation API
# ===============================================================
def load_point_cloud_input(input_data):
    """Load a point cloud from a file path, numpy array, or Open3D object."""
    if isinstance(input_data, o3d.geometry.PointCloud):
        return o3d.geometry.PointCloud(input_data)

    if isinstance(input_data, np.ndarray):
        if input_data.ndim != 2 or input_data.shape[1] < 3:
            raise ValueError("Point array must have shape [N, 3+]")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(input_data[:, :3])
        if input_data.shape[1] >= 6:
            pcd.normals = o3d.utility.Vector3dVector(input_data[:, 3:6])
        return pcd

    if not isinstance(input_data, str):
        raise TypeError("input_data must be a file path, numpy array, or Open3D point cloud")

    if not os.path.exists(input_data):
        raise FileNotFoundError(f"Point cloud file not found: {input_data}")

    ext = os.path.splitext(input_data)[1].lower()
    if ext in {'.ply', '.pcd', '.xyz', '.xyzn', '.xyzrgb', '.pts'}:
        pcd = o3d.io.read_point_cloud(input_data)
    elif ext in {'.txt', '.csv'}:
        data = np.loadtxt(input_data, delimiter=',' if ext == '.csv' else None)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        if data.shape[1] < 3:
            raise ValueError(f"File {input_data} must contain at least 3 columns")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(data[:, :3])
        if data.shape[1] >= 6:
            pcd.normals = o3d.utility.Vector3dVector(data[:, 3:6])
    else:
        raise ValueError(f"Unsupported point cloud format: {ext}")

    if pcd.is_empty():
        raise ValueError(f"Loaded point cloud is empty: {input_data}")
    return pcd


def collect_leaf_segments(root):
    """Return leaf nodes only. If tree has no children, return the root itself."""
    leaves = []

    def _walk(node):
        if not node.children:
            leaves.append(node)
            return
        for child in node.children:
            _walk(child)

    _walk(root)
    return leaves


def assign_segment_labels_to_points(points, root, decimals=8):
    """Map BITSI leaf segments back to an [N] label array for the original points."""
    points = np.asarray(points)
    if points.size == 0:
        return np.array([], dtype=int)

    leaves = collect_leaf_segments(root)
    labels = np.full(len(points), -1, dtype=int)

    rounded_lookup = {}
    for label_id, leaf in enumerate(leaves):
        leaf_points = np.asarray(leaf.points)
        if leaf_points.size == 0:
            continue
        for point in np.round(leaf_points, decimals=decimals):
            rounded_lookup[tuple(point.tolist())] = label_id

    rounded_points = np.round(points, decimals=decimals)
    unresolved = []
    for idx, point in enumerate(rounded_points):
        labels[idx] = rounded_lookup.get(tuple(point.tolist()), -1)
        if labels[idx] == -1:
            unresolved.append(idx)

    if unresolved:
        leaf_point_blocks = []
        leaf_labels = []
        for label_id, leaf in enumerate(leaves):
            leaf_points = np.asarray(leaf.points)
            if leaf_points.size == 0:
                continue
            leaf_point_blocks.append(leaf_points)
            leaf_labels.extend([label_id] * len(leaf_points))

        if leaf_point_blocks:
            all_leaf_points = np.vstack(leaf_point_blocks)
            tree = cKDTree(all_leaf_points)
            _, nn_idx = tree.query(points[unresolved], k=1)
            leaf_labels = np.asarray(leaf_labels, dtype=int)
            labels[unresolved] = leaf_labels[nn_idx]

    if np.any(labels < 0):
        labels[labels < 0] = 0
    return labels


def segment_single_point_cloud(input_data, gripper_width=0.13, gripper_height=0.07,
                               epsilon=5e-3, ibr_tolerance=0.003,
                               strength_threshold=0.01, output_dir=None,
                               visualize=False, segment_name='segment'):
    """Run BITSI single-object segmentation on a point cloud and return labels."""
    # ADDED: reusable API for the visualization tool and any other caller
    pcd = load_point_cloud_input(input_data)
    pcd.paint_uniform_color([0, 0, 1])

    cloud_object = build_cloud_object(pcd, gripper_width, gripper_height)
    thickness = auto_thickness(cloud_object, scale=0.03)
    bitsi_x, bitsi_y, bitsi_z, slice_idx, points_per_slice_x, points_per_slice_y, points_per_slice_z = bitsi_metric(
        cloud_object, epsilon=epsilon, thickness=thickness
    )
    points_per_slice = [points_per_slice_x, points_per_slice_y, points_per_slice_z]
    points_per_slice_to_use = points_per_slice[slice_idx]

    pretty_print_bitsi(bitsi_x, bitsi_y, bitsi_z, slice_idx)

    try:
        root = build_segmentation_tree(
            points_per_slice_to_use, bitsi_x, bitsi_y, bitsi_z, slice_idx,
            strength_threshold, output_dir=output_dir
        )
        root = fuse_consecutive_segments(root, ibr_tolerance=ibr_tolerance)
        root = rename_segments(root, prefix=segment_name)
    except Exception as exc:
        print(f"⚠️ BITSI tree construction failed, falling back to one segment: {exc}")
        root = SegmentNode(
            name=segment_name,
            points=np.asarray(pcd.points),
            slice_idx=slice_idx,
            gripper_width=gripper_width,
            gripper_height=gripper_height,
            epsilon=epsilon,
        )

    labels = assign_segment_labels_to_points(np.asarray(pcd.points), root)

    if visualize and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        visualize_tree_segments(root, output_dir=output_dir, filename=f"{segment_name}_segmented_parts.png")

    return {
        'pcd': pcd,
        'root': root,
        'labels': labels,
        'slice_idx': slice_idx,
        'thickness': thickness,
        'bitsi_x': bitsi_x,
        'bitsi_y': bitsi_y,
        'bitsi_z': bitsi_z,
        'points_per_slice': points_per_slice_to_use,
    }


def main():
    #np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    
    # Create an ArgumentParser Object
    parser = argparse.ArgumentParser(description='Task-Oriented Grasp Synthesis')

    # Add a command-line argument for the input filenamep
    parser.add_argument('--object', type=str, help='Name of the ycb object', default='toy')
    parser.add_argument('--width', type=float, help='gripper width', default=0.13)
    parser.add_argument('--height', type=float, help='gripper height', default=0.07)      
    parser.add_argument('--partial', action='store_true', help='use partial or not partial data')
    parser.add_argument('--epsilon', type=float, default=5e-3 )
    parser.add_argument('--parent_dir', type=str, default='YCBV')
    parser.add_argument('--obj_num', type=int, default=1)
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save visualization outputs')
    parser.add_argument('--adversarial', action='store_true', help='use adversarial or not adversarial data')
    # ADDED: allow direct segmentation from a point-cloud file path
    parser.add_argument('--input_path', type=str, default=None, help='Direct path to a point cloud file')
    
    args = parser.parse_args()
    gripper_width = args.width 
    gripper_height = args.height  
    epsilon = args.epsilon 
    
    object_path = args.object
    parent_dir = args.parent_dir
    
    # Parameters for segmentation 
    ibr_tolerance = 0.003
    strength_threshold = 0.01
    epsilon = 5e-3 
    
    # -------------------------------------------------------------------------
    # Directory management
    # -------------------------------------------------------------------------
    if args.input_path:
        # ADDED: direct file-based loading for visualization-tool integration
        pcd = load_point_cloud_input(args.input_path)
        object_path = os.path.splitext(os.path.basename(args.input_path))[0]
        parent_dir = 'CUSTOM'

    elif parent_dir == 'HANDAL':
        model_num = f"{args.obj_num:0{6}}"
        parent_dir = os.path.join("/home/khiem/Point-Cloud-Decomposition-for-Grasping/",args.parent_dir)
        pcd = o3d.io.read_point_cloud(f"{parent_dir}/{object_path}/models/obj_{model_num}.ply")
        pcd_handle =  o3d.io.read_point_cloud(f"{parent_dir}/{object_path}/models_parts/obj_{model_num}_handle.ply")
        pcd_body =  o3d.io.read_point_cloud(f"{parent_dir}/{object_path}/models_parts/obj_{model_num}_not.ply")
        pcd = adjust_handal(pcd)
        pcd_body = adjust_handal(pcd_body)
        pcd_handle = adjust_handal(pcd_handle)
        pcd_handle.paint_uniform_color([0, 1, 0])

    elif parent_dir == "YCBV" or parent_dir == 'YCBV-Partial':
        parent_dir = os.path.join("/home/khiem/Point-Cloud-Decomposition-for-Grasping/",args.parent_dir)
      
        pcd = o3d.io.read_point_cloud(f"{parent_dir}/{object_path}/nontextured.ply")            
        pcd.paint_uniform_color([0, 0, 1])
        
    elif parent_dir == 'KITTI':
        parent_dir = os.path.join("/home/khiem/Point-Cloud-Decomposition-for-Grasping/",args.parent_dir)
        pcd = o3d.io.read_point_cloud(f"{parent_dir}/{object_path}.ply")   
        pcd = adjust_kitti(pcd)
    
    elif parent_dir == 'ShapeNetPart':
        points, normals, part_ids, file_path, category_id = get_random_object(object_path)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.paint_uniform_color([0, 0, 1])
        
    # -------------------------------------------------------------------------
    # Loading Point Cloud
    # -------------------------------------------------------------------------
    object_path = args.object
    print(f"📦 Loading object: {args.object}")
    pcd.paint_uniform_color([0, 0, 1]) # --> use only pc color 
    if pcd.is_empty():
        sys.exit(f"❌ Error: Loaded point cloud is empty at {object_path}")

    # -------------------------------------------------------------------------
    # Calculate BITSI Metric
    # -------------------------------------------------------------------------
    cloud_object = build_cloud_object(pcd, gripper_width, gripper_height)
    thickness = auto_thickness(cloud_object, scale=0.03)
    print(f"📏 Adaptive thickness selected: {thickness:.6f} (scale=5%)")
    print("⚙️  Computing BITSI metrics...")
    bitsi_x, bitsi_y, bitsi_z, slice_idx, points_per_slice_x, points_per_slice_y, points_per_slice_z = bitsi_metric(cloud_object, epsilon=epsilon, thickness=thickness)
    points_per_slice = [points_per_slice_x, points_per_slice_y, points_per_slice_z] 
    points_per_slice_to_use = points_per_slice[slice_idx]
    pretty_print_bitsi(bitsi_x, bitsi_y, bitsi_z, slice_idx)
   
    #visualize_bitsi(bitsi_x, bitsi_y, bitsi_z)

    # -------------------------------------------------------------------------
    # Segmentation With BITSI Metric
    # -------------------------------------------------------------------------
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    root = build_segmentation_tree(points_per_slice_to_use, bitsi_x, bitsi_y, bitsi_z, slice_idx, strength_threshold, output_dir=output_dir)
    root = fuse_consecutive_segments(root, ibr_tolerance=ibr_tolerance)
    #root = apply_slicing_to_children(root, bitsi_x, bitsi_y, bitsi_z, epsilon=epsilon, thickness=thickness)
    root = rename_segments(root)
    
    
    #root.print_tree()
    #fuse_oversegmented_children(root, ibr_tol=0.05, var_tol=0.01) 
    root.print_tree()
    if parent_dir == 'ShapeNetPart':
        # For ShapeNetPart, show comparison with ground truth
        visualize_shapepart_comparison(root, points, part_ids, parent_dir, output_dir=output_dir)
    else:
        # For other datasets, use regular visualization
        visualize_tree_segments(root, output_dir=output_dir, filename=f"{object_path}_segmented_parts.png")
    
if __name__ == "__main__":
    main()