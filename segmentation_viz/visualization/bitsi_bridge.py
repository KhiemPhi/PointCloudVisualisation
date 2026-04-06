import os
import sys
import logging
from collections import defaultdict

import numpy as np
from django.conf import settings


logger = logging.getLogger(__name__)

# NEW FILE:
# This file is the bridge between the visualization tool and the BITSI repo.
# The visualization tool calls functions here; this file imports BITSI and runs it.

# Cache for loaded modules so Open3D is only imported once (it is slow to load).
_cached_modules = {}


def _bitsi_paths():
    """Resolve the path to the BITSI repository.

    The function checks several possible locations in priority order:
    1. third_party/bitsi-main inside the Django project (vendored copy)
    2. bitsi_main as a sibling directory of the Django project
    This makes deployment flexible: either vendor BITSI inside the project,
    or keep it as a sibling folder during development.
    """
    # Option 1: vendored inside the Django project
    vendored = os.path.join(settings.BASE_DIR, 'third_party', 'bitsi-main')
    if os.path.isdir(vendored):
        slicer_root = os.path.join(vendored, 'bitsi_slicer')
        return vendored, slicer_root

    # Option 2: sibling directory (typical development layout)
    # settings.BASE_DIR = .../PointCloudVisualisation-master/segmentation_viz
    # bitsi_main lives inside PointCloudVisualisation-master, alongside segmentation_viz
    # We go up one level from BASE_DIR to reach PointCloudVisualisation-master
    project_parent = os.path.dirname(settings.BASE_DIR)
    sibling = os.path.join(project_parent, 'bitsi_main')
    if os.path.isdir(sibling):
        slicer_root = os.path.join(sibling, 'bitsi_slicer')
        return sibling, slicer_root

    raise FileNotFoundError(
        "Cannot find the BITSI repository. Expected it either at "
        f"'{vendored}' (vendored) or '{sibling}' (sibling of segmentation_viz "
        "inside PointCloudVisualisation-master). Please copy or symlink the "
        "bitsi_main directory to one of these locations."
    )


def _ensure_bitsi_imports():
    """Add the BITSI repo root to sys.path so its modules are importable.

    Only repo_root is added. This allows:
    - 'import main' and 'import main_multi_category' to work
    - 'from bitsi_slicer.bitsi_slicer import ...' inside main.py to resolve
      correctly (Python finds the outer bitsi_slicer/ directory as a namespace
      package, then the inner bitsi_slicer/ package with __init__.py).

    IMPORTANT: Do NOT add slicer_root (repo_root/bitsi_slicer) to sys.path.
    Doing so causes Python to find the inner bitsi_slicer package directly,
    which breaks the 'from bitsi_slicer.bitsi_slicer import ...' chain that
    main.py relies on.
    """
    repo_root, _slicer_root = _bitsi_paths()
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


def _load_modules():
    """Import the original BITSI modules instead of rewriting the algorithm here.

    Modules are cached after the first import to avoid the overhead of
    re-importing Open3D on every request.
    """
    if _cached_modules:
        return _cached_modules['o3d'], _cached_modules['main'], _cached_modules['multi']

    _ensure_bitsi_imports()
    import open3d as o3d
    import main as bitsi_main
    import main_multi_category as bitsi_multi

    _cached_modules['o3d'] = o3d
    _cached_modules['main'] = bitsi_main
    _cached_modules['multi'] = bitsi_multi

    return o3d, bitsi_main, bitsi_multi


def _entry_to_pcd(entry, o3d):
    """Convert the viewer's JSON point format into an Open3D point cloud for BITSI."""
    coords = entry.get('coordinates', []) or []
    pts = []
    for c in coords:
        if isinstance(c, (list, tuple)) and len(c) >= 3:
            pts.append([float(c[0]), float(c[1]), float(c[2])])
    if not pts:
        raise ValueError('No valid XYZ points found in the selected point cloud.')
    pts = np.asarray(pts, dtype=float)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    colors = entry.get('colors')
    if isinstance(colors, list) and len(colors) == len(pts):
        arr = np.asarray(colors, dtype=float)
        if arr.max() > 1.0:
            arr = arr / 255.0
        pcd.colors = o3d.utility.Vector3dVector(arr[:, :3])
    return pcd, pts


def _leaf_nodes(root):
    """Flatten BITSI's segmentation tree so each leaf becomes one final segment label."""
    if root is None:
        return []
    if not getattr(root, 'children', None):
        return [root]
    out = []
    stack = [root]
    while stack:
        node = stack.pop()
        children = getattr(node, 'children', None) or []
        if children:
            stack.extend(children)
        else:
            out.append(node)
    return out


def _build_index_map(points, decimals=8):
    """Map rounded XYZ coordinates back to original point indices in the viewer file."""
    key_to_indices = defaultdict(list)
    for idx, p in enumerate(points):
        key = tuple(np.round(p, decimals=decimals))
        key_to_indices[key].append(idx)
    return key_to_indices


def _assign_points_to_label(points_subset, key_to_indices, labels, label_value, decimals=8):
    """Take BITSI output points and assign a part label back onto the original point list."""
    matched = 0
    for p in np.asarray(points_subset):
        key = tuple(np.round(p, decimals=decimals))
        if key_to_indices[key]:
            idx = key_to_indices[key].pop()
            labels[idx] = int(label_value)
            matched += 1
    return matched


def _run_single_segmentation(pcd, bitsi_main, epsilon=5e-3, gripper_width=0.13, gripper_height=0.07,
                             strength_threshold=0.01, ibr_tolerance=0.003):
    """Wrapper for BITSI single-object segmentation using main.py."""
    cloud_object = bitsi_main.build_cloud_object(pcd, gripper_width, gripper_height)
    thickness = bitsi_main.auto_thickness(cloud_object, scale=0.03)
    bitsi_x, bitsi_y, bitsi_z, slice_idx, pps_x, pps_y, pps_z = bitsi_main.bitsi_metric(
        cloud_object, epsilon=epsilon, thickness=thickness
    )
    points_per_slice = [pps_x, pps_y, pps_z][slice_idx]
    root = bitsi_main.build_segmentation_tree(
        points_per_slice, bitsi_x, bitsi_y, bitsi_z, slice_idx, strength_threshold,
        max_deg=12, visualize=False
    )
    root = bitsi_main.fuse_consecutive_segments(root, ibr_tolerance=ibr_tolerance)
    root = bitsi_main.rename_segments(root)
    return root


def _run_multi_segmentation(pcd, bitsi_main, bitsi_multi, epsilon=5e-3, gripper_width=0.05,
                            gripper_height=0.07, strength_threshold=0.001, ibr_tolerance=0.02,
                            octree_depth=8, min_points_per_voxel=10):
    """Wrapper for multi-object flow.

    First split the scene with BITSI's main_multi_category.py,
    then run the regular BITSI segmentation on each chunk.
    """
    segmented_pcds = bitsi_multi.octree_based_segmentation(
        pcd, max_depth=octree_depth, min_points_per_voxel=min_points_per_voxel
    )
    if not segmented_pcds:
        segmented_pcds = [pcd]

    roots = []
    for seg_idx, seg_pcd in enumerate(segmented_pcds, start=1):
        seg_cloud_object = bitsi_main.build_cloud_object(seg_pcd, gripper_width, gripper_height)
        overall_ibr_ratio = bitsi_main.get_overall_ibr_ratio(
            seg_cloud_object, epsilon=epsilon, thickness=0.001,
            gripper_length=gripper_width, base_ibr=0.7
        )

        if (seg_cloud_object.x_dim <= gripper_width and seg_cloud_object.y_dim <= gripper_width) or overall_ibr_ratio >= 0.7:
            simple_root = bitsi_main.SegmentNode(
                name=f"octree_segment_{seg_idx}",
                points=np.asarray(seg_pcd.points).tolist(),
                slice_idx=0,
                gripper_width=gripper_width,
                gripper_height=gripper_height,
                epsilon=epsilon,
            )
            roots.append(simple_root)
            continue

        dimensions = np.array([seg_cloud_object.x_dim, seg_cloud_object.y_dim, seg_cloud_object.z_dim])
        seg_slice_idx = int(np.argmax(dimensions))
        seg_thickness = bitsi_main.auto_thickness(seg_cloud_object, scale=0.001)
        seg_bitsi_x, seg_bitsi_y, seg_bitsi_z, _, seg_pps_x, seg_pps_y, seg_pps_z = bitsi_main.bitsi_metric(
            seg_cloud_object, epsilon=epsilon, thickness=seg_thickness
        )
        seg_points_per_slice = [seg_pps_x, seg_pps_y, seg_pps_z][seg_slice_idx]
        try:
            seg_root = bitsi_main.build_segmentation_tree(
                seg_points_per_slice, seg_bitsi_x, seg_bitsi_y, seg_bitsi_z,
                seg_slice_idx, strength_threshold, max_deg=10, visualize=False
            )
            seg_root = bitsi_main.fuse_consecutive_segments(seg_root, ibr_tolerance=ibr_tolerance)
            seg_root = bitsi_main.rename_segments(seg_root, prefix=f"octree_segment_{seg_idx}")
        except Exception:
            all_points = []
            for slice_points in seg_points_per_slice:
                all_points.extend(slice_points)
            seg_root = bitsi_main.SegmentNode(
                name=f"octree_segment_{seg_idx}",
                points=all_points,
                slice_idx=seg_slice_idx,
                gripper_width=gripper_width,
                gripper_height=gripper_height,
                epsilon=epsilon,
            )
        roots.append(seg_root)
    return roots


def apply_bitsi_segmentation_to_entry(entry, mode='single'):
    """Main bridge entry point called from views.py.

    Converts the selected point cloud to Open3D, runs BITSI, then maps
    the segment labels back onto the original point list.
    """
    o3d, bitsi_main, bitsi_multi = _load_modules()
    pcd, original_points = _entry_to_pcd(entry, o3d)
    labels = np.full(len(original_points), -1, dtype=int)
    key_to_indices = _build_index_map(original_points)

    if mode == 'single':
        roots = [_run_single_segmentation(pcd, bitsi_main)]
    elif mode == 'multi':
        roots = _run_multi_segmentation(pcd, bitsi_main, bitsi_multi)
    else:
        raise ValueError("mode must be 'single' or 'multi'")

    next_label = 0
    total_matched = 0
    for root in roots:
        leaf_nodes = _leaf_nodes(root)
        if not leaf_nodes:
            leaf_nodes = [root]
        for leaf in leaf_nodes:
            pts = np.asarray(leaf.points)
            if len(pts) == 0:
                continue
            matched = _assign_points_to_label(pts, key_to_indices, labels, next_label)
            total_matched += matched
            next_label += 1

    # Warn if many points were not matched (possible precision issue)
    unmatched = int(np.sum(labels < 0))
    if unmatched > 0:
        logger.warning(
            "BITSI segmentation: %d of %d points could not be matched back "
            "to the original point cloud (assigned fallback label 0). "
            "This may indicate a floating-point precision issue.",
            unmatched, len(original_points)
        )
    # Fallback so no point remains unlabeled in the viewer file
    labels[labels < 0] = 0

    entry['part_label'] = labels.astype(int).tolist()
    meta = {
        'mode': mode,
        'num_segments': int(next_label if next_label > 0 else 1),
        'unmatched_points': unmatched,
        'total_points': len(original_points),
    }
    entry['segmentation_meta'] = meta
    return entry, meta
