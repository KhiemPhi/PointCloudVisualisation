#!/usr/bin/env python3
"""
ransac_clean.py — Standalone point cloud noise cleaning script.

Reads every *_points.json from the static/ directory, applies:
  - RANSAC plane removal (full-scene files only, 1 pass → removes table surface)
  - Statistical outlier removal (all files)

Writes cleaned versions to static/ransac_cleaned/ without touching the originals.

Usage
-----
    python ransac_clean.py                        # process all files
    python ransac_clean.py --full_scene_only       # only *full_scene* files
    python ransac_clean.py --std_ratio 1.5         # tighter outlier filter
    python ransac_clean.py --input_dir /path/to/static --output_dir /path/to/out
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------
# Path resolution — works whether you run from repo root, bitsi_main/, etc.
# ---------------------------------------------------------------------------

def _resolve_dirs(input_dir_arg: str | None, output_dir_arg: str | None):
    """Find the static/ directory relative to this script's location."""
    script_dir = Path(__file__).resolve().parent          # bitsi_main/
    repo_root  = script_dir.parent                        # PointCloudVisualisation/
    default_in  = repo_root / "segmentation_viz" / "static"
    default_out = default_in / "ransac_cleaned"

    input_dir  = Path(input_dir_arg)  if input_dir_arg  else default_in
    output_dir = Path(output_dir_arg) if output_dir_arg else default_out

    if not input_dir.is_dir():
        sys.exit(f"❌  Input directory not found: {input_dir}")

    return input_dir, output_dir


# ---------------------------------------------------------------------------
# Threshold helper
# ---------------------------------------------------------------------------

def auto_threshold(xyz: np.ndarray) -> float:
    """
    Compute RANSAC inlier distance threshold as 1% of the point cloud's
    largest bounding-box extent.

    For a ~1.5 m tabletop scene this gives ~1.5–1.8 cm, which sits comfortably
    between the median point spacing (~9 mm) and typical object sizes (>5 cm).
    """
    extent = xyz.max(axis=0) - xyz.min(axis=0)
    return float(0.01 * extent.max())


# ---------------------------------------------------------------------------
# Cleaning functions
# ---------------------------------------------------------------------------

def ransac_remove_table(pcd: o3d.geometry.PointCloud,
                        threshold: float,
                        ransac_n: int = 3,
                        num_iterations: int = 1000,
                        max_removal_fraction: float = 0.60):
    """
    Remove the dominant plane (table surface) with one RANSAC pass.

    Returns
    -------
    cleaned_pcd : o3d.geometry.PointCloud
        Point cloud with the plane inliers removed.
    kept_local : np.ndarray
        Integer indices into the *input* pcd that survive (non-plane points).
    n_removed : int
        Number of plane inlier points removed.
    plane_eq : tuple
        (a, b, c, d) coefficients of the detected plane ax+by+cz+d=0.
    skipped : bool
        True if the safety guard fired and RANSAC was not applied.
    """
    plane_model, inlier_idx = pcd.segment_plane(
        distance_threshold=threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    n_total   = len(pcd.points)
    n_inliers = len(inlier_idx)

    # Safety guard: don't remove more than max_removal_fraction of the cloud.
    if n_inliers > max_removal_fraction * n_total:
        print(f"    [RANSAC] ⚠️  Safety guard: would remove "
              f"{n_inliers}/{n_total} pts ({100*n_inliers/n_total:.1f}%) "
              f"— skipping this pass.")
        all_idx = np.arange(n_total, dtype=np.int64)
        return pcd, all_idx, 0, tuple(plane_model), True

    # Build a boolean mask over current pcd; invert to get non-plane points.
    keep_mask = np.ones(n_total, dtype=bool)
    keep_mask[np.asarray(inlier_idx)] = False
    kept_local = np.where(keep_mask)[0]

    cleaned = pcd.select_by_index(kept_local.tolist())
    a, b, c, d = plane_model
    return cleaned, kept_local, n_inliers, (a, b, c, d), False


def statistical_outlier_removal(pcd: o3d.geometry.PointCloud,
                                nb_neighbors: int = 20,
                                std_ratio: float = 2.0):
    """
    Remove isolated points that are statistical outliers.

    Returns
    -------
    cleaned_pcd, kept_local (indices into input pcd), n_removed
    """
    cleaned, keep_idx = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )
    kept_local = np.asarray(keep_idx, dtype=np.int64)
    n_removed  = len(pcd.points) - len(cleaned.points)
    return cleaned, kept_local, n_removed


# ---------------------------------------------------------------------------
# JSON I/O helpers
# ---------------------------------------------------------------------------

def _extract_xyz(coordinates: list) -> np.ndarray:
    """Pull [x, y, z] from coordinate entries that may be [x, y, z] or [x, y, z, 'custom']."""
    return np.array([[float(c[0]), float(c[1]), float(c[2])] for c in coordinates],
                    dtype=np.float64)


def _build_pcd(xyz: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd


def _subset_list(lst: list, indices: np.ndarray) -> list:
    """Return elements of lst at the given integer indices."""
    return [lst[i] for i in indices]


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def process_file(json_path: Path,
                 output_path: Path,
                 is_full_scene: bool,
                 nb_neighbors: int,
                 std_ratio: float,
                 max_removal_fraction: float) -> dict:
    """
    Clean one JSON file and write the result.

    Returns a summary dict for the final report.
    """
    try:
        with open(json_path, "r") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"  ⚠️  Corrupt JSON ({exc}), skipping.")
        return {"file": json_path.name, "skipped": True, "reason": "corrupt json"}

    if not data or not isinstance(data, list):
        print(f"  ⚠️  Unexpected format, skipping.")
        return {"file": json_path.name, "skipped": True}

    entry = data[0]
    coordinates = entry.get("coordinates", [])
    part_labels  = entry.get("part_label", [])
    colors       = entry.get("colors", None)

    if len(coordinates) < 10:
        print(f"  ⚠️  Too few points ({len(coordinates)}), skipping.")
        return {"file": json_path.name, "skipped": True}

    n_original = len(coordinates)
    xyz = _extract_xyz(coordinates)
    pcd = _build_pcd(xyz)

    # surviving_indices tracks which *original* rows survive each filter step.
    surviving_indices = np.arange(n_original, dtype=np.int64)

    meta = {
        "original_points": n_original,
        "ransac_applied":  False,
        "ransac_threshold": None,
        "ransac_removed":  0,
        "plane_equation":  None,
        "ransac_skipped_safety": False,
        "statistical_removed": 0,
    }

    # ------------------------------------------------------------------
    # Step 1: RANSAC — full-scene files only
    # ------------------------------------------------------------------
    if is_full_scene:
        threshold = auto_threshold(xyz)
        meta["ransac_threshold"] = round(threshold, 6)
        meta["ransac_applied"]   = True

        print(f"    Auto threshold: {threshold:.4f} m  "
              f"(1% of {(xyz.max(0)-xyz.min(0)).max():.3f} m extent)")

        pcd, kept_local, n_removed, plane_eq, skipped = ransac_remove_table(
            pcd, threshold,
            max_removal_fraction=max_removal_fraction,
        )
        surviving_indices = surviving_indices[kept_local]

        meta["ransac_removed"]       = int(n_removed)
        meta["ransac_skipped_safety"] = skipped
        meta["plane_equation"]       = {
            "a": round(plane_eq[0], 6),
            "b": round(plane_eq[1], 6),
            "c": round(plane_eq[2], 6),
            "d": round(plane_eq[3], 6),
        }

        if skipped:
            print(f"    [RANSAC] Skipped (safety guard).")
        else:
            a, b, c, d = plane_eq
            print(f"    [RANSAC] Removed {n_removed:,} pts  "
                  f"— plane ({a:.3f}x + {b:.3f}y + {c:.3f}z = {-d:.3f})")

    # ------------------------------------------------------------------
    # Step 2: Statistical outlier removal — all files
    # ------------------------------------------------------------------
    pcd, kept_local, n_stat_removed = statistical_outlier_removal(
        pcd, nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    surviving_indices = surviving_indices[kept_local]
    meta["statistical_removed"] = int(n_stat_removed)
    print(f"    [StatOutlier] Removed {n_stat_removed:,} pts")

    # ------------------------------------------------------------------
    # Step 3: Rebuild JSON entry using surviving_indices
    # ------------------------------------------------------------------
    cleaned_coords  = _subset_list(coordinates, surviving_indices)
    cleaned_labels  = (_subset_list(part_labels, surviving_indices)
                       if len(part_labels) == n_original else [])
    cleaned_colors  = (_subset_list(colors, surviving_indices)
                       if (colors and len(colors) == n_original) else None)

    cleaned_entry = {
        "coordinates": cleaned_coords,
        "part_label":  cleaned_labels,
        "cls_label":   entry.get("cls_label", ""),
        "batch_num":   entry.get("batch_num", ""),
        "ransac_meta": meta,
    }
    if cleaned_colors is not None:
        cleaned_entry["colors"] = cleaned_colors

    # Preserve any other keys the entry might have (e.g. source_ply, arrows)
    for extra_key in entry:
        if extra_key not in cleaned_entry:
            cleaned_entry[extra_key] = entry[extra_key]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump([cleaned_entry], fh, indent=2)

    n_final = len(cleaned_coords)
    pct_removed = 100.0 * (n_original - n_final) / n_original
    print(f"    ✅ {n_original:,} → {n_final:,} pts  (-{pct_removed:.1f}%)")

    return {
        "file":              json_path.name,
        "skipped":           False,
        "is_full_scene":     is_full_scene,
        "n_original":        n_original,
        "n_final":           n_final,
        "n_removed_ransac":  meta["ransac_removed"],
        "n_removed_stat":    meta["statistical_removed"],
        "pct_removed":       round(pct_removed, 1),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RANSAC + statistical outlier cleaning for point cloud JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir",  default=None,
                        help="Path to static/ directory (auto-detected if omitted).")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: static/ransac_cleaned/).")
    parser.add_argument("--nb_neighbors", type=int,   default=20,
                        help="Neighbours for statistical outlier removal.")
    parser.add_argument("--std_ratio",    type=float, default=2.0,
                        help="Std-dev multiplier for statistical outlier removal.")
    parser.add_argument("--max_removal_fraction", type=float, default=0.60,
                        help="Safety: skip RANSAC if it would remove more than this fraction.")
    parser.add_argument("--full_scene_only", action="store_true",
                        help="Only process files whose name contains 'full_scene'.")
    args = parser.parse_args()

    input_dir, output_dir = _resolve_dirs(args.input_dir, args.output_dir)

    # Collect JSON files
    all_json = sorted(input_dir.glob("*_points.json"))

    # Exclude files already inside ransac_cleaned/ (in case input_dir == output_dir)
    all_json = [p for p in all_json if "ransac_cleaned" not in str(p)]

    if args.full_scene_only:
        all_json = [p for p in all_json if "full_scene" in p.name]

    if not all_json:
        sys.exit("❌  No *_points.json files found in the input directory.")

    print(f"\n🔍 Found {len(all_json)} point cloud file(s)")
    print(f"📂 Input:  {input_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"⚙️  Settings: nb_neighbors={args.nb_neighbors}, "
          f"std_ratio={args.std_ratio}, "
          f"max_removal_fraction={args.max_removal_fraction:.0%}")
    print("=" * 70)

    summaries = []
    for i, json_path in enumerate(all_json, 1):
        is_full_scene = "full_scene" in json_path.name
        tag = "full_scene" if is_full_scene else "single obj"
        print(f"\n[{i}/{len(all_json)}]  {json_path.name}  ({tag})")

        output_path = output_dir / json_path.name
        summary = process_file(
            json_path, output_path,
            is_full_scene=is_full_scene,
            nb_neighbors=args.nb_neighbors,
            std_ratio=args.std_ratio,
            max_removal_fraction=args.max_removal_fraction,
        )
        summaries.append(summary)

    # ------------------------------------------------------------------
    # Final report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📊  SUMMARY")
    print("=" * 70)

    processed = [s for s in summaries if not s.get("skipped")]
    skipped   = [s for s in summaries if s.get("skipped")]
    full_scene = [s for s in processed if s.get("is_full_scene")]
    single_obj = [s for s in processed if not s.get("is_full_scene")]

    total_before = sum(s["n_original"] for s in processed)
    total_after  = sum(s["n_final"]    for s in processed)
    total_removed = total_before - total_after
    overall_pct = 100.0 * total_removed / total_before if total_before else 0.0

    print(f"  Files processed:  {len(processed)}  "
          f"({len(full_scene)} full-scene, {len(single_obj)} single-object)")
    if skipped:
        print(f"  Files skipped:    {len(skipped)}")

    if full_scene:
        fs_before = sum(s["n_original"] for s in full_scene)
        fs_after  = sum(s["n_final"]    for s in full_scene)
        fs_pct    = 100.0 * (fs_before - fs_after) / fs_before
        print(f"\n  Full-scene files ({len(full_scene)} files):")
        print(f"    Before: {fs_before:>10,} pts")
        print(f"    After:  {fs_after:>10,} pts  (-{fs_pct:.1f}%)")
        fs_ransac = sum(s["n_removed_ransac"] for s in full_scene)
        fs_stat   = sum(s["n_removed_stat"]   for s in full_scene)
        print(f"    Removed by RANSAC:      {fs_ransac:>8,} pts")
        print(f"    Removed by stat filter: {fs_stat:>8,} pts")

    if single_obj:
        so_before = sum(s["n_original"] for s in single_obj)
        so_after  = sum(s["n_final"]    for s in single_obj)
        so_pct    = 100.0 * (so_before - so_after) / so_before
        print(f"\n  Single-object files ({len(single_obj)} files):")
        print(f"    Before: {so_before:>10,} pts")
        print(f"    After:  {so_after:>10,} pts  (-{so_pct:.1f}%)")

    print(f"\n  Total points before: {total_before:>12,}")
    print(f"  Total points after:  {total_after:>12,}  (-{overall_pct:.1f}%)")
    print(f"\n📁 Cleaned files saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
