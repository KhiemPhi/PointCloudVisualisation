import numpy as np 
import matplotlib.pyplot as plt
from numpy import linalg as la
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import open3d as o3d
import string
from .point_cloud import PointCloud
from .occupancy import OccupancyStruct
import os 
from numba import njit
from itertools import groupby

def build_cloud_object(pcd, gripper_width, gripper_height):
    cloud_object = PointCloud()
    cloud_object.processed_cloud = pcd
    
    import numpy as np
    from sklearn.decomposition import PCA

   
    
    cloud_object.points = np.asarray(cloud_object.processed_cloud.points)
    # if cloud_object.points.shape[0] >= 3:
    #     pca = PCA(n_components=3)
    #     pca.fit(cloud_object.points)
    #     cloud_object.pca_mean = np.mean(cloud_object.points, axis=0)
    #     cloud_object.pca_components = pca.components_
    #     cloud_object.points = np.dot(cloud_object.points - cloud_object.pca_mean, cloud_object.pca_components.T)
    #     cloud_object.processed_cloud.points = o3d.utility.Vector3dVector(cloud_object.points)

    # Computing the normals for this point cloud:
    cloud_object.processed_cloud.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    cloud_object.processed_cloud.estimate_normals()
    cloud_object.processed_cloud.orient_normals_consistent_tangent_plane(30)
    cloud_object.normals_base_frame = np.asarray(cloud_object.processed_cloud.normals)
    
    # Specifying gripper tolerances:
    cloud_object.gripper_width_tolerance = gripper_width
    cloud_object.gripper_height_tolerance = gripper_height

    # Computing the bounding boxes corresponding to the object point cloud: 
    cloud_object.compute_bounding_box()
    cloud_object.vertices = cloud_object.oriented_bounding_box_vertices
    cloud_object.plot_cube()

    return cloud_object



def get_face_distance(faces, object_frame_points, cloud_object=None):
    """
    Calculate the distance between selected bbox faces and points of an object in a cloud point.

    Args:
    - faces (list): A list of detected faces, each represented by a set [x1,y1,x2,y2]
    - object_frame_points (list): The coordinates of the object in the frame.
    - cloud_object (point_cloud): The point cloud representing the object.

    Returns:
    - distances (list): A list of distances between each selected bbox face and the specified object.
    """
    center_coordinates = np.array([np.mean(face, axis=0) for face in faces])
    threshold = 1e-10
    center_coordinates[np.abs(center_coordinates) < threshold] = 0
    
    face_distances = []
    unit_vecs = []
   
    for point in object_frame_points:
        distances = []
        unit_vec = []
        for face, center in zip(faces, center_coordinates):
            distance, unit_u = get_plane_distance(face, center, point)
            distances.append(distance)
            unit_vec.append(unit_u)
        face_distances.append(distances)
        unit_vecs.append(unit_vec)

    face_distances = np.asarray(face_distances)  
    unit_vecs = np.stack(unit_vecs)   
    return face_distances, unit_vecs

def get_faces(cloud_object): 
    left_face = [  cloud_object.transformed_vertices_object_frame[1], cloud_object.transformed_vertices_object_frame[0], cloud_object.transformed_vertices_object_frame[3], cloud_object.transformed_vertices_object_frame[6]    ]
    right_face = [ cloud_object.transformed_vertices_object_frame[2],  cloud_object.transformed_vertices_object_frame[7],  cloud_object.transformed_vertices_object_frame[4],  cloud_object.transformed_vertices_object_frame[5] ]
    
    front_face = [  cloud_object.transformed_vertices_object_frame[0], cloud_object.transformed_vertices_object_frame[2], cloud_object.transformed_vertices_object_frame[5], cloud_object.transformed_vertices_object_frame[3]    ]
    back_face = [ cloud_object.transformed_vertices_object_frame[7],  cloud_object.transformed_vertices_object_frame[1],  cloud_object.transformed_vertices_object_frame[6],  cloud_object.transformed_vertices_object_frame[4] ]

    top_face = [ cloud_object.transformed_vertices_object_frame[1],  cloud_object.transformed_vertices_object_frame[0],  cloud_object.transformed_vertices_object_frame[2],  cloud_object.transformed_vertices_object_frame[7] ]
    bottom_face = [ cloud_object.transformed_vertices_object_frame[6],  cloud_object.transformed_vertices_object_frame[3],  cloud_object.transformed_vertices_object_frame[5],  cloud_object.transformed_vertices_object_frame[4] ]
    
    front_face = np.asarray(front_face)
    back_face = np.asarray(back_face)

    left_face = np.asarray(left_face) # this is front face
    right_face = np.asarray(right_face) # this is front face

    top_face = np.asarray(top_face) # some bugs here
    bottom_face = np.asarray(bottom_face) # some bugs here
    
    return left_face, right_face, front_face, back_face, top_face, bottom_face


def slice_bbox(vertices, cloud_object, thickness, slice_idx):
    """
    Partition a 3D bounding box (8x3 vertices) into multiple slices along a given axis.

    Parameters:
    - vertices: (8,3) array of bbox corners.
    - cloud_object: (unused in current function, kept for API consistency).
    - thickness: step size along the slicing axis.
    - slice_idx: 0, 1, or 2 → axis along which to slice.

    Returns:
    - List of (8,3) numpy arrays, one per sliced bbox.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    assert vertices.shape == (8, 3), "vertices must be shape (8, 3)"

    # Axis bounds
    start_value = np.min(vertices[:, slice_idx])
    end_value = np.max(vertices[:, slice_idx])
    direction = np.sign(thickness)
    stop_value = end_value + direction * np.finfo(float).eps

    # Compute slice boundaries
    steps = np.arange(start_value, stop_value + thickness, thickness)
    n_slices = len(steps) - 1

    # Vectorized creation of all slices
    low_vals = steps[:-1][:, None]
    high_vals = steps[1:][:, None]

    # Create (n_slices, 8, 3) array of boxes
    all_boxes = np.repeat(vertices[None, :, :], n_slices, axis=0)

    # Replace the slice coordinate per box
    start_mask = np.isclose(vertices[:, slice_idx], start_value)
    end_mask = ~start_mask

    # Assign low/high bounds efficiently
    all_boxes[:, start_mask, slice_idx] = low_vals
    all_boxes[:, end_mask, slice_idx] = high_vals

    # Return as list of (8,3) arrays
    return [all_boxes[i] for i in range(n_slices)]


# def points_inside_bbox(points, bbox_min, bbox_max):
#     """
#     Check which points are inside a bounding box.

#     Parameters:
#     - points: List of 3D points where each point is represented as [x, y, z].
#     - bbox_min: Minimum coordinates of the bounding box, e.g., [min_x, min_y, min_z].
#     - bbox_max: Maximum coordinates of the bounding box, e.g., [max_x, max_y, max_z].

#     Returns:
#     - List of boolean values indicating whether each point is inside the bounding box.
#     """
#     return points[np.where(np.all((bbox_min <= points) & (points <= bbox_max), axis=1)==True)], np.where(np.all((bbox_min <= points) & (points <= bbox_max), axis=1)==True)

@njit
def points_inside_bbox(points, bbox_min, bbox_max):
    n = points.shape[0]
    mask = np.empty(n, dtype=np.bool_)
    for i in range(n):
        mask[i] = (bbox_min[0] <= points[i,0] <= bbox_max[0] and
                   bbox_min[1] <= points[i,1] <= bbox_max[1] and
                   bbox_min[2] <= points[i,2] <= bbox_max[2])
    return points[mask], np.where(mask)[0]

def organize_coordinates(coordinates):
    left_face = [  coordinates[1], coordinates[0], coordinates[3], coordinates[6]    ]
    right_face = [ coordinates[2],  coordinates[7],  coordinates[4],  coordinates[5] ]
    
    front_face = [  coordinates[0], coordinates[2], coordinates[5], coordinates[3]    ]
    back_face = [ coordinates[7],  coordinates[1],  coordinates[6],  coordinates[4] ]

    top_face = [ coordinates[1],  coordinates[0],  coordinates[2],  coordinates[7] ]
    bottom_face = [ coordinates[6],  coordinates[3],  coordinates[5],  coordinates[4] ]
    
    front_face = np.asarray(front_face)
    back_face = np.asarray(back_face)

    left_face = np.asarray(left_face) # this is front face
    right_face = np.asarray(right_face) # this is front face

    top_face = np.asarray(top_face) # some bugs here
    bottom_face = np.asarray(bottom_face) # some bugs here
    
    return [left_face, right_face, front_face, back_face, top_face, bottom_face]
    


def accumulate_slices(bbox_vertices, object_frame_points, face_distances, epsilon, ax2=None): 
     #3. Calculate Fill Ratio
    ibr_ratios = []
    points_per_slice = []
    empty_ratios = []
    point_slice_idx = []
    for idx, vertices in enumerate(bbox_vertices):
        
        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)        
        
        points_inside=[]
        points_idx = []
       
      
        
        points_inside, points_idx = points_inside_bbox(object_frame_points, bbox_min, bbox_max) 
        
               
        slice_dist = face_distances[points_idx]
        slice_dist = np.min(slice_dist, axis=1)
        
        
        boundary_idx = np.where(slice_dist < epsilon)[0]
        interior_idx = np.where(slice_dist >= epsilon)[0] 
        grid_vertices = np.hstack([  np.min(vertices, axis=0), np.max(vertices, axis=0) ])
        
        occupancy_struct = OccupancyStruct(grid_vertices, object_frame_points, 0.01)
        empty_ratio = occupancy_struct.occupied_comps
        empty_ratios.append(empty_ratio)
        
        total = len(boundary_idx) + len(interior_idx)
        ibr_ratio = len(boundary_idx) / total if total > 0 else 0
        ibr_ratio = ibr_ratio * empty_ratio

        if len(points_inside) > 0:
            points_per_slice.append(points_inside)
            point_slice_idx.append(points_idx)
            if len(interior_idx) > 0 and len(boundary_idx) <= len(interior_idx):
                ibr_ratios.append(ibr_ratio)
            elif len(interior_idx) == 0:
                ibr_ratios.append(ibr_ratio) 
            else: 
                ibr_ratios.append(ibr_ratio) 
        else: 
            points_per_slice.append([])
            ibr_ratios.append(-1) 
        
    
    
    return ibr_ratios, empty_ratios, points_per_slice, point_slice_idx

# def fuse_slices(points_per_slice, ibr_ratios, empty_ratios, theta, thickness, slice_idx, gripper_width): 
#     point_groups = [ [points_per_slice[0]] ] # init to first location of non-empty points and get that idx
#     group_idx = 0
#     empty_hit = False
    
#     #3. Group Fusion
    
#     for ratio_idx in range(len(ibr_ratios) - 1):
#         pair_diff = (ibr_ratios[ratio_idx]+empty_ratios[ratio_idx+1]) - (ibr_ratios[ratio_idx + 1]+empty_ratios[ratio_idx+1])
       
#         if np.abs(pair_diff) >= theta or ( (len(point_groups[group_idx])+1)*thickness > gripper_width  ) or len(points_per_slice[ratio_idx]) == 0: # add to current group if no sharp-change and no exceeding grip and not too small 
            
#             if len(points_per_slice[ratio_idx]) > 0:
#                 point_groups[group_idx].append(points_per_slice[ratio_idx])
                
#                 group_idx += 1 
#                 point_groups.append([])
#                 empty_hit = False
#             elif empty_hit == False:                 
#                 empty_hit = True

            
#             if ratio_idx < len(ibr_ratios) - 1 and len(points_per_slice[ratio_idx+1]) > 0:
#                 point_groups[group_idx].append(points_per_slice[ratio_idx+1])
            
#         else: 
#             point_groups[group_idx].append(points_per_slice[ratio_idx])
#             if ratio_idx == len(ibr_ratios) - 1 and len(points_per_slice[ratio_idx+1]) > 0:
#                 point_groups[group_idx].append(points_per_slice[ratio_idx+1])
    
   
#     return point_groups

@njit
def fuse_slices_fast(ibr_ratios, empty_ratios, theta, thickness, gripper_width):
    n = len(ibr_ratios)
    group_flags = np.zeros(n, dtype=np.int32)
    group_idx = 0
    current_width = 0.0

    for i in range(n - 1):
        pair_diff = (ibr_ratios[i] + empty_ratios[i+1]) - (ibr_ratios[i+1] + empty_ratios[i+1])
        sharp_change = abs(pair_diff) >= theta
        too_wide = current_width + thickness > gripper_width

        if sharp_change or too_wide:
            group_idx += 1
            current_width = 0.0
        else:
            current_width += thickness

        group_flags[i] = group_idx

    return group_flags

def fuse_slices(points_per_slice, ibr_ratios, empty_ratios, theta, thickness, slice_idx, gripper_width):
    group_flags = fuse_slices_fast(ibr_ratios, empty_ratios, theta, thickness, gripper_width)
    point_groups = []
    for _, group in groupby(zip(group_flags, points_per_slice), key=lambda x: x[0]):
        pts = [p for _, p in group if len(p) > 0]
        if pts:
            point_groups.append(pts)
    return point_groups

def build_new_cloud_bbox(group, cloud_object):  
    group = list(filter(lambda x: len(x) > 0, group)) 
    points = np.vstack(group)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.paint_uniform_color([0, 0, 1])

    # Creating the cloud object and loading the necessary file:
    cloud_object_part = PointCloud()
    cloud_object_part.processed_cloud = pcd
    cloud_object_part.points = np.asarray(cloud_object_part.processed_cloud.points)
    cloud_object_part.compute_bounding_box()
    cloud_object_part.vertices = cloud_object_part.oriented_bounding_box_vertices
    cloud_object_part.gripper_width_tolerance = cloud_object.gripper_width_tolerance
    cloud_object_part.gripper_height_tolerance = cloud_object.gripper_height_tolerance
    cloud_object_part.plot_cube()
    return cloud_object_part

def refit_bbox(cloud_object, point_groups):
    parts = []
    for idx, group in enumerate(point_groups): 
        num_points = np.vstack(group).shape[0]
        if len(group) > 0 and num_points > 100:
            try:
                cloud_object_part = build_new_cloud_bbox(group, cloud_object)
            except: 
                breakpoint()
            parts.append(cloud_object_part)   
    return parts


def calculate_face_area(vertices):
    # Convert vertices to NumPy arrays
    v1 = np.array(vertices[1]) - np.array(vertices[0])
    v2 = np.array(vertices[2]) - np.array(vertices[0])
    v3 = np.array(vertices[3]) - np.array(vertices[0])

    # Calculate the cross products for the two triangles
    cross_product1 = np.cross(v1, v2)
    cross_product2 = np.cross(v2, v3)

    # Calculate the areas of the two triangles
    area = 0.5 * (np.linalg.norm(cross_product1) + np.linalg.norm(cross_product2))

    return area



def get_ibr_ratio(cloud_object, epsilon):
    left_face, right_face, front_face, back_face, top_face, bottom_face = get_faces(cloud_object)
    faces = [left_face, right_face, front_face, back_face, top_face, bottom_face]
    object_frame_points = np.asarray(cloud_object.cloud_object_frame.points)   
    face_distances, unit_vec = get_face_distance(faces, object_frame_points, cloud_object) 

    #1. Select Slicing Dimesions
    boundary_idx = np.where(np.any(face_distances <= epsilon, axis=1))[0]
    interior_idx =np.where(np.any(face_distances > epsilon, axis=1))[0]
    ibr_ratio = len(boundary_idx) / len(interior_idx)
    vertices = cloud_object.oriented_bounding_box_vertices
    grid_vertices = np.hstack([  np.min(vertices, axis=0), np.max(vertices, axis=0) ])
    occupancy_struct = OccupancyStruct(grid_vertices, cloud_object.points, 0.01)
    empty_ratio = occupancy_struct.weighted_empty_percentage  
    ibr_ratio = ibr_ratio * empty_ratio
    
    
    return ibr_ratio  



def fuse_consecutive_boxes(cloud_object_parts, cloud_object, slice_idx, area_threshold=0.8, gripper_length=0.11, resolution=0.01):
    fused_boxes = [cloud_object_parts[0]]
    merged_indices = []
    fused_pointer = 0

    for i in range(1, len(cloud_object_parts)):
        curr_left_face, curr_right_face, curr_front_face, curr_back_face, curr_top_face, curr_bottom_face = get_faces(cloud_object_parts[i])
        prev_left_face, prev_right_face, prev_front_face, prev_back_face, prev_top_face, prev_bottom_face = get_faces(fused_boxes[fused_pointer])  
        prev_points = np.asarray(fused_boxes[fused_pointer].processed_cloud.points)

        if slice_idx == 0:                
            area_diff =  calculate_face_area(curr_back_face) -  calculate_face_area(prev_front_face) 
        elif slice_idx == 1: 
            area_diff =  calculate_face_area(curr_left_face) - calculate_face_area(prev_right_face) 
        elif slice_idx == 2:   
            area_diff = calculate_face_area(curr_bottom_face) - calculate_face_area(prev_top_face)  
   
        if np.abs(area_diff) <= area_threshold: 
            new_points = [np.asarray(cloud_object_parts[i].processed_cloud.points), prev_points] 
            fused_object = build_new_cloud_bbox(new_points, cloud_object)
            # check empty
            empty_ratio, occupancy_struct = get_empty_ratio(fused_object, resolution)
            # if occupancy_struct.occupied_comps == 1 and (fused_object.x_dim < gripper_length or fused_object.y_dim < gripper_length or fused_object.z_dim < gripper_length):           
            #     # Either choose the fused object or just appent the parts
            #     fused_boxes[fused_pointer] = fused_object
            #     merged_indices.append(i-1)
            #     merged_indices.append(i)
            # else: 
            #     fused_boxes.append(cloud_object_parts[i])
            #     fused_pointer += 1
            fused_boxes[fused_pointer] = fused_object
            merged_indices.append(i-1)
            merged_indices.append(i)
        
           
        else: 
            fused_boxes.append(cloud_object_parts[i])
            fused_pointer += 1    
    
    if len(fused_boxes) > 0 and len(fused_boxes) < len(cloud_object_parts):
        new_points = [np.asarray(cloud_object_parts[-1].processed_cloud.points), prev_points] 
        fused_object = build_new_cloud_bbox(new_points, cloud_object)
        fused_boxes[fused_pointer] = fused_object
        merged_indices.append(i-1)
        merged_indices.append(i)

       
    
    return fused_boxes
       
def slice_subparts(part, cloud_object, face_distances, uncheck_idx, thickness, epsilon, theta,  area_threshold, gripper_length, resolution):
    vertices = part.oriented_bounding_box_vertices
    left_face, right_face, front_face, back_face, top_face, bottom_face = get_faces(part)
    faces = [left_face, right_face, front_face, back_face, top_face, bottom_face]
    object_frame_points = np.asarray(part.points)   
    face_distances, unit_vec = get_face_distance(faces, object_frame_points, cloud_object)
    sub_slice_bbox_vertices = slice_bbox(vertices, part, thickness=thickness, slice_idx=uncheck_idx) 
    object_frame_points = np.asarray(part.points)   
    sub_ibr_ratios, empty_ratios, sub_point_groups, sub_point_slice_idx = accumulate_slices(sub_slice_bbox_vertices, object_frame_points, face_distances, epsilon)
    sub_point_groups                  = fuse_slices(sub_point_groups, sub_ibr_ratios, empty_ratios,  theta, thickness, slice_idx=uncheck_idx, gripper_width=gripper_length)
   
    sub_parts = refit_bbox(cloud_object, sub_point_groups)
    sub_parts = fuse_consecutive_boxes(sub_parts, cloud_object, slice_idx=uncheck_idx, area_threshold=area_threshold, resolution=resolution)
    return sub_parts, sub_point_groups, sub_point_slice_idx     

def get_empty_ratio(part, resolution=0.01):
    vertices = part.oriented_bounding_box_vertices
    grid_vertices = np.hstack([  np.min(vertices, axis=0), np.max(vertices, axis=0) ])
    occupancy_struct = OccupancyStruct(grid_vertices, part.points, resolution)
    empty_ratio = occupancy_struct.weighted_empty_percentage  
    return empty_ratio, occupancy_struct

def oversize_correct(parts, point_groups,ibr_ratios, cloud_object, face_distances, uncheck_idx, thickness, epsilon, theta, base_ibr, area_threshold, gripper_length, resolution=0.01):
   
    fixed_parts = []
    fixed_point_groups = []
    
    for idx, ratio in enumerate(ibr_ratios): 
        empty_ratio, occupancy_struct = get_empty_ratio(parts[idx], resolution)
        dimensions = np.array( [parts[idx].x_dim, parts[idx].y_dim, parts[idx].z_dim ])
        uncheck_dimensions = dimensions[uncheck_idx]
        new_slice_idx = uncheck_idx[np.argmax(uncheck_dimensions)]
      
        if ( (parts[idx].x_dim > gripper_length or parts[idx].y_dim > gripper_length) or occupancy_struct.occupied_comps > 1 or ratio < base_ibr  )    :
            
            sub_parts, sub_point_groups = slice_subparts(parts[idx], cloud_object, face_distances, new_slice_idx, thickness, epsilon, theta, area_threshold, gripper_length, resolution)
            for group in sub_point_groups: 
                fixed_point_groups.append(group)
            for part in sub_parts:
                fixed_parts.append(part)
        else: 
            fixed_parts.append(parts[idx])
            fixed_point_groups.append(point_groups[idx])
       
    
    return fixed_parts, fixed_point_groups


def search_dimesion_slice(cloud_object, gripper_length, ibr_ratio, base_ibr):
    graspable_dim = np.array([cloud_object.x_dim, cloud_object.y_dim, cloud_object.z_dim])
    depth_dim = cloud_object.z_dim
    
   
    slice_idx = np.argmax(graspable_dim)
    if slice_idx == 2: 
        uncheck_idx = [0,1]
    else:
        uncheck_idx = [np.int64(np.logical_not(np.argmax(graspable_dim)))]

    
   
    return slice_idx, uncheck_idx

def get_overall_ibr_ratio(cloud_object, epsilon=1e-3, thickness=0.001, gripper_length=0.13, base_ibr=0.7):
    left_face, right_face, front_face, back_face, top_face, bottom_face = get_faces(cloud_object)
    faces = [left_face, right_face, front_face, back_face, top_face, bottom_face]
    object_frame_points = np.asarray(cloud_object.cloud_object_frame.points)   
    face_distances, unit_vec = get_face_distance(faces, object_frame_points, cloud_object) 
    vertices = cloud_object.transformed_vertices_object_frame
    boundary_idx = np.where(np.any(face_distances <= epsilon, axis=1))[0]
    interior_idx =np.where(np.any(face_distances > epsilon, axis=1))[0]
    total = len(boundary_idx) + len(interior_idx)
    ibr_ratio = len(boundary_idx) / total if total > 0 else 0

    return ibr_ratio

def bitsi_metric (cloud_object, epsilon=1e-3, thickness=0.001, gripper_length=0.13, base_ibr=0.7):
    
    left_face, right_face, front_face, back_face, top_face, bottom_face = get_faces(cloud_object)
    faces = [left_face, right_face, front_face, back_face, top_face, bottom_face]
    object_frame_points = np.asarray(cloud_object.cloud_object_frame.points)   
    face_distances, unit_vec = get_face_distance(faces, object_frame_points, cloud_object) 
    vertices = cloud_object.transformed_vertices_object_frame

    boundary_idx = np.where(np.any(face_distances <= epsilon, axis=1))[0]
    interior_idx =np.where(np.any(face_distances > epsilon, axis=1))[0]
    ibr_ratio = len(boundary_idx) / len(interior_idx)
    
    slice_idx, uncheck_idx  = search_dimesion_slice(cloud_object, gripper_length, ibr_ratio, base_ibr)
       
    slice_bbox_vertices_x = slice_bbox(vertices, cloud_object, thickness=thickness, slice_idx=0)
    slice_bbox_vertices_y = slice_bbox(vertices, cloud_object, thickness=thickness, slice_idx=1)
    slice_bbox_vertices_z = slice_bbox(vertices, cloud_object, thickness=thickness, slice_idx=2)

    ibr_ratios_x, _, points_per_slice_x, point_slice_idx_x = accumulate_slices(slice_bbox_vertices_x, object_frame_points, face_distances, epsilon)
    ibr_ratios_y, _, points_per_slice_y, point_slice_idx_y = accumulate_slices(slice_bbox_vertices_y, object_frame_points, face_distances, epsilon)
    ibr_ratios_z, _, points_per_slice_z, point_slice_idx_z = accumulate_slices(slice_bbox_vertices_z, object_frame_points, face_distances, epsilon)
    
    points_per_slice_x = [cloud_object.original_world_points[idx] for idx in point_slice_idx_x]
    points_per_slice_y = [cloud_object.original_world_points[idx] for idx in point_slice_idx_y]
    points_per_slice_z = [cloud_object.original_world_points[idx] for idx in point_slice_idx_z]
    
  

    return ibr_ratios_x, ibr_ratios_y, ibr_ratios_z, slice_idx, points_per_slice_x, points_per_slice_y, points_per_slice_z


def slice_object_bbox(cloud_object, gripper_length=0.11, epsilon=1e-3, thickness=0.001, theta=0.1, base_ibr=0.7, area_threshold=1e-4, resolution=0.01, check=False, obj_name="" ):
    
    left_face, right_face, front_face, back_face, top_face, bottom_face = get_faces(cloud_object)
    faces = [left_face, right_face, front_face, back_face, top_face, bottom_face]
    object_frame_points = np.asarray(cloud_object.cloud_object_frame.points)   
    face_distances, unit_vec = get_face_distance(faces, object_frame_points, cloud_object) 
    vertices = cloud_object.transformed_vertices_object_frame

    #1. Select Slicing Dimesions
    boundary_idx = np.where(np.any(face_distances <= epsilon, axis=1))[0]
    interior_idx =np.where(np.any(face_distances > epsilon, axis=1))[0]
    ibr_ratio = len(boundary_idx) / len(interior_idx)
    
   
    slice_idx, uncheck_idx  = search_dimesion_slice(cloud_object, gripper_length, ibr_ratio, base_ibr)
    if slice_idx == None: 
        return None, None  
    
    
    
    #1. Calculate Fill Ratio and Collect Slices
    slice_bbox_vertices = slice_bbox(vertices, cloud_object, thickness=thickness, slice_idx=slice_idx) 
    
    
    
    ibr_ratios, empty_ratios, points_per_slice = accumulate_slices(slice_bbox_vertices, object_frame_points, face_distances, epsilon)
    point_groups                  = fuse_slices(points_per_slice, ibr_ratios, empty_ratios, theta, thickness, slice_idx, gripper_length)
    #2. Fit New Bounding Box
    parts = refit_bbox(cloud_object, point_groups)
    #3. Fuse Parts
    parts = fuse_consecutive_boxes(parts, cloud_object, slice_idx, area_threshold=area_threshold)
    ibr_ratios = [ get_ibr_ratio(part, epsilon=epsilon) for part in parts ] 
    #4. Correction Refinment Step
    if slice_idx != 2:
       parts, point_groups = oversize_correct(parts, point_groups,ibr_ratios, cloud_object, face_distances, uncheck_idx, thickness, epsilon, theta, base_ibr, area_threshold, gripper_length, resolution)
    
    return parts, point_groups, slice_bbox_vertices
  



def get_plane_distance(plane_points, center_point, grasp_center):
    # We need to compute the distance of the grasp center to the first plane only:
    vect_1 = np.subtract(plane_points[3, :], plane_points[0, :])
    vect_2 = np.subtract(plane_points[1, :], plane_points[0, :])
    # vect_2 = np.subtract(plane_points[1, :], plane_points[2, :])

    # Normal vector corresponding to the plane:
    n = np.cross(vect_1, vect_2)
    unit_u = np.divide(n, la.norm(n))

    # Translating each of the points by a specified distance: 
    # Computing the D in the equation of the plane Ax + By + Cz + D = 0:
    D = np.dot(center_point, unit_u)

    num = la.norm(unit_u[0]*grasp_center[0] + unit_u[1]*grasp_center[1] + unit_u[2]*grasp_center[2] - D)
    den = np.sqrt(unit_u[0]**2 + unit_u[1]**2 + unit_u[2]**2)
    distance = np.divide(num, den)
    return distance, unit_u



