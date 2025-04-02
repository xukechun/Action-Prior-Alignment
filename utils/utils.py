import math
import numpy as np
import pybullet as p
import cv2
import trimesh
import random
import open3d as o3d
import open3d_plus as o3dp
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from env.constants import WORKSPACE_LIMITS, PIXEL_SIZE, IMAGE_SIZE
from env.constants import PLACE_LANG_TEMPLATES, OBJ_DIRECTION, OBJ_DIRECTION_MAP, OBJ_UNSEEN_DIRECTION_MAP, MIN_DIS, MAX_DIS


reconstruction_config = {
    'nb_neighbors': 50,
    'std_ratio': 2.0,
    'voxel_size': 0.0015,
    'icp_max_try': 5,
    'icp_max_iter': 2000,
    'translation_thresh': 3.95,
    'rotation_thresh': 0.02,
    'max_correspondence_distance': 0.02
}

graspnet_config = {
    'graspnet_checkpoint_path': 'models/graspnet/logs/log_rs/checkpoint.tar',
    'refine_approach_dist': 0.01,
    'dist_thresh': 0.05,
    'angle_thresh': 15,
    'mask_thresh': 0.5
}

def get_heightmap(points, colors, bounds, pixel_size):
    """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.

    Args:
        points: HxWx3 float array of 3D points in world coordinates.
        colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
        bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
            region in 3D space to generate heightmap in world coordinates.
        pixel_size: float defining size of each pixel in meters.
    Returns:
        heightmap: HxW float array of height (from lower z-bound) in meters.
        colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
    """
    width = int(np.round((bounds[0, 1] - bounds[0, 0]) / pixel_size))
    height = int(np.round((bounds[1, 1] - bounds[1, 0]) / pixel_size))
    heightmap = np.zeros((height, width), dtype=np.float32)
    colormap = np.zeros((height, width, colors.shape[-1]), dtype=np.uint8)

    # Filter out 3D points that are outside of the predefined bounds.
    ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
    iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
    iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
    valid = ix & iy & iz
    points = points[valid]
    colors = colors[valid]

    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points, colors = points[iz], colors[iz]
    px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size))
    py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size))
    px = np.clip(px, 0, width - 1)
    py = np.clip(py, 0, height - 1)
    heightmap[px, py] = points[:, 2] - bounds[2, 0]
    for c in range(colors.shape[-1]):
        colormap[px, py, c] = colors[:, c]
    return heightmap, colormap

def get_pointcloud(depth, intrinsics):
    """Get 3D pointcloud from perspective depth image.
    Args:
        depth: HxW float array of perspective depth in meters.
        intrinsics: 3x3 float array of camera intrinsics matrix.
    Returns:
        points: HxWx3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    points = np.float32([px, py, depth]).transpose(1, 2, 0)
    return points

def get_pointcloud_w_mask(depth, mask, intrinsics):
    """Get 3D pointcloud from perspective depth image with a depth mask.
    Args:
        depth: HxW float array of perspective depth in meters.
        intrinsics: 3x3 float array of camera intrinsics matrix.
    Returns:
        points: (HxW)x3 float array of 3D points in camera coordinates.
    """
    height, width = depth.shape
    mask = np.logical_and(mask, depth > 0.0)
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)
    px = px[mask]
    py = py[mask]
    px = (px - intrinsics[0, 2]) * (depth[mask] / intrinsics[0, 0])
    py = (py - intrinsics[1, 2]) * (depth[mask] / intrinsics[1, 1])
    points = np.float32([px, py, depth[mask]]).transpose(1, 0)
    return points    

def transform_pointcloud(points, transform):
    """Apply rigid transformation to 3D pointcloud.
    Args:
        points: HxWx3 float array of 3D points in camera coordinates.
        transform: 4x4 float array representing a rigid transformation matrix.
    Returns:
        points: HxWx3 float array of transformed 3D points.
    """
    padding = ((0, 0), (0, 0), (0, 1))
    homogen_points = np.pad(points.copy(), padding, "constant", constant_values=1)
    for i in range(3):
        points[Ellipsis, i] = np.sum(transform[i, :] * homogen_points, axis=-1)
    return points

def reconstruct_heightmaps(color, depth, configs, bounds, pixel_size):
    """Reconstruct top-down heightmap views from multiple 3D pointclouds."""
    heightmaps, colormaps = [], []
    for color, depth, config in zip(color, depth, configs):
        intrinsics = config["intrinsics"]
        xyz = get_pointcloud(depth, intrinsics)
        position = np.array(config["position"]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        xyz = transform_pointcloud(xyz, transform)
        heightmap, colormap = get_heightmap(xyz, color, bounds, pixel_size)
        heightmaps.append(heightmap)
        colormaps.append(colormap)

    return heightmaps, colormaps

def get_true_heightmap(env):
    """Get RGB-D orthographic heightmaps and segmentation masks in simulation."""

    # Capture near-orthographic RGB-D images and segmentation masks.
    color, depth, segm = env.render_camera(env.oracle_cams[0])

    # Combine color with masks for faster processing.
    color = np.concatenate((color, segm[Ellipsis, None]), axis=2)

    # Reconstruct real orthographic projection from point clouds.
    hmaps, cmaps = reconstruct_heightmaps(
        [color], [depth], env.oracle_cams, env.bounds, env.pixel_size
    )

    # Split color back into color and masks.
    cmap = np.uint8(cmaps)[0, Ellipsis, :3]
    hmap = np.float32(hmaps)[0, Ellipsis]
    mask = np.int32(cmaps)[0, Ellipsis, 3:].squeeze()

    return cmap, hmap, mask

def get_camera_configs(env, single_view=False, diff_views=False):
    if single_view:
        configs = [env.agent_cams[0]]
    elif diff_views:
        configs = [env.agent_cams[3], env.agent_cams[4], env.agent_cams[5]]
    else:
        configs = [env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
    multi_camera_extrinsics = []
    multi_camera_intrinsics = []
    # multi_camera_params = []
    for config in configs:
        position = np.array(config["position"]).reshape(3, 1)
        rotation = p.getMatrixFromQuaternion(config["rotation"])
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        camera_extrinsics = np.linalg.inv(transform)
        camera_intrinsics = config["intrinsics"]
        multi_camera_extrinsics.append(camera_extrinsics)
        multi_camera_intrinsics.append(camera_intrinsics)


    multi_camera_extrinsics = np.stack(multi_camera_extrinsics, axis=0) # [N, 4, 4]
    multi_camera_intrinsics = np.stack(multi_camera_intrinsics, axis=0) # [N, 3, 3]
    # multi_camera_params = np.stack(multi_camera_params, axis=0) # [N, 4]

    return multi_camera_extrinsics, multi_camera_intrinsics

def get_multi_view_images(env, single_view=False, diff_views=False):
    if single_view:
        configs = [env.agent_cams[0]]
    elif diff_views:
        configs = [env.agent_cams[3], env.agent_cams[4], env.agent_cams[5]]
    else:
        configs = [env.agent_cams[0], env.agent_cams[1], env.agent_cams[2]]
    # Capture near-orthographic RGB-D images and segmentation masks.
    color_images = []
    depth_images = []
    mask_images = []
    for config in configs:
        color, depth, mask = env.render_camera(config)     
        color_images.append(color)
        depth_images.append(depth)
        mask_images.append(mask)
        
    color_images = np.stack(color_images, axis=0) # [N, H, W, C]
    depth_images = np.stack(depth_images, axis=0) # [N, H, W]
    mask_images = np.stack(mask_images, axis=0) # [N, H, W]
    
    return color_images, depth_images, mask_images

def generate_points_for_feature_extraction(pcd, cut_table=True, downsample_radius=0.01, visualize=False):
    pts = np.array(pcd.points)
    if cut_table:
        # cut the table
        pz = pts[..., 2]
        mask = pz > 0.001
        pts = pts[mask]
    
    processed_pcd = o3d.geometry.PointCloud()
    processed_pcd.points = o3d.utility.Vector3dVector(pts) # [N, 3]
    
    if downsample_radius > 0:
        # radius = 0.015, # resulting in around 1500 points with 15 objects, 700 points with 8 objects
        # radius = 0.003, # resulting in aournd 28000 poiints with 15 objects, 14000 points with 8 objects
        # radius = 0.01 # resulting in around 3000 points with 8 objects, around 1500 points with 8 objects
        processed_pcd = processed_pcd.voxel_down_sample(downsample_radius)
    processed_pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=0.2)

    if visualize:
        # visualization
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        o3d.visualization.draw_geometries([processed_pcd, frame])

    return np.array(processed_pcd.points)

def get_multi_view_images_w_pointcloud(env, output_mask=False, downsample=True, visualize=False, single_view=False, diff_views=False):
    # colors: [N, H, W, 3] numpy array in uint8; depths: [N, H, W] numpy array in meters
    colors, depths, masks = get_multi_view_images(env, single_view=single_view, diff_views=diff_views) 
    # intrinsics: [N, 3, 3] numpy array; extrinsics: [N, 4, 4] numpy array, masks: [N, H, W] numpy array in bool
    extrinsics, intrinsics = get_camera_configs(env, diff_views=diff_views)
    colors_ = colors / 255.
    # visualize scaled COLMAP poses
    pcds = []
    for i in range(colors_.shape[0]):
        depth = depths[i]
        color = colors_[i]
        K = intrinsics[i]
        mask = depth > 0.0
        # mask = np.ones_like(depth, dtype=bool)
        pcd = get_pointcloud_w_mask(depth, mask, K)
        
        pose = extrinsics[i]
        pose = np.linalg.inv(pose)
        trans_pcd = pose @ np.concatenate([pcd.T, np.ones((1, pcd.shape[0]))], axis=0)
        trans_pcd = trans_pcd[:3, :].T

        # clip the pcd with worksapce boundary        
        trans_pcd_mask = (trans_pcd[:, 0] > env.bounds[0, 0]) &\
            (trans_pcd[:, 0] < env.bounds[0, 1]) &\
                (trans_pcd[:, 1] > env.bounds[1, 0]) &\
                    (trans_pcd[:, 1] < env.bounds[1, 1]) &\
                        (trans_pcd[:, 2] > env.bounds[2, 0]) &\
                            (trans_pcd[:, 2] < env.bounds[2, 1])
        
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(trans_pcd[trans_pcd_mask]) # [N, 3]
        assert color.max() <= 1
        assert color.min() >= 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(color[mask][trans_pcd_mask]) # [N, 3]

        # downsample
        if downsample:
            radius = 0.0015
            pcd_o3d = pcd_o3d.voxel_down_sample(radius)
            # pcd_o3d = pcd_o3d.voxel_down_sample(reconstruction_config['voxel_size'])
        pcds.append(pcd_o3d)

    aggr_pcd = o3d.geometry.PointCloud()
    for pcd in pcds:
        aggr_pcd += pcd
    if downsample:
        radius = 0.0015
        aggr_pcd = aggr_pcd.voxel_down_sample(radius)
    aggr_pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=0.2)
    
    if visualize:
        # visualization
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        o3d.visualization.draw_geometries([aggr_pcd, frame])

    if output_mask:
        return colors, depths, masks, aggr_pcd
    else:
        return colors, depths, aggr_pcd   
 
def filter_pcd(pcd, range, visualize=False):
    points = np.array(pcd.points)
    colors = np.array(pcd.colors)
    
    np.asarray([[0.276, 0.724], [-0.000, 0.336], [-0.0001, 0.4]])

    # Filter out 3D points that are outside of the predefined bounds.
    ix = (points[Ellipsis, 0] >= range[0, 0]) & (points[Ellipsis, 0] < range[0, 1])
    iy = (points[Ellipsis, 1] >= range[1, 0]) & (points[Ellipsis, 1] < range[1, 1])
    iz = (points[Ellipsis, 2] >= range[2, 0]) & (points[Ellipsis, 2] < range[2, 1])

    valid = ix & iy & iz
    points = points[valid]
    colors = colors[valid]
    # Sort 3D points by z-value, which works with array assignment to simulate
    # z-buffering for rendering the heightmap image.
    iz = np.argsort(points[:, -1])
    points = points[iz]
    colors = colors[iz]

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points) 
    filtered_pcd.colors = o3d.utility.Vector3dVector(colors) 
    
    if visualize: 
        # visualization
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        o3d.visualization.draw_geometries([filtered_pcd, frame])
    
    return filtered_pcd

def generate_drop_positions(num_obj, workspace_limits, min_dist=0.15):
    def distance(p1, p2):
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    drop_positions = []
    tried_times = 0
    while len(drop_positions) < num_obj:
        new_position = (random.uniform(workspace_limits[0][0] + 0.05, workspace_limits[0][1] - 0.05), random.uniform(workspace_limits[1][0] + 0.05, workspace_limits[1][1] - 0.05))
        
        if all(distance(new_position, p) > min_dist for p in drop_positions):
            drop_positions.append(new_position)

        tried_times += 1
        if tried_times - num_obj >= 50:
            print("invalid position generations")
            break
        
    return drop_positions    

def get_true_bbox_of_obj_id(mask_image, obj_id):
    mask = np.zeros(mask_image.shape).astype(np.uint8)
    mask[mask_image == obj_id] = 255
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    stats = stats[stats[:,4].argsort()]
    bbox_pixel = None
    if stats[:-1].shape[0] > 0:
        bbox = stats[:-1][0]
        # for bbox
        # |(y0, x0)         |   
        # |                 |
        # |                 |
        # |         (y1, x1)|
        x0, y0 = bbox[0], bbox[1]
        x1 = bbox[0] + bbox[2]
        y1 = bbox[1] + bbox[3]

        bbox_pixel = [y0, y1, x0, x1]

    return bbox_pixel

def get_true_bboxes(env, color_image, depth_image, mask_image, obj_filter=None):
    # get mask of all objects
    bbox_ids = []
    bbox_images = []
    bbox_positions = []
    bbox_centers = []
    bbox_sizes = []
    for obj_id in env.obj_ids["rigid"]:
        if obj_filter is not None:
            if obj_id not in obj_filter:
                continue
        mask = np.zeros(mask_image.shape).astype(np.uint8)
        mask[mask_image == obj_id] = 255
        _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        stats = stats[stats[:,4].argsort()]
        if stats[:-1].shape[0] > 0:
            bbox = stats[:-1][0]
            # for bbox
            # |(y0, x0)         |   
            # |                 |
            # |                 |
            # |         (y1, x1)|
            x0, y0 = bbox[0], bbox[1]
            x1 = bbox[0] + bbox[2]
            y1 = bbox[1] + bbox[3]

            # visualization
            start_point, end_point = (x0, y0), (x1, y1)
            color = (0, 0, 255) # Red color in BGR；红色：rgb(255,0,0)
            thickness = 1 # Line thickness of 1 px 
            mask_BGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # 转换为3通道图，使得color能够显示红色。
            mask_bboxs = cv2.rectangle(mask_BGR, start_point, end_point, color, thickness)
            cv2.imwrite('mask_bboxs.png', mask_bboxs)

            bbox_ids.append(obj_id)
            bbox_image = color_image[y0:y1, x0:x1]
            bbox_images.append(bbox_image)
            bbox_sizes.append([bbox[2], bbox[3]])
            
            pixel_x = (x0 + x1) // 2
            pixel_y = (y0 + y1) // 2
            bbox_centers.append([pixel_y, pixel_x])
            
            bbox_pos = [
                pixel_y * env.pixel_size + env.bounds[0][0],
                pixel_x * env.pixel_size + env.bounds[1][0],
                depth_image[pixel_y][pixel_x] + env.bounds[2][0],
            ]
            bbox_positions.append(bbox_pos)
            
    return bbox_ids, bbox_images, bbox_sizes, bbox_centers, bbox_positions 

def gaussian_2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return 1 / (2*math.pi*sx*sy) * \
            np.exp(-((x - mx)**2 / (2*sx**2) + (y - my)**2 / (2*sy**2)))
             
def generate_mask_map(bbox_centers, bbox_sizes):
    # 0: being occupied -> 1: available 
    mask_map =  np.ones((IMAGE_SIZE, IMAGE_SIZE))
    for i in range(len(bbox_centers)):
        cx, cy = bbox_centers[i]
        w, h = bbox_sizes[i]
        mask_map[max(0, cx - h//2) : min(IMAGE_SIZE-1, cx + h//2), max(0, cy - w//2) : min(IMAGE_SIZE-1, cy + w//2)] = 0 
    # plt.imshow(mask_map, cmap='gray', vmin=0, vmax = 1)
    plt.show()
    return mask_map

# adaptive sampling
def sample_from_mask(all_mask, mx, my, mx2=None, my2=None) :
    x = np.linspace(0, IMAGE_SIZE, num=IMAGE_SIZE, endpoint=False)
    y = np.linspace(0, IMAGE_SIZE, num=IMAGE_SIZE, endpoint=False)
    x, y = np.meshgrid(x, y)

    sample_distribution = gaussian_2d(y, x, mx=mx, my=my, sx=IMAGE_SIZE/7, sy=IMAGE_SIZE/7)
    if mx2 and my2:
        sample_distribution += gaussian_2d(y, x, mx=mx2, my=my2, sx=IMAGE_SIZE/7, sy=IMAGE_SIZE/7)
    # ic(sample_distribution.shape)
    sample_distribution = sample_distribution * all_mask
    sample_distribution = sample_distribution / np.sum(sample_distribution)
    sample_idx = np.random.choice(a=range(IMAGE_SIZE * IMAGE_SIZE), size=1, p=sample_distribution.flatten())[0]
    sample_x = sample_idx // IMAGE_SIZE
    sample_y = sample_idx % IMAGE_SIZE
    sample_pt = (sample_x, sample_y)
    plt.imshow(sample_distribution)
    # plt.show()
    plt.savefig('sample_dist.png')
    return sample_pt, sample_distribution

def generate_sector_mask(shape, centre, angle_range, radius_min=MIN_DIS//PIXEL_SIZE, radius_max=MAX_DIS//PIXEL_SIZE):
    """
    shape image shape [h, w]
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """
    x,y = np.ogrid[:shape[0], :shape[1]] # grid_index 
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2 * np.pi

    # convert cartesian --> polar coordinates
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    theta = np.arctan2(x - cx, y - cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2 * np.pi)

    # circular mask (ring)
    circmask_in = r2 >= radius_min * radius_min
    circmask_out = r2 <= radius_max * radius_max 
    # angular mask
    anglemask = theta <= (tmax - tmin)

    return circmask_in * circmask_out * anglemask

def generate_unseen_place_inst(obj_labels, obj_dirs, bbox_ids, bbox_centers, bbox_sizes, pixel_size=PIXEL_SIZE):
    # add functional objects
    ori_lang_goal, ref_obj_ids, ref_obj_centers, ref_regions, all_mask = None, None, None, None, None
    mask_map = generate_mask_map(bbox_centers, bbox_sizes)

    choices = np.random.choice(a=range(len(obj_labels)), size=len(obj_labels), replace=False)
    for choice_idx in choices:  # object checking id 
        obj_id = list(obj_labels.keys())[choice_idx]
        # if have unique label, use unique label
        obj_name = obj_labels[obj_id][0]
        obj_dir = obj_dirs[obj_id]
        
        if obj_id in bbox_ids:
            choice_bbox_idx = bbox_ids.index(obj_id)
        else:
            # the reference object is completely occluded
            continue
        
        dir_choices = []
        for direction in OBJ_UNSEEN_DIRECTION_MAP.keys():          
            if obj_dir in OBJ_UNSEEN_DIRECTION_MAP[direction]:
                dir_choices.append(direction)                
        region_name = np.random.choice(a=dir_choices, size=1)[0]

        if region_name in ["on", "in", "to", "upside", "on top of"]:
            sector_mask = generate_sector_mask((IMAGE_SIZE, IMAGE_SIZE), bbox_centers[choice_bbox_idx], angle_range=[0, 359.99], \
                                                radius_min=0, radius_max=min(bbox_sizes[choice_bbox_idx])//2)
            all_mask = sector_mask
        elif region_name in ["near", "around", "next to", "beside", "close to", "surrounding to"]:
            sector_mask = generate_sector_mask((IMAGE_SIZE, IMAGE_SIZE), bbox_centers[choice_bbox_idx], angle_range=[0, 359.99], \
                                                radius_min=max(MIN_DIS//pixel_size, max(bbox_sizes[choice_bbox_idx])//2), radius_max=MAX_DIS//pixel_size) 
            all_mask = mask_map * sector_mask
        
        if np.count_nonzero(all_mask) > np.count_nonzero(sector_mask) * 2 / 3 and np.count_nonzero(all_mask) > 800:
            ref_obj_ids = [obj_id]
            dir_phrase = region_name
            # diverse
            place_lang_goal = np.random.choice(a=PLACE_LANG_TEMPLATES, size=1)[0]
            # not diverse
            # place_lang_goal = PLACE_LANG_TEMPLATES[0]
            ori_lang_goal = place_lang_goal.format(reference=obj_name, direction=dir_phrase)
            
            ref_obj_centers = [bbox_centers[choice_bbox_idx]]
            ref_regions = [region_name]

            if ori_lang_goal:
                break

    return ori_lang_goal, ref_obj_ids, ref_obj_centers, ref_regions, all_mask

def generate_place_inst(obj_labels, obj_dirs, bbox_ids, bbox_centers, bbox_sizes, pixel_size=PIXEL_SIZE):
    # add functional objects
    ori_lang_goal, ref_obj_ids, ref_obj_centers, ref_regions, all_mask = None, None, None, None, None
    mask_map = generate_mask_map(bbox_centers, bbox_sizes)

    choices = np.random.choice(a=range(len(obj_labels)), size=len(obj_labels), replace=False)
    for choice_idx in choices:  # object checking id 
        obj_id = list(obj_labels.keys())[choice_idx]
        # if have unique label, use unique label
        obj_name = obj_labels[obj_id][0]
        obj_dir = obj_dirs[obj_id]
        
        if obj_id in bbox_ids:
            choice_bbox_idx = bbox_ids.index(obj_id)
        else:
            # the reference object is completely occluded
            continue
        
        dir_choices = []
        for direction in OBJ_DIRECTION_MAP.keys():          
            if obj_dir in OBJ_DIRECTION_MAP[direction]:
                dir_choices.append(direction)
        
        region_name = np.random.choice(a=dir_choices, size=1)[0]

        if region_name in ["on", "in", "to", "upside", "on top of"]:
            sector_mask = generate_sector_mask((IMAGE_SIZE, IMAGE_SIZE), bbox_centers[choice_bbox_idx], angle_range=[0, 359.99], \
                                                radius_min=0, radius_max=min(bbox_sizes[choice_bbox_idx])//2)
            all_mask = sector_mask
        elif region_name in ["near", "around", "next to", "beside", "close to", "surrounding to"]:
            sector_mask = generate_sector_mask((IMAGE_SIZE, IMAGE_SIZE), bbox_centers[choice_bbox_idx], angle_range=[0, 359.99], \
                                                radius_min=max(MIN_DIS//pixel_size, max(bbox_sizes[choice_bbox_idx])//2), radius_max=MAX_DIS//pixel_size) 
            all_mask = mask_map * sector_mask
        
        if np.count_nonzero(all_mask) > np.count_nonzero(sector_mask) * 2 / 3 and np.count_nonzero(all_mask) > 800:
            ref_obj_ids = [obj_id]
            dir_phrase = region_name
            place_lang_goal = np.random.choice(a=PLACE_LANG_TEMPLATES, size=1)[0]
            ori_lang_goal = place_lang_goal.format(reference=obj_name, direction=dir_phrase)
            
            ref_obj_centers = [bbox_centers[choice_bbox_idx]]
            ref_regions = [region_name]

            if ori_lang_goal:
                break

    return ori_lang_goal, ref_obj_ids, ref_obj_centers, ref_regions, all_mask

def generate_all_place_dist(ref_obj_id, ref_obj_name, ref_dir_phase, place_valid_mask, obj_labels, bbox_ids, bbox_centers, bbox_sizes, grasped_obj_size=None, pixel_size=PIXEL_SIZE, visualize=False):    
    mask_map = generate_mask_map(bbox_centers, bbox_sizes)
    # check if there are other target objects (general label)
    candidate_ref_obj_ids = []
    if place_valid_mask is None:
        candidate_ref_obj_ids.append(ref_obj_id)
    
    # !!! add object with the same object label to the refernece objects !!!
    for id in obj_labels.keys():
        if id != ref_obj_id:
            for obj_name in obj_labels[id]:
                if obj_name == ref_obj_name:
                    candidate_ref_obj_ids.append(id)
        
    all_place_valid_mask = place_valid_mask if place_valid_mask is not None else np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    for candidate_ref_obj_id in candidate_ref_obj_ids:
        
        if candidate_ref_obj_id in bbox_ids:
            choice_bbox_idx = bbox_ids.index(candidate_ref_obj_id)
        else:
            # the reference object is completely occluded
            continue
    
        if ref_dir_phase in ["on", "in", "to", "into", "upside", "on top of"]:
            sector_mask = generate_sector_mask((IMAGE_SIZE, IMAGE_SIZE), bbox_centers[choice_bbox_idx], angle_range=[0, 359.99], \
                                                radius_min=0, radius_max=min(bbox_sizes[choice_bbox_idx])//2)
            all_mask = sector_mask
        elif ref_dir_phase in ["near", "around", "next to", "beside", "close to", "surrounding to"]:
            # sector_mask = generate_sector_mask((IMAGE_SIZE, IMAGE_SIZE), bbox_centers[choice_bbox_idx], angle_range=[0, 359.99], \
            #                                     radius_min=max(MIN_DIS//pixel_size, max(bbox_sizes[choice_bbox_idx])//2), radius_max=MAX_DIS//pixel_size) #max(MAX_DIS//pixel_size, max(bbox_sizes[choice_bbox_idx])//2 + 0.1)
            if grasped_obj_size is None:    
                sector_mask = generate_sector_mask((IMAGE_SIZE, IMAGE_SIZE), bbox_centers[choice_bbox_idx], angle_range=[0, 359.99], \
                                                radius_min=max(MIN_DIS//pixel_size, max(bbox_sizes[choice_bbox_idx])//2), radius_max=max(MAX_DIS//pixel_size, max(bbox_sizes[choice_bbox_idx])//2 + 0.1))
            else:
                sector_mask = generate_sector_mask((IMAGE_SIZE, IMAGE_SIZE), bbox_centers[choice_bbox_idx], angle_range=[0, 359.99], \
                                                radius_min=max(MIN_DIS//pixel_size, max(bbox_sizes[choice_bbox_idx])//2), radius_max=max(MAX_DIS//pixel_size, max(bbox_sizes[choice_bbox_idx])//2 + max(grasped_obj_size)//2 + 0.1)) 


            all_mask = mask_map * sector_mask
        
        if np.count_nonzero(all_mask) > np.count_nonzero(sector_mask) * 2 / 3 and np.count_nonzero(all_mask) > 800:
            all_place_valid_mask = np.logical_or(all_place_valid_mask, all_mask)
             
    # if visualize:
    cv2.imwrite('place_mask_map.png', mask_map * 255)
    cv2.imwrite('place_valid_mask.png', all_place_valid_mask * 255)
    
    return all_place_valid_mask

def relabel_mask(env, mask_image):
    assert env.target_obj_id != -1
    num_obj = 50
    for i in np.unique(mask_image):
        if i == env.target_obj_id:
            mask_image[mask_image == i] = 255
        elif i in env.obj_ids["rigid"]:
            mask_image[mask_image == i] = num_obj
            num_obj += 10
        else:
            mask_image[mask_image == i] = 0
    mask_image = mask_image.astype(np.uint8)
    return mask_image

def rotate(image, angle, is_mask=False):
    """Rotate an image using cv2, counterclockwise in degrees"""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    if is_mask:
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST)
    else:
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

    return rotated

def preprocess_pp(pts, feat_dict, grasp_pose_set, place_pose_set, sample_num, sample_grasp=False, sample_place=False, downsample_interval=10, visualize=False):    
    pts = torch.from_numpy(pts)
    clip_feats = feat_dict['clip_feats']
    clip_sims = feat_dict['clip_sims'][..., 0]

    # !!! sample top 50%(8 objs)/25%(15 objs) points !!!
    sample_indices = torch.argsort(clip_sims, descending=True)[:sample_num]
    sampled_pts = pts[sample_indices]
    sampled_clip_feats = clip_feats[sample_indices]
    sampled_clip_sims = clip_sims[sample_indices][..., None]
    sampled_pts = sampled_pts.unsqueeze(0).to(dtype=torch.float32) # shape = [1, sample_num, 3]
    sampled_clip_feats = sampled_clip_feats.unsqueeze(0).to(dtype=torch.float32) # shape = [1, sample_num, feat_dim]
    sampled_clip_sims = sampled_clip_sims.unsqueeze(0).to(dtype=torch.float32) # shape = [1, sample_num, 1]

    if visualize:
        # visualization
        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_pts[0].detach().cpu().numpy()) # [N, 3]
        cmap = plt.get_cmap("turbo")
        sampled_clip_sims_color = cmap(sampled_clip_sims[0, ..., 0].detach().cpu().numpy())[..., :3]
        sampled_pcd.colors = o3d.utility.Vector3dVector(sampled_clip_sims_color) # [N, 3]
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        o3d.visualization.draw_geometries([sampled_pcd, frame])   

    # normalized sampled clip sims
    z = 1e-4
    sampled_clip_sims = (sampled_clip_sims - sampled_clip_sims.min() + z) / (sampled_clip_sims.max() - sampled_clip_sims.min() + z) 

    if visualize:
        cmap = plt.get_cmap("turbo")
        sampled_clip_sims_color = cmap(sampled_clip_sims[0, ..., 0].detach().cpu().numpy())[..., :3]
        sampled_pcd.colors = o3d.utility.Vector3dVector(sampled_clip_sims_color) # [N, 3]
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        o3d.visualization.draw_geometries([sampled_pcd, frame])    

    grasps = None
    for grasp in grasp_pose_set:  
        grasp = torch.from_numpy(grasp)
        grasp = grasp.unsqueeze(0)
        if grasps == None:
            grasps = grasp
        else:
            grasps = torch.cat((grasps, grasp), dim=0) # shape = [n_grasp, grasp_dim]
    grasps = grasps.unsqueeze(0).to(dtype=torch.float32) # shape = [1, n_grasp, grasp_dim]

    if sample_grasp:
        pos_grasps = grasps[..., :3][0].unsqueeze(1)
        dist_map = torch.norm(pos_grasps - sampled_pts, dim=-1)
        min_dist = torch.min(dist_map, dim=1)[0]
        sampled_indices = torch.where(min_dist < 0.05)[0]
        grasps = grasps[:, sampled_indices, :]
        grasp_pose_set = list(np.array(grasp_pose_set)[sampled_indices.cpu().numpy()])
        
    places = None
    for place in place_pose_set:
        place = torch.from_numpy(place)
        place = place.unsqueeze(0)
        if places == None:
            places = place
        else:
            places = torch.cat((places, place), dim=0) # shape = [n_place, place_dim]
    places = places.unsqueeze(0).to(dtype=torch.float32) # shape = [1, n_place, place_dim]
    
    if sample_place:
        pos_places = places[..., :3][0].unsqueeze(1)
        dist_map = torch.norm(pos_places - sampled_pts, dim=-1)
        min_dist = torch.min(dist_map, dim=1)[0]
        sampled_indices = torch.where(min_dist < 0.05)[0]
        places = places[:, sampled_indices, :]
        place_pose_set = list(np.array(place_pose_set)[sampled_indices.cpu().numpy()])

    return sampled_pts, sampled_clip_feats, sampled_clip_sims, grasps, grasp_pose_set, places, place_pose_set

def preprocess_pp_unified(pts, feat_dict, action_set, sample_num, sample_action=False, downsample_interval=10, visualize=False):    
    pts = torch.from_numpy(pts)
    clip_feats = feat_dict['clip_feats']
    clip_sims = feat_dict['clip_sims'][..., 0]

    # !!! sample top 50%(8 objs)/25%(15 objs) points !!!
    sample_indices = torch.argsort(clip_sims, descending=True)[:sample_num]
    sampled_pts = pts[sample_indices]
    sampled_clip_feats = clip_feats[sample_indices]
    sampled_clip_sims = clip_sims[sample_indices][..., None]
    sampled_pts = sampled_pts.unsqueeze(0).to(dtype=torch.float32) # shape = [1, sample_num, 3]
    sampled_clip_feats = sampled_clip_feats.unsqueeze(0).to(dtype=torch.float32) # shape = [1, sample_num, feat_dim]
    sampled_clip_sims = sampled_clip_sims.unsqueeze(0).to(dtype=torch.float32) # shape = [1, sample_num, 1]

    if visualize:
        # visualization
        sampled_pcd = o3d.geometry.PointCloud()
        sampled_pcd.points = o3d.utility.Vector3dVector(sampled_pts[0].detach().cpu().numpy()) # [N, 3]
        cmap = plt.get_cmap("turbo")
        sampled_clip_sims_color = cmap(sampled_clip_sims[0, ..., 0].detach().cpu().numpy())[..., :3]
        sampled_pcd.colors = o3d.utility.Vector3dVector(sampled_clip_sims_color) # [N, 3]
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        o3d.visualization.draw_geometries([sampled_pcd, frame])   

    # normalized sampled clip sims
    z = 1e-4
    sampled_clip_sims = (sampled_clip_sims - sampled_clip_sims.min() + z) / (sampled_clip_sims.max() - sampled_clip_sims.min() + z) 

    if visualize:
        cmap = plt.get_cmap("turbo")
        sampled_clip_sims_color = cmap(sampled_clip_sims[0, ..., 0].detach().cpu().numpy())[..., :3]
        sampled_pcd.colors = o3d.utility.Vector3dVector(sampled_clip_sims_color) # [N, 3]
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        o3d.visualization.draw_geometries([sampled_pcd, frame])    

    actions = None
    for action in action_set:
        # transfer to rotation vector
        # rot_vec = R.from_quat(grasp[-4:]).as_rotvec()
        # grasp[-3:] = rot_vec
        # grasp = grasp[:6]   
        action = torch.from_numpy(action)
        action = action.unsqueeze(0)
        if actions == None:
            actions = action
        else:
            actions = torch.cat((actions, action), dim=0) # shape = [n_grasp, grasp_dim]
    actions = actions.unsqueeze(0).to(dtype=torch.float32) # shape = [1, n_grasp, grasp_dim]

    if sample_action:
        pos_actions = actions[..., :3][0].unsqueeze(1)
        dist_map = torch.norm(pos_actions - sampled_pts, dim=-1)
        min_dist = torch.min(dist_map, dim=1)[0]
        sampled_indices = torch.where(min_dist < 0.05)[0]
        action = actions[:, sampled_indices, :]
        action_set = list(np.array(action_set)[sampled_indices.cpu().numpy()])
        
    return sampled_pts, sampled_clip_feats, sampled_clip_sims, actions, action_set

# preprocess of object bboxes
def preprocess_bboxes(bbox_ids, bbox_images, bbox_sizes, bbox_centers, bbox_positions):
    remain_bbox_ids = []
    remain_bbox_images = []
    remain_bbox_sizes = []
    remain_bbox_centers = []
    remain_bbox_positions = []

    for i in range(len(bbox_images)):
        if bbox_images[i].shape[0] >= 5 and bbox_images[i].shape[1] >= 5:
            remain_bbox_ids.append(bbox_ids[i])
            remain_bbox_images.append(bbox_images[i])  # shape = [n_obj, H, W, C]
            remain_bbox_sizes.append(bbox_sizes[i])
            remain_bbox_centers.append(bbox_centers[i])
            remain_bbox_positions.append(bbox_positions[i])
    print('Remaining bbox number', len(remain_bbox_images))

    return remain_bbox_ids, remain_bbox_images, remain_bbox_sizes, remain_bbox_centers, remain_bbox_positions

# Preprocess of model input
def preprocess_2d(bbox_images, bbox_positions, grasp_pose_set, n_px):
    from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
    transform = Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    
    remain_bboxes = []
    remain_bbox_positions = []
    for i in range(len(bbox_images)):
        if bbox_images[i].shape[0] >= 15 and bbox_images[i].shape[1] >= 15:
            remain_bboxes.append(bbox_images[i])  # shape = [n_obj, H, W, C]
            remain_bbox_positions.append(bbox_positions[i])
    print('Remaining bbox number', len(remain_bboxes))
    bboxes = None
    for remain_bbox in remain_bboxes:
        remain_bbox = Image.fromarray(remain_bbox)
        # padding
        w,h = remain_bbox.size
        if w >= h:
            remain_bbox_ = Image.new(mode='RGB', size=(w,w))
            remain_bbox_.paste(remain_bbox, box=(0, (w-h)//2))
        else:
            remain_bbox_ = Image.new(mode='RGB', size=(h,h))
            remain_bbox_.paste(remain_bbox, box=((h-w)//2, 0))
        remain_bbox_ = transform(remain_bbox_)

        remain_bbox_ = remain_bbox_.unsqueeze(0)
        if bboxes == None:
            bboxes = remain_bbox_
        else:
            bboxes = torch.cat((bboxes, remain_bbox_), dim=0) # shape = [n_obj, C, patch_size, patch_size]
    if bboxes != None:
        bboxes = bboxes.unsqueeze(0) # shape = [1, n_obj, C, patch_size, patch_size]
    
    pos_bboxes = None
    for bbox_pos in remain_bbox_positions:
        bbox_pos = torch.from_numpy(np.array(bbox_pos))
        bbox_pos = bbox_pos.unsqueeze(0)
        if pos_bboxes == None:
            pos_bboxes = bbox_pos
        else:
            pos_bboxes = torch.cat((pos_bboxes, bbox_pos), dim=0) # shape = [n_obj, pos_dim]
    if pos_bboxes != None:
        pos_bboxes = pos_bboxes.unsqueeze(0).to(dtype=torch.float32) # shape = [1, n_obj, pos_dim]
    

    grasps = None
    for grasp in grasp_pose_set:
        # transfer to rotation vector
        # rot_vec = R.from_quat(grasp[-4:]).as_rotvec()
        # grasp[-3:] = rot_vec
        # grasp = grasp[:6]
        grasp = torch.from_numpy(grasp)
        grasp = grasp.unsqueeze(0)
        if grasps == None:
            grasps = grasp
        else:
            grasps = torch.cat((grasps, grasp), dim=0) # shape = [n_grasp, grasp_dim]
    grasps = grasps.unsqueeze(0).to(dtype=torch.float32) # shape = [1, n_grasp, grasp_dim]

    return remain_bboxes, bboxes, pos_bboxes, grasps

def resize(img, target_res, resize=True, to_pil=True, edge=False):
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2: (width + height) // 2] = img
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2: (height + width) // 2] = img
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(img, pad_width=[(top_pad, bottom_pad), (0, 0), (0, 0)], mode='edge')
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(img, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode='edge')
        canvas = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas

def plot_attnmap(attn_map):
    fig, ax = plt.subplots()
    ax.set_yticks([])
    # ax.set_yticks(range(attn_map.shape[0]))
    ax.set_xticks([])
    h, w = attn_map.shape
    # darker, larger
    im = ax.imshow(attn_map, cmap="YlGnBu", interpolation='nearest', extent=[0, w, 0, h], aspect='auto')
    # brigher, larger
    # im = ax.imshow(attn_map, cmap="viridis", interpolation='nearest', extent=[0, w, 0, h], aspect='auto')

    if h > w:
        ort = 'vertical'
    else:
        ort = 'horizontal'

    plt.colorbar(im, orientation=ort)

    # for i in range(attn_map.shape[0]):
    #     for j in range(attn_map.shape[1]):
    #         # print('data[{},{}]:{}'.format(i, j, attn_map[i, j]))
    #         ax.text(j, i, round(attn_map[i, j]*100, 2),
    #                 ha="center", va="center", color="black")

    plt.xlabel('point')
    plt.ylabel('grasp')

    # show
    fig.tight_layout()
    plt.show()

def get_pca_map(feature_map, img_size, interpolation="bicubic", return_pca_stats=False, pca_stats=None):
    """
    feature_map: (1, h, w, C) is the feature map of a single image.
    """
    if feature_map.shape[0] != 1:
        # make it (1, h, w, C)
        feature_map = feature_map[None]
    if pca_stats is None:
        reduct_mat, color_min, color_max = get_robust_pca(
            feature_map.reshape(-1, feature_map.shape[-1])
        )
    else:
        reduct_mat, color_min, color_max = pca_stats
    pca_color = feature_map @ reduct_mat
    pca_color = (pca_color - color_min) / (color_max - color_min)
    pca_color = pca_color.clamp(0, 1)
    pca_color = F.interpolate(
        pca_color.permute(0, 3, 1, 2),
        size=img_size,
        mode=interpolation,
    ).permute(0, 2, 3, 1)
    pca_color = pca_color.cpu().numpy().squeeze(0)
    if return_pca_stats:
        return pca_color, (reduct_mat, color_min, color_max)
    return pca_color

def get_robust_pca(features, m=2, remove_first_component=False):
    # features: (N, C)
    # m: a hyperparam controlling how many std dev outside for outliers
    assert len(features.shape) == 2, "features should be (N, C)"
    reduction_mat = torch.pca_lowrank(features, q=3, niter=20)[2]
    colors = features @ reduction_mat
    if remove_first_component:
        colors_min = colors.min(dim=0).values
        colors_max = colors.max(dim=0).values
        tmp_colors = (colors - colors_min) / (colors_max - colors_min)
        fg_mask = tmp_colors[..., 0] < 0.2
        reduction_mat = torch.pca_lowrank(features[fg_mask], q=3, niter=20)[2]
        colors = features @ reduction_mat
    else:
        fg_mask = torch.ones_like(colors[:, 0]).bool()
    d = torch.abs(colors[fg_mask] - torch.median(colors[fg_mask], dim=0).values)
    mdev = torch.median(d, dim=0).values
    s = d / mdev
    try:
        rins = colors[fg_mask][s[:, 0] < m, 0]
        gins = colors[fg_mask][s[:, 1] < m, 1]
        bins = colors[fg_mask][s[:, 2] < m, 2]
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])
    except:
        rins = colors
        gins = colors
        bins = colors
        rgb_min = torch.tensor([rins.min(), gins.min(), bins.min()])
        rgb_max = torch.tensor([rins.max(), gins.max(), bins.max()])

    return reduction_mat, rgb_min.to(reduction_mat), rgb_max.to(reduction_mat)

# Get rotation matrix from euler angles
def euler2rotm(theta):
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )
    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )
    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

def update_intrinsics_on_crop(intrinsics, crop_x, crop_y):
    """Update camera intrinsics after cropping."""
    K = intrinsics.copy()
    K[0, 2] -= crop_x
    K[1, 2] -= crop_y
    return K

def update_intrinsics_on_pad(intrinsics, pad_x, pad_y):
    """Update camera intrinsics after padding."""
    K = intrinsics.copy()
    K[0, 2] += pad_x
    K[1, 2] += pad_y
    return K

def update_intrinsics_on_resize(intrinsics, orig_size, new_size):
    """Update camera intrinsics after resizing."""
    K = intrinsics.copy()
    scale_x = new_size[0] / orig_size[0]
    scale_y = new_size[1] / orig_size[1]
    K[0, 0] *= scale_x
    K[1, 1] *= scale_y
    K[0, 2] *= scale_x
    K[1, 2] *= scale_y
    return K

def normalize_pos(pos, workspace_limits=WORKSPACE_LIMITS.T, device="cuda"):
    pos_min = torch.from_numpy(workspace_limits[0]).float().to(device)
    pos_max = torch.from_numpy(workspace_limits[1]).float().to(device)
    return (pos - pos_min) / (pos_max - pos_min) * 2.0 - 1.0

def unnormalize_pos(pos, workspace_limits=WORKSPACE_LIMITS.T, device="cuda"):
    pos_min = torch.from_numpy(workspace_limits[0]).float().to(device)
    pos_max = torch.from_numpy(workspace_limits[1]).float().to(device)
    return (pos + 1.0) / 2.0 * (pos_max - pos_min) + pos_min


# Checks if a matrix is a valid rotation matrix.
def isRotm(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
def rotm2euler(R):
    assert isRotm(R)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def angle2rotm(angle, axis, point=None):
    # Copyright (c) 2006-2018, Christoph Gohlke
    sina = math.sin(angle)
    cosa = math.cos(angle)
    axis = axis / np.linalg.norm(axis)

    # Rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(axis, axis) * (1.0 - cosa)
    axis *= sina
    R += np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float32,
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:

        # Rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M


def rotm2angle(R):
    # From: euclideanspace.com
    epsilon = 0.01  # Margin to allow for rounding errors
    epsilon2 = 0.1  # Margin to distinguish between 0 and 180 degrees

    assert isRotm(R)

    if (
        (abs(R[0][1] - R[1][0]) < epsilon)
        and (abs(R[0][2] - R[2][0]) < epsilon)
        and (abs(R[1][2] - R[2][1]) < epsilon)
    ):
        # Singularity found
        # First check for identity matrix which must have +1 for all terms in leading diagonaland zero in other terms
        if (
            (abs(R[0][1] + R[1][0]) < epsilon2)
            and (abs(R[0][2] + R[2][0]) < epsilon2)
            and (abs(R[1][2] + R[2][1]) < epsilon2)
            and (abs(R[0][0] + R[1][1] + R[2][2] - 3) < epsilon2)
        ):
            # this singularity is identity matrix so angle = 0
            return [0, 1, 0, 0]  # zero angle, arbitrary axis

        # Otherwise this singularity is angle = 180
        angle = np.pi
        xx = (R[0][0] + 1) / 2
        yy = (R[1][1] + 1) / 2
        zz = (R[2][2] + 1) / 2
        xy = (R[0][1] + R[1][0]) / 4
        xz = (R[0][2] + R[2][0]) / 4
        yz = (R[1][2] + R[2][1]) / 4
        if (xx > yy) and (xx > zz):  # R[0][0] is the largest diagonal term
            if xx < epsilon:
                x = 0
                y = 0.7071
                z = 0.7071
            else:
                x = np.sqrt(xx)
                y = xy / x
                z = xz / x
        elif yy > zz:  # R[1][1] is the largest diagonal term
            if yy < epsilon:
                x = 0.7071
                y = 0
                z = 0.7071
            else:
                y = np.sqrt(yy)
                x = xy / y
                z = yz / y
        else:  # R[2][2] is the largest diagonal term so base result on this
            if zz < epsilon:
                x = 0.7071
                y = 0.7071
                z = 0
            else:
                z = np.sqrt(zz)
                x = xz / z
                y = yz / z
        return [angle, x, y, z]  # Return 180 deg rotation

    # As we have reached here there are no singularities so we can handle normally
    s = np.sqrt(
        (R[2][1] - R[1][2]) * (R[2][1] - R[1][2])
        + (R[0][2] - R[2][0]) * (R[0][2] - R[2][0])
        + (R[1][0] - R[0][1]) * (R[1][0] - R[0][1])
    )  # used to normalise
    if abs(s) < 0.001:
        s = 1

    # Prevent divide by zero, should not happen if matrix is orthogonal and should be
    # Caught by singularity test above, but I've left it in just in case
    angle = np.arccos((R[0][0] + R[1][1] + R[2][2] - 1) / 2)
    x = (R[2][1] - R[1][2]) / s
    y = (R[0][2] - R[2][0]) / s
    z = (R[1][0] - R[0][1]) / s
    return [angle, x, y, z]
