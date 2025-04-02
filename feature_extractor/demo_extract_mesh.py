import os
import sys
sys.path.append(os.getcwd())
import argparse

import cv2
import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA
import torch
import trimesh

from models.feature_fusion import Fusion, create_init_grid
from utils.draw_utils import aggr_point_cloud_from_data


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', action='store', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed (default: 1234)')
    parser.add_argument('--scene', action='store', type=str, default='banana')
    parser.add_argument('--data_path', action='store', type=str, default='data/2023-12-11')
    parser.add_argument('--query_text', action='store', type=str, default='banana')
    parser.add_argument('--query_threshold', type=float, default=0.35, metavar='N',
                        help='query threshold of text, argument for Grounding SAM')

    args = parser.parse_args()
    return args


args = parse_args()
scene = args.scene
data_path = args.data_path
query_texts = [args.query_text]
query_thresholds = [args.query_threshold]


# hyper-parameter
t = 0 #50
num_cam = 3

step = 0.004
# workspace of the simulation
x_upper = 0.724
x_lower = 0.276
y_upper = 0.224
y_lower = -0.224
z_upper = 0.4
z_lower = -0.0001


boundaries = {'x_lower': x_lower,
              'x_upper': x_upper,
              'y_lower': y_lower,
              'y_upper': y_upper,
              'z_lower': z_lower,
              'z_upper': z_upper,}

# pca = pickle.load(open(pca_path, 'rb'))

fusion = Fusion(num_cam=num_cam, feat_backbone='clip')

# !!! bgr!!!
colors = np.stack([cv2.imread(os.path.join(data_path, f'camera_{i}', 'color', f'{t}.png')) for i in range(num_cam)], axis=0) # [N, H, W, C]
depths = np.stack([cv2.imread(os.path.join(data_path, f'camera_{i}', 'depth', f'{t}.png'), cv2.IMREAD_ANYDEPTH) for i in range(num_cam)], axis=0) / 1000. # [N, H, W]

H, W = colors.shape[1:3]

extrinsics = np.stack([np.load(os.path.join(data_path, f'camera_{i}', 'camera_extrinsics.npy')) for i in range(num_cam)])
cam_param = np.stack([np.load(os.path.join(data_path, f'camera_{i}', 'camera_params.npy')) for i in range(num_cam)])
intrinsics = np.zeros((num_cam, 3, 3))
intrinsics[:, 0, 0] = cam_param[:, 0]
intrinsics[:, 1, 1] = cam_param[:, 1]
intrinsics[:, 0, 2] = cam_param[:, 2]
intrinsics[:, 1, 2] = cam_param[:, 3]
intrinsics[:, 2, 2] = 1

obs = {
    'color': colors[..., ::-1], # !!! Note that here colros[..., ::-1] change the images from bgr to rgb
    'depth': depths,
    'pose': extrinsics[:, :3], # (N, 3, 4)
    'K': intrinsics,
}

import time
a = time.time()
# !!! Note that here colros[..., ::-1] change the images from bgr to rgb
pcd = aggr_point_cloud_from_data(colors[..., ::-1], depths, intrinsics, extrinsics, downsample=True, boundaries=boundaries)
pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=0.2)

fusion.update(obs, query_texts[0], visualize=True)
# fusion.text_queries_for_inst_mask_no_track(query_texts, query_thresholds)

### 3D vis
device = 'cuda'

# visualize mesh
init_grid, grid_shape = create_init_grid(boundaries, step)
init_grid = init_grid.to(device=device, dtype=torch.float32)


print('eval init grid')
with torch.no_grad():
    out = fusion.batch_eval(init_grid, return_names=[])

# extract mesh
print('extract mesh')
vertices, triangles = fusion.extract_mesh(init_grid, out, grid_shape)

# eval mask and feature of vertices
vertices_tensor = torch.from_numpy(vertices).to(device, dtype=torch.float32)
print('eval mesh vertices')
with torch.no_grad():
    out = fusion.batch_eval(vertices_tensor, return_names=['clip_feats', 'clip_sims', 'color_tensor'])

cam = trimesh.scene.Camera(resolution=(1920, 1043), fov=(60, 60))

cam_matrix = np.array([[ 0.87490918, -0.24637599,  0.41693261,  0.63666708],
                       [-0.44229374, -0.75717002,  0.4806972,   0.66457463],
                       [ 0.19725663, -0.60497308, -0.77142556, -1.16125645],
                       [ 0.        , -0.        , -0.        ,  1.        ]])

# create mask mesh
# mask_meshes = fusion.create_instance_mask_mesh(vertices, triangles, out)

# for mask_mesh in mask_meshes:
#     mask_mesh_name = scene + '_mask.obj'
#     mask_mesh.export(mask_mesh_name)
#     mask_scene = trimesh.Scene(mask_mesh, camera=cam, camera_transform=cam_matrix)
#     mask_scene.show(viewer='gl')

# create feature mesh
feature_mesh = fusion.create_descriptor_mesh(vertices, triangles, out, {'pca': None}, mask_out_bg=True, decriptor_type='clip_feats')
feat_mesh_name = scene + '_feature_clip.obj'
# feature_mesh = fusion.create_descriptor_mesh(vertices, triangles, out, {'pca': None}, mask_out_bg=True, decriptor_type='dinov2_feats')
# feat_mesh_name = scene + '_feature_dino.obj'
feature_mesh.export(feat_mesh_name)
# feature_scene = trimesh.Scene(feature_mesh, camera=cam, camera_transform=cam_matrix)
# feature_scene.show()
sim_mesh = fusion.create_descriptor_mesh(vertices, triangles, out, {'pca': None}, mask_out_bg=False, decriptor_type='clip_sims')
sim_mesh_name = scene + '_clip_sim.obj'
sim_mesh.export(sim_mesh_name)
# sim_scene = trimesh.Scene(sim_mesh, camera=cam, camera_transform=cam_matrix)
# sim_scene.show()


# create color mesh
color_mesh = fusion.create_color_mesh(vertices, triangles, out)
color_mesh_name = scene + '_color.obj'
color_mesh.export(color_mesh_name)
# color_scene = trimesh.Scene(color_mesh, camera=cam, camera_transform=cam_matrix)
# color_scene.show()
