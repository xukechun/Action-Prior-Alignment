import os
import open3d as o3d
import numpy as np
import torch
import trimesh
from models.feature_fusion import Fusion, create_init_grid
from utils.draw_utils import aggr_point_cloud_from_data


class FeatureField:
    def __init__(self, num_cam, 
                    query_threshold, 
                    grid_size, boundaries,
                    feat_backbone='clip', 
                    device='cuda:0'):
        self.grid_size = grid_size
        self.query_threshold = query_threshold
        self.device = device
        self.boundaries = {'x_lower': boundaries[0, 0],
                        'x_upper': boundaries[0, 1],
                        'y_lower': boundaries[1, 0],
                        'y_upper': boundaries[1, 1],
                        'z_lower': boundaries[2, 0],
                        'z_upper': boundaries[2, 1],}
        
        self.fusion = Fusion(num_cam=num_cam, feat_backbone=feat_backbone, device=device)        
        
    def generate_feature_field(self, colors, depths, extrinsics, intrinsics, query_text, feature_types, pts=None, last_text_feature=False, visualize=False):
        # camera view number: N, colors: [N, H, W, 3], depths: [N, H, W], extrinsics: [N, 4, 4], intrinsics: [N, 3, 3]
        # return name: the list of feature type 
        obs = {
                'color': colors,
                'depth': depths,
                'pose': extrinsics[:, :3], # (N, 3, 4)
                'K': intrinsics,
            }
        
        self.fusion.update(obs, query_text, last_text_feature, visualize) # 2.88s
        # self.extract_mask([query_text])
        # print('t1: ', time.time() - t0)

        # extract feature
        # vertices, triangles, feat_dict = self.extract_feature(feature_types) # 3.77s
        vertices, feat_dict = self.extract_pts_feature(pts, feature_types)

        # add text feature
        feat_dict['text_feat'] = self.fusion.curr_obs_torch['text_feat']
        
        if visualize:
            # extract color mesh
            # print('create color mesh') 
            # color_mesh = self.extract_color_mesh(vertices, triangles, feat_dict, visualize=visualize)
            # extract mask mesh
            # print('create mask mesh')
            # mask_mesh = self.extract_mask_mesh(vertices, triangles, out, visualize=visualize)
            # extract feature mesh
            for feat in feature_types:
                print('create %s mesh' % feat)
                # mesh = self.extract_feature_mesh(vertices, triangles, feat_dict, mask_out_bg=False, decriptor_type=feat, visualize=visualize)
                pcd = self.extract_feature_pcd(vertices, feat_dict, mask_out_bg=False,  decriptor_type=feat, visualize=visualize)                
                
        return vertices, feat_dict

    def generate_object_feature_field(self, colors, depths, masks, extrinsics, intrinsics, query_text, feature_types, pts=None, last_text_feature=False, visualize=False):
        # camera view number: N, colors: [N, H, W, 3], depths: [N, H, W], extrinsics: [N, 4, 4], intrinsics: [N, 3, 3]
        # return name: the list of feature type         
        
        obs = {
                'color': colors,
                'depth': depths,
                'mask': masks,
                'pose': extrinsics[:, :3], # (N, 3, 4)
                'K': intrinsics,
            }
        
        # import time
        # t0 = time.time()
        self.fusion.update(obs, query_text, last_text_feature, visualize) # 2.88s
        # self.extract_mask([query_text])
        # print('t1: ', time.time() - t0)

        # extract feature
        # vertices, triangles, feat_dict = self.extract_feature(feature_types) # 3.77s
        vertices, feat_dict = self.extract_pts_feature(pts, feature_types)

        # add text feature
        feat_dict['text_feat'] = self.fusion.curr_obs_torch['text_feat']
        
        if visualize:
            # extract color mesh
            # print('create color mesh') 
            # color_mesh = self.extract_color_mesh(vertices, triangles, feat_dict, visualize=visualize)
            # extract mask mesh
            # print('create mask mesh')
            # mask_mesh = self.extract_mask_mesh(vertices, triangles, out, visualize=visualize)
            # extract feature mesh
            for feat in feature_types:
                print('create %s mesh' % feat)
                # mesh = self.extract_feature_mesh(vertices, triangles, feat_dict, mask_out_bg=False, decriptor_type=feat, visualize=visualize)
                pcd = self.extract_feature_pcd(vertices, feat_dict, mask_out_bg=False,  decriptor_type=feat, visualize=visualize)                
                
        return vertices, feat_dict
    
    def extract_mask(self, query_text):
        # extract multi-view masks with grounding SAM
        self.fusion.text_queries_for_inst_mask_no_track(query_text, self.query_threshold)        

    def get_pcd(self, colors, depths, intrinsics, extrinsics):
        pcd = aggr_point_cloud_from_data(colors, depths, intrinsics, extrinsics, downsample=True, boundaries=self.boundaries)
        pcd.remove_statistical_outlier(nb_neighbors=5, std_ratio=0.2)
        return pcd

    def extract_feature(self, feature_types):
        # build init grid
        init_grid, grid_shape = create_init_grid(self.boundaries, self.grid_size)
        init_grid = init_grid.to(device=self.device, dtype=torch.float32)
        print('eval init grid')
        with torch.no_grad():
            out = self.fusion.batch_eval(init_grid, return_names=[])

        # extract mesh
        print('extract mesh')
        # Note that here vertices are exactly point cloud
        vertices, triangles = self.fusion.extract_mesh(init_grid, out, grid_shape) # an example shape = (16105, 3)
        vertices_tensor = torch.from_numpy(vertices).to(self.device, dtype=torch.float32)

        print('eval mesh vertices')
        with torch.no_grad():
            feat_dict = self.fusion.batch_eval(vertices_tensor, return_names=feature_types)

        return vertices, triangles, feat_dict

    def extract_pts_feature(self, pts, feature_types):
        print('in extract feature')
        # use aggregation point cloud
        pts_tensor = torch.from_numpy(pts).to(self.device, dtype=torch.float32)

        print('eval mesh vertices')
        # torch.cuda.empty_cache()
        with torch.no_grad():
            feat_dict = self.fusion.batch_eval(pts_tensor, return_names=feature_types)

        return pts, feat_dict

    def extract_mask_mesh(self, vertices, triangles, out, visualize=False):
        mask_meshes, colors = self.fusion.create_instance_mask_mesh(vertices, triangles, out)
        for mask_mesh, color in zip(mask_meshes, colors):
            if visualize:
                # self.visualize_mesh(mask_mesh)
                self.visualize_mesh_as_pcd(vertices, color[..., :3]/255)
        
        return mask_meshes

    def extract_color_mesh(self, vertices, triangles, out, visualize=False):
        color_mesh, colors = self.fusion.create_color_mesh(vertices, triangles, out)
        if visualize:
            # self.visualize_mesh(color_mesh)
            self.visualize_mesh_as_pcd(vertices, colors[..., :3]/255)
        
        return color_mesh

    def extract_feature_mesh(self, vertices, triangles, out, mask_out_bg, decriptor_type, visualize=False):
        feature_mesh, colors = self.fusion.create_descriptor_mesh(vertices, triangles, out, mask_out_bg, decriptor_type)
        if visualize:
            # self.visualize_mesh(feature_mesh)
            self.visualize_mesh_as_pcd(vertices, colors[..., :3]/255)
        
        return feature_mesh

    def extract_feature_pcd(self, pts, out, mask_out_bg, decriptor_type, visualize=False):
        feature_pcd = self.fusion.create_descriptor_pcd(pts, out, mask_out_bg, decriptor_type, visualize)
        return feature_pcd


    def visualize_mesh(self, mesh):
        cam = trimesh.scene.Camera(resolution=(1920, 1043), fov=(60, 60))

        cam_matrix = np.array([[ 0.87490918, -0.24637599,  0.41693261,  0.63666708],
                            [-0.44229374, -0.75717002,  0.4806972,   0.66457463],
                            [ 0.19725663, -0.60497308, -0.77142556, -1.16125645],
                            [ 0.        , -0.        , -0.        ,  1.        ]])
        mesh_scene = trimesh.Scene(mesh, camera=cam, camera_transform=cam_matrix)
        mesh_scene.show()

    def visualize_mesh_as_pcd(self, vertices, colors):
        print('visulize mesh as pcd')
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(vertices)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
        o3d.visualization.draw_geometries([point_cloud])