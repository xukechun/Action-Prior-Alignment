import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

class CLIPGrasp(object):
    def __init__(self, args):

        self.device = args.device

    def select_action_random(self, pts, clip_sims, actions):
        if actions.shape[0] == 1:
            pos_pts = pts[0].unsqueeze(1)
            pose_grasps = actions[0].unsqueeze(0)
            pos_grasps = pose_grasps[:, :, :3]
            dist_map = torch.norm((pos_pts - pos_grasps), dim=2)
            pts_grasp_map = (dist_map<0.05).float()
        
        clip_pred_pts = torch.argmax(clip_sims)
        grasps = torch.where(pts_grasp_map[clip_pred_pts]==1)[0]
        if len(grasps) >= 1:
            grasps = grasps[0]
        else:
            grasps = torch.randint(0, actions.shape[1], (1,))

        return int(grasps.detach().cpu().numpy())

    def select_action_greedy(self, pts, clip_sims, actions):
        if actions.shape[0] == 1:
            pos_pts = pts[0].unsqueeze(1)
            pose_grasps = actions[0].unsqueeze(0)
            pos_grasps = pose_grasps[:, :, :3]
            dist_map = torch.norm((pos_pts - pos_grasps), dim=2)
        
        clip_pred_pts = torch.argmax(clip_sims)
        grasps = torch.where(dist_map[clip_pred_pts]<0.05)[0]
        if len(grasps) >= 1:
            grasps = torch.argmin(dist_map[clip_pred_pts])
        else:
            grasps = torch.randint(0, actions.shape[1], (1,))

        return int(grasps.detach().cpu().numpy())

    def select_action_knn_greedy(self, pts, clip_sims, actions, M=500):
        if actions.shape[0] == 1:
            pos_pts = pts[0].unsqueeze(1)
            pose_grasps = actions[0].unsqueeze(0)
            pos_grasps = pose_grasps[:, :, :3]
            dist_map = torch.norm((pos_pts - pos_grasps), dim=2)

        k = int(0.05 * M)
        # 构建 KD-Tree 计算最近邻
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(pts[0])
        distances, indices = nbrs.kneighbors(pts[0])
        clip_sims_ = clip_sims[0].cpu().numpy()
        average_affordances = np.array([clip_sims_[idx].mean() for idx in indices])
        max_avg_affordance = average_affordances.max()
        best_point_index = average_affordances.argmax()
        # grasps = torch.argmin(dist_map[best_point_index])
        
        grasps = torch.where(dist_map[best_point_index]<0.05)[0]
        if len(grasps) >= 1:
            grasps = torch.argmin(dist_map[best_point_index])
        else:
            grasps = torch.randint(0, actions.shape[1], (1,))

        return int(grasps.detach().cpu().numpy())

class CLIPPlace(object):
    def __init__(self, args):

        self.device = args.device

    def select_action_random(self, pts, clip_sims, actions):
        if actions.shape[0] == 1:
            pos_pts = pts[0].unsqueeze(1)
            pose_places = actions[0].unsqueeze(0)
            pos_places = pose_places[:, :, :3]
            dist_map = torch.norm((pos_pts - pos_places), dim=2)
            pts_place_map = (dist_map<0.05).float()
        
        clip_pred_pts = torch.argmax(clip_sims)
        places = torch.where(pts_place_map[clip_pred_pts]==1)[0]
        if len(places) >= 1:
            places = places[0]
        else:
            places = torch.randint(0, actions.shape[1], (1,))

        return int(places.detach().cpu().numpy())

    def select_action_greedy(self, pts, clip_sims, actions):
        if actions.shape[0] == 1:
            pos_pts = pts[0].unsqueeze(1)
            pose_places = actions[0].unsqueeze(0)
            pos_places = pose_places[:, :, :3]
            dist_map = torch.norm((pos_pts - pos_places), dim=2)
        
        clip_pred_pts = torch.argmax(clip_sims)
        places = torch.where(dist_map[clip_pred_pts]<0.05)[0]
        if len(places) >= 1:
            places = torch.argmin(dist_map[clip_pred_pts])
        else:
            places = torch.randint(0, actions.shape[1], (1,))

        return int(places.detach().cpu().numpy())

    def select_action_knn_greedy(self, pts, clip_sims, actions, M=500):
        if actions.shape[0] == 1:
            pos_pts = pts[0].unsqueeze(1)
            pose_grasps = actions[0].unsqueeze(0)
            pos_grasps = pose_grasps[:, :, :3]
            dist_map = torch.norm((pos_pts - pos_grasps), dim=2)

        k = int(0.05 * M)
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(pts[0])
        distances, indices = nbrs.kneighbors(pts[0])
        clip_sims_ = clip_sims[0].cpu().numpy()
        average_affordances = np.array([clip_sims_[idx].mean() for idx in indices])
        max_avg_affordance = average_affordances.max()
        best_point_index = average_affordances.argmax()

        # places = torch.argmin(dist_map[best_point_index])
        
        places = torch.where(dist_map[best_point_index]<0.05)[0]
        if len(places) >= 1:
            places = torch.argmin(dist_map[best_point_index])
        else:
            places = torch.randint(0, actions.shape[1], (1,))

        return int(places.detach().cpu().numpy())