import numpy as np


class MBGrasp(object):
    def __init__(self, args):

        self.device = args.device

    def select_action_random(self, action_set, action_dict, target_objs):
        selected_obj = np.random.choice(target_objs)
        grasps = action_dict[selected_obj]

        if grasps is not None:
            grasp = grasps[np.random.choice(len(grasps))]
            grasp_idx = action_set.index(grasp)
        else:
            grasp_idx = np.random.choice(len(action_set))

        return grasp_idx

    def select_action_greedy(self, action_set, target_obj_poses):     
        target_objs = list(target_obj_poses.keys())
        selected_obj = np.random.choice(target_objs)
        selected_obj_pos = target_obj_poses[selected_obj][:3, 3][np.newaxis, :]

        pose_grasps = np.array(action_set)
        pos_grasps = pose_grasps[:, :3]
        dist_map = np.linalg.norm((selected_obj_pos - pos_grasps), axis=-1)

        grasp_idx = np.argmin(dist_map)

        return grasp_idx
