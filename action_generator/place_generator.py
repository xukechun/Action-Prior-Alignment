import numpy as np
from env.constants import WORKSPACE_LIMITS, PIXEL_SIZE, IMAGE_SIZE, MIN_DIS, MAX_DIS, IN_REGION, OUT_REGION
from utils.utils import generate_mask_map, generate_sector_mask
from graspnetAPI import GraspGroup, Grasp
from scipy.spatial.transform import Rotation as R
import utils.utils as utils

class Placenet:
    def __init__(self):
        pass
       
    def place_generation_return_gt(self, depth, obj_centers, obj_sizes, ref_obj_centers, ref_regions, valid_place_mask=None, grasped_obj_size=None, \
                                    sample_num_each_object=5, workspace_limits=WORKSPACE_LIMITS, pixel_size=PIXEL_SIZE, topdown_place_rot=False, \
                                    action_variance=False):
        mask_map = generate_mask_map(obj_centers, obj_sizes)
        
        sample_inds = None
        # sample_num_each_object = 5
        valid_places_list = []
        # uniformly generate pixels on the objects and near the objects
        for i in range(len(obj_centers)):                
            in_mask = generate_sector_mask((IMAGE_SIZE, IMAGE_SIZE), obj_centers[i], angle_range=[0, 359.99], \
                                            radius_min=0, radius_max=min(obj_sizes[i])//3) # 2
            in_mask_indices = np.argwhere(in_mask)
            
            if in_mask_indices.shape[0] > 0:
                min_sample_num_each_object = min(in_mask_indices.shape[0], sample_num_each_object)
                if action_variance:
                    in_sample_indices = in_mask_indices[np.random.choice(in_mask_indices.shape[0], min_sample_num_each_object, replace=False)]
                else:
                    in_sample_indices = in_mask_indices[np.linspace(0, in_mask_indices.shape[0] - 1, min_sample_num_each_object, dtype=int)]
                
                if sample_inds is None:
                    sample_inds = in_sample_indices
                else:
                    sample_inds = np.concatenate((sample_inds, in_sample_indices), axis=0)
                
                # !!! Note that if there is not valid place mask, this function only takes the first region !!!
                # !!! valid place mask is important if there are interactions of "around" region !!!
                if valid_place_mask is None:
                    if obj_centers[i] in ref_obj_centers and ref_regions[0] in IN_REGION:
                        valid_places_list += list(range(sample_inds.shape[0] - min_sample_num_each_object, sample_inds.shape[0]))            
            
            if grasped_obj_size is None:    
                out_mask = generate_sector_mask((IMAGE_SIZE, IMAGE_SIZE), obj_centers[i], angle_range=[0, 359.99], \
                                                radius_min=max(MIN_DIS//pixel_size, max(obj_sizes[i])//2), radius_max=max(MAX_DIS//pixel_size, max(obj_sizes[i])//2 + 0.1))
            else:
                out_mask = generate_sector_mask((IMAGE_SIZE, IMAGE_SIZE), obj_centers[i], angle_range=[0, 359.99], \
                                                radius_min=max(MIN_DIS//pixel_size, max(obj_sizes[i])//2 + max(grasped_obj_size)//2), radius_max=max(MAX_DIS//pixel_size, max(obj_sizes[i])//2 + max(grasped_obj_size)//2 + 0.1)) 
                
            out_mask = out_mask * mask_map
            out_mask_indices = np.argwhere(out_mask)
 
            if out_mask_indices.shape[0] > 0:
                min_sample_num_each_object = min(out_mask_indices.shape[0], sample_num_each_object)    
                if action_variance:           
                    out_sample_indices = out_mask_indices[np.random.choice(out_mask_indices.shape[0], min_sample_num_each_object, replace=False)]
                else:
                    out_sample_indices = out_mask_indices[np.linspace(0, out_mask_indices.shape[0] - 1, min_sample_num_each_object, dtype=int)]
                
                if sample_inds is None:
                    sample_inds = out_sample_indices
                else:
                    sample_inds = np.concatenate((sample_inds, out_sample_indices), axis=0)

                if valid_place_mask is None:
                    if obj_centers[i] in ref_obj_centers and ref_regions[0] in OUT_REGION:
                        valid_places_list += list(range(sample_inds.shape[0] - min_sample_num_each_object, sample_inds.shape[0]))   
        
        if valid_place_mask is not None:
            valid_places_list = list(np.where(valid_place_mask[sample_inds[:, 0], sample_inds[:, 1]])[0])
        
        if sample_inds is not None:
            sample_inds = sample_inds.transpose(1, 0)
            pos_z = depth[sample_inds[0, :], sample_inds[1, :]]
        
            
            pos_x = sample_inds[0, :] * pixel_size + workspace_limits[0][0]
            pos_y = sample_inds[1, :] * pixel_size + workspace_limits[1][0]
            place_points = np.float32([pos_x, pos_y, pos_z]).transpose(1, 0)
        
            if topdown_place_rot:
                place_poses = np.concatenate((place_points, np.array([[0.707, 0, -0.707, 0]]).repeat(place_points.shape[0], axis=0)), axis=1)
            else:
                place_poses = np.concatenate((place_points, np.array([[0, 0, 0, 1]]).repeat(place_points.shape[0], axis=0)), axis=1)

            place_pose_list = [place_poses[i] for i in range(len(place_poses))]
        else:
            place_pose_list = []
        
        place_array = []
        for place_pose in place_pose_list:
            pp = Grasp()
            pp.translation = place_pose[:3]
            vis_quat = np.array([[0.707, 0, -0.707, 0]])
            pp.width = 0.04
            # pp.rotation_matrix = R.from_quat(place_pose[-4:]).as_matrix()
            pp.rotation_matrix = R.from_quat(vis_quat).as_matrix()
            place_array.append(pp.grasp_array)
        
        pps = GraspGroup(np.array(place_array))
        
        return sample_inds, place_pose_list, pps, valid_places_list
    
    def quat_multiply(self, q1, q2):
  
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        return np.array([w, x, y, z])

    def generate_quat(self, theta):
        quat_x_rotation = np.array([np.sqrt(2)/2, np.sqrt(2)/2, 0, 0])
        
        # rotate theta across z axis
        half_theta = theta / 2.0
        quat_z_rotation = np.array([np.cos(half_theta), 0, 0, np.sin(half_theta)])
        
        quat_result = self.quat_multiply(quat_x_rotation, quat_z_rotation)
        
        return quat_result