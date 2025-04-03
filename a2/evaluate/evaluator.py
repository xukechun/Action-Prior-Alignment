import os
import time
import argparse
import numpy as np
import random
import torch
import shutil
import open3d as o3d
import matplotlib.pyplot as plt

import utils.utils as utils
from env.constants import WORKSPACE_LIMITS, GRASP_WORKSPACE_LIMITS, PLACE_WORKSPACE_LIMITS, PP_WORKSPACE_LIMITS, PP_PIXEL_SIZE, WORKSPACE_LIMITS, PP_SHIFT_Y
from helpers.logger import Logger
from feature_extractor.feature_field_builder import FeatureField
from action_generator.grasp_detetor import Graspnet
from action_generator.place_generator import Placenet
from models.bc_agent import ViLGP3D, AdaptViLGP3D, LangEmbViLGP3D
from models.clip_agent import CLIPGrasp, CLIPPlace

class BaseEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.args.device = self.device
        self.args.task_num = 2 if self.args.task_emb else None
        
        # Set random seeds
        random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        
        # Initialize components
        self.init_environment()
        self.init_logger()
        self.init_networks()
        self.init_feature_fields()
        
        # Get camera configurations
        self.extrinsics, self.intrinsics = utils.get_camera_configs(
            self.env, 
            single_view=self.args.single_view, 
            diff_views=self.args.diff_views
        )
        self.logger.save_camera_configs(self.extrinsics, self.intrinsics)
        
        # Initialize evaluation variables
        self.iteration = 0
        self.case = 0
        
    def init_environment(self):
        if not self.args.unseen:
            from env.environment_sim import Environment
        else:
            from env.unseen_environment_sim import Environment
        self.env = Environment(gui=self.args.gui)
        self.env.seed(self.args.seed)
        
    def init_logger(self):
        self.logger = Logger(
            case_dir=self.args.testing_case_dir,
            case=self.args.testing_case,
            suffix=self.args.log_suffix
        )
        
    def init_networks(self):
        raise NotImplementedError("Subclasses must implement init_networks")
        
    def init_feature_fields(self):
        raise NotImplementedError("Subclasses must implement init_feature_fields")
        
    def process_episode(self, episode, file_path):
        raise NotImplementedError("Subclasses must implement process_episode")
        
    def evaluate(self):
        if os.path.exists(self.args.testing_case_dir):
            filelist = os.listdir(self.args.testing_case_dir)
            filelist.sort(key=lambda x:int(x[4:6]))
        else:
            filelist = []
            
        if self.args.testing_case is not None:
            filelist = [self.args.testing_case]
            
        for f in filelist:
            f = os.path.join(self.args.testing_case_dir, f)
            
            # Reset logs for new case
            self.logger.episode_reward_logs = []
            self.logger.episode_step_logs = []
            self.logger.episode_success_logs = []
            
            # Process episodes
            for episode in range(self.args.num_episode):
                self.process_episode(episode, f)
                
            self.case += 1 

class PickEvaluator(BaseEvaluator):
    def __init__(self, args):
        super().__init__(args)
        
    def init_networks(self):
        self.graspnet = Graspnet()
        
        if self.args.direct_grounding:
            self.agent = CLIPGrasp(self.args)
        else:
            if not self.args.adaptive:
                if not self.args.lang_emb:
                    self.agent = ViLGP3D(action_dim=7, args=self.args)
                else:
                    self.agent = LangEmbViLGP3D(action_dim=7, args=self.args)
            else:
                self.agent = AdaptViLGP3D(action_dim=7, args=self.args)
                
            if self.args.load_model:
                self.logger.load_sl_checkpoint(self.agent.vilg3d, self.args.model_path, self.args.evaluate)
                
    def init_feature_fields(self):
        self.field_builder = FeatureField(
            num_cam=3,
            query_threshold=[0.4],
            grid_size=0.004,
            boundaries=WORKSPACE_LIMITS,
            feat_backbone=self.args.feat_backbone,
            device=self.device
        )
        
    def process_episode(self, episode, file_path):
        episode_reward = 0
        episode_steps = 0
        done = False
        reset = False
        
        # Reset environment
        while not reset:
            self.env.reset(workspace=self.args.workspace)
            reset, lang_goal = self.env.add_object_push_from_file(file_path)
            print(f"\033[032m Reset environment of episode {episode}, language goal {lang_goal}\033[0m")
            
        while not done:
            # Check workspace bounds
            out_of_workspace = []
            for obj_id in self.env.target_obj_ids:
                pos, _, _ = self.env.obj_info(obj_id)
                if pos[0] < WORKSPACE_LIMITS[0][0] or pos[0] > WORKSPACE_LIMITS[0][1] \
                    or pos[1] < WORKSPACE_LIMITS[1][0] or pos[1] > WORKSPACE_LIMITS[1][1]:
                    out_of_workspace.append(obj_id)
            if len(out_of_workspace) == len(self.env.target_obj_ids):
                print("\033[031m Target objects are not in the scene!\033[0m")
                break
                
            # Get visual data
            color_images, depth_images, pcd = utils.get_multi_view_images_w_pointcloud(
                self.env,
                visualize=self.args.visualize,
                single_view=self.args.single_view,
                diff_views=self.args.diff_views
            )
            
            # Generate features
            pts = utils.generate_points_for_feature_extraction(pcd, visualize=self.args.visualize)
            feature_list = ['clip_feats', 'clip_sims']
            pts, feat_dict = self.field_builder.generate_feature_field(
                color_images,
                depth_images,
                self.extrinsics,
                self.intrinsics,
                lang_goal,
                feature_list,
                pts,
                visualize=self.args.visualize
            )
            
            if self.args.save_vis:
                self.logger.save_visualizations(self.iteration, [feat_dict['clip_feats_pcd']], suffix="feats")
                self.logger.save_visualizations(self.iteration, [feat_dict['clip_sims_pcd']], suffix="sims")

            # Get grasp poses
            with torch.no_grad():
                grasp_pose_set, _, gg = self.graspnet.grasp_detection(
                    pcd,
                    self.env.get_true_object_poses(),
                    visualize=self.args.visualize
                )
                
            if len(grasp_pose_set) == 0:
                break
                
            # Preprocess data
            sampled_pts, sampled_clip_feats, sampled_clip_sims, grasps, grasp_pose_set = utils.preprocess_pp_unified(
                pts,
                feat_dict,
                grasp_pose_set,
                sample_num=self.args.sample_num,
                sample_action=self.args.sample_grasp,
                visualize=self.args.visualize
            )

            if self.args.save_vis:
                # normalized sampled clip sims
                z = 1e-4
                sampled_clip_sims = (sampled_clip_sims - sampled_clip_sims.min() + z) / (sampled_clip_sims.max() - sampled_clip_sims.min() + z) 

                sampled_pcd = o3d.geometry.PointCloud()
                sampled_pcd.points = o3d.utility.Vector3dVector(sampled_pts[0].detach().cpu().numpy()) # [N, 3]
                cmap = plt.get_cmap("turbo")
                sampled_clip_sims_color = cmap(sampled_clip_sims[0, ..., 0].detach().cpu().numpy())[..., :3]
                sampled_pcd.colors = o3d.utility.Vector3dVector(sampled_clip_sims_color) # [N, 3]
                self.logger.save_visualizations(self.iteration, [sampled_pcd], suffix="sampled_pts")
            
            # Select action
            if len(grasp_pose_set) == 1:
                action_idx = 0
            else:
                if self.args.direct_grounding:
                    with torch.no_grad():
                        action_idx = self.agent.select_action_knn_greedy(sampled_pts, sampled_clip_sims, grasps)
                else:
                    with torch.no_grad():
                        if self.args.normalize:
                            sampled_pts = utils.normalize_pos(sampled_pts, WORKSPACE_LIMITS.T, device=sampled_pts.device)
                            grasps[:, :, :3] = utils.normalize_pos(grasps[:, :, :3], WORKSPACE_LIMITS.T, device=grasps.device)
                            
                        if not self.args.lang_emb:
                            if self.args.adaptive:
                                logits, action_idx = self.agent.select_action(sampled_pts, sampled_clip_feats, sampled_clip_sims, grasps, ratio=self.args.ratio)
                            else:
                                logits, action_idx = self.agent.select_action(sampled_pts, sampled_clip_feats, sampled_clip_sims, grasps)
                        else:
                            logits, action_idx = self.agent.select_action(sampled_pts, sampled_clip_feats, grasps, lang_goal)

                        z = 1e-4
                        gg.scores = (logits[0] - logits[0].min() + z) / (logits[0].max() - logits[0].min() + z)
                        if self.args.visualize:
                            # !! Note that if sample grasp, gg should be sampled too!!
                            print("predicted logits of agent!!!")
                            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                            o3d.visualization.draw_geometries([frame, pcd, *gg.to_open3d_geometry_list()])

                            topk = np.argsort(logits[0])[-1:]
                            gg = gg[topk]
                            o3d.visualization.draw_geometries([frame, pcd, *gg.to_open3d_geometry_list()])

                        if self.args.save_vis:
                            topk = np.argsort(logits[0])[-1:]
                            gg = gg[topk]
                            self.logger.save_visualizations(self.iteration, [pcd, *gg.to_open3d_geometry_list()], suffix="action")

            # Execute action
            action = grasp_pose_set[action_idx]
            reward, done = self.env.step(action)
            
            # Update logs
            self.iteration += 1
            episode_steps += 1
            episode_reward += reward
            
            # Log progress
            print("\033[034m Episode: {}, step: {}, reward: {}\033[0m".format(episode, episode_steps, round(reward, 2)))
            
            if episode_steps == self.args.max_episode_step:
                break
            
        self.logger.episode_reward_logs.append(episode_reward)
        self.logger.episode_step_logs.append(episode_steps)
        self.logger.episode_success_logs.append(done)
        
        # Write logs
        self.logger.write_to_log('episode_reward', self.logger.episode_reward_logs)
        self.logger.write_to_log('episode_step', self.logger.episode_step_logs)
        self.logger.write_to_log('episode_success', self.logger.episode_success_logs)
        
        print("\033[034m Episode: {}, episode steps: {}, episode reward: {}, success: {}\033[0m".format(
            episode, episode_steps, round(episode_reward, 2), done
        ))

        if episode == self.args.num_episode - 1:
            self.save_case_results(lang_goal)
        
    def save_case_results(self, lang_goal):
        avg_success = sum(self.logger.episode_success_logs)/len(self.logger.episode_success_logs)
        avg_reward = sum(self.logger.episode_reward_logs)/len(self.logger.episode_reward_logs)
        avg_step = sum(self.logger.episode_step_logs)/len(self.logger.episode_step_logs)
        
        success_steps = []
        for i in range(len(self.logger.episode_success_logs)):
            if self.logger.episode_success_logs[i]:
                success_steps.append(self.logger.episode_step_logs[i])
        if len(success_steps) > 0:
            avg_success_step = sum(success_steps) / len(success_steps)
        else:
            avg_success_step = 1000
            
        result_file = os.path.join(self.logger.result_directory, "case" + str(self.case) + ".txt")
        with open(result_file, "w") as out_file:
            out_file.write(
                "%s %.18e %.18e %.18e %.18e\n"
                % (
                    lang_goal,
                    avg_success,
                    avg_step,
                    avg_success_step,
                    avg_reward,
                )
            )
        print("\033[034m Language goal: {}, average steps: {}/{}, average reward: {}, average success: {}\033[0m".format(
            lang_goal, avg_step, avg_success_step, avg_reward, avg_success
        ))

class PlaceEvaluator(BaseEvaluator):
    def __init__(self, args):
        super().__init__(args)
        
    def init_networks(self):        
        self.placenet = Placenet()
        
        if self.args.direct_grounding:
            self.agent = CLIPPlace(self.args)
        else:
            if not self.args.adaptive:
                if not self.args.lang_emb:
                    self.agent = ViLGP3D(action_dim=7, args=self.args)
                else:
                    self.agent = LangEmbViLGP3D(action_dim=7, args=self.args)
            else:
                self.agent = AdaptViLGP3D(action_dim=7, args=self.args)
                
            if self.args.load_model:
                self.logger.load_sl_checkpoint(self.agent.vilg3d, self.args.model_path, self.args.evaluate)
                
    def init_feature_fields(self):
        self.field_builder = FeatureField(
            num_cam=3,
            query_threshold=[0.4],
            grid_size=0.004,
            boundaries=WORKSPACE_LIMITS,
            feat_backbone=self.args.feat_backbone,
            device=self.device
        )
        
    def process_episode(self, episode, file_path):
        episode_reward = 0
        episode_steps = 0
        done = False
        reset = False
        reset_times = 0
        
        # Reset environment
        while not reset:
            self.env.reset(workspace=self.args.workspace)
            reset, lang_goal = self.env.add_object_push_from_place_file(file_path)
            print(f"\033[032m Reset environment of episode {episode}, language goal {lang_goal}\033[0m")
            if not reset:
                reset_times += 1
            if reset_times >= 3:
                break
                
        if reset_times >= 3:
            return
            
        while not done:
            # Check workspace bounds
            out_of_workspace = []
            for obj_id in self.env.reference_obj_ids:
                pos, _, _ = self.env.obj_info(obj_id)
                if pos[0] < WORKSPACE_LIMITS[0][0] or pos[0] > WORKSPACE_LIMITS[0][1] \
                    or pos[1] < WORKSPACE_LIMITS[1][0] or pos[1] > WORKSPACE_LIMITS[1][1]:
                    out_of_workspace.append(obj_id)
                    
            if len(self.env.reference_obj_ids) > 0 and len(out_of_workspace) == len(self.env.reference_obj_ids):
                print("\033[031m Target objects are not in the scene!\033[0m")
                break
                
            # Get visual data
            color_images, depth_images, pcd = utils.get_multi_view_images_w_pointcloud(
                self.env,
                visualize=self.args.visualize,
                single_view=self.args.single_view,
                diff_views=self.args.diff_views
            )
            
            # Get heightmap and bboxes
            color_heightmap, depth_heightmap, mask_heightmap = utils.get_true_heightmap(self.env)
            bbox_ids, bbox_images, bbox_sizes, bbox_centers, bbox_positions = utils.get_true_bboxes(
                self.env, color_heightmap, depth_heightmap, mask_heightmap
            )
            bbox_ids, remain_bbox_images, bbox_sizes, bbox_centers, bbox_positions = utils.preprocess_bboxes(
                bbox_ids, bbox_images, bbox_sizes, bbox_centers, bbox_positions
            )
            
            # Generate features
            pts = utils.generate_points_for_feature_extraction(pcd, cut_table=False, visualize=self.args.visualize)
            feature_list = ['clip_feats', 'clip_sims']
            pts, feat_dict = self.field_builder.generate_feature_field(
                color_images,
                depth_images,
                self.extrinsics,
                self.intrinsics,
                lang_goal,
                feature_list,
                pts,
                last_text_feature=False,
                visualize=self.args.visualize
            )
                    
            # Generate place poses
            all_place_valid_mask = utils.generate_all_place_dist(
                self.env.reference_obj_ids[0],
                self.env.obj_labels[self.env.reference_obj_ids[0]][0],
                self.env.reference_obj_dirs[0],
                None,
                self.env.obj_labels,
                bbox_ids,
                bbox_centers,
                bbox_sizes
            )

            # Get reference object information
            ref_obj_centers = []
            ref_regions = []
            for obj_id in self.env.reference_obj_ids:
                if obj_id in bbox_ids:
                    bbox_idx = bbox_ids.index(obj_id)
                    ref_obj_centers.append(bbox_centers[bbox_idx])
                    obj_idx = self.env.reference_obj_ids.index(obj_id)
                    ref_regions.append(self.env.reference_obj_dirs[obj_idx])
            
            place_pixels, place_pose_set, pp, valid_places_list = self.placenet.place_generation_return_gt(
                depth_heightmap,
                bbox_centers,
                bbox_sizes,
                ref_obj_centers,
                ref_regions,
                valid_place_mask=all_place_valid_mask,
                grasped_obj_size=None,
                sample_num_each_object=3,
                topdown_place_rot=self.args.topdown_place_rot,
                action_variance=self.args.action_var
            )
            
            if len(valid_places_list) == 0:
                print("\033[031m No valid places!\033[0m")
                break
                
            # Preprocess data
            sampled_pts, sampled_clip_feats, sampled_clip_sims, places, place_pose_set = utils.preprocess_pp_unified(
                pts,
                feat_dict,
                place_pose_set,
                sample_num=self.args.sample_num,
                sample_action=self.args.sample_place,
                visualize=self.args.visualize
            )
            
            # Select action
            if len(place_pose_set) == 1:
                action_idx = 0
            else:
                if self.args.direct_grounding:
                    with torch.no_grad():
                        action_idx = self.agent.select_action_knn_greedy(sampled_pts, sampled_clip_sims, places)
                else:
                    with torch.no_grad():
                        if self.args.normalize:
                            sampled_pts = utils.normalize_pos(sampled_pts, WORKSPACE_LIMITS.T, device=sampled_pts.device)
                            places[:, :, :3] = utils.normalize_pos(places[:, :, :3], WORKSPACE_LIMITS.T, device=places.device)
                            
                        if not self.args.lang_emb:
                            if self.args.adaptive:
                                logits, action_idx = self.agent.select_action(sampled_pts, sampled_clip_feats, sampled_clip_sims, places, ratio=self.args.ratio)
                            else:
                                logits, action_idx = self.agent.select_action(sampled_pts, sampled_clip_feats, sampled_clip_sims, places)
                        else:
                            logits, action_idx = self.agent.select_action(sampled_pts, sampled_clip_feats, places, lang_goal)

                        if self.args.visualize:
                            print("predicted logits of agent!!!")
                            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                            o3d.visualization.draw_geometries([frame, pcd, *pp.to_open3d_geometry_list()])

                            z = 1e-4
                            pp.scores = (logits[0] - logits[0].min() + z) / (logits[0].max() - logits[0].min() + z)
                            topk = np.argsort(logits[0])[-1:]
                            pp = pp[topk]
                                    
                            # !! Note that if sample place, pps should be sampled too!!
                            # print("predicted logits of agent!!!")
                            o3d.visualization.draw_geometries([frame, pcd, *pp.to_open3d_geometry_list()])
                            
                        if self.args.save_vis:
                            z = 1e-4
                            pp.scores = (logits[0] - logits[0].min() + z) / (logits[0].max() - logits[0].min() + z)
                            topk = np.argsort(logits[0])[-1:]
                            pp = pp[topk]
                            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                            self.logger.save_visualizations(self.iteration, [pcd, *pp.to_open3d_geometry_list()])

            # Execute action
            action = place_pose_set[action_idx]
            if action_idx in valid_places_list:
                done = True
                reward = 1
            else:
                reward = 0
                
            # Update logs
            self.iteration += 1
            episode_steps += 1
            episode_reward += reward
            
            # Log progress
            print("\033[034m Episode: {}, total numsteps: {}\033[0m".format(episode, self.iteration))
            
            if episode_steps == self.args.max_episode_step:
                break
                
        # Save episode results
        self.logger.episode_reward_logs.append(episode_reward)
        self.logger.episode_step_logs.append(episode_steps)
        self.logger.episode_success_logs.append(done)
        
        # Write logs
        self.logger.write_to_log('episode_reward', self.logger.episode_reward_logs)
        self.logger.write_to_log('episode_step', self.logger.episode_step_logs)
        self.logger.write_to_log('episode_success', self.logger.episode_success_logs)
        
        print("\033[034m Episode: {}, total numsteps: {}, episode steps: {}, episode reward: {}, success: {}\033[0m".format(
            episode, self.iteration, episode_steps, round(episode_reward, 2), done
        ))

        if episode == self.args.num_episode - 1:
            self.save_case_results(lang_goal)
        
    def save_case_results(self, lang_goal):
        avg_success = sum(self.logger.episode_success_logs)/len(self.logger.episode_success_logs)
        avg_reward = sum(self.logger.episode_reward_logs)/len(self.logger.episode_reward_logs)
        avg_step = sum(self.logger.episode_step_logs)/len(self.logger.episode_step_logs)
        
        success_steps = []
        for i in range(len(self.logger.episode_success_logs)):
            if self.logger.episode_success_logs[i]:
                success_steps.append(self.logger.episode_step_logs[i])
        if len(success_steps) > 0:
            avg_success_step = sum(success_steps) / len(success_steps)
        else:
            avg_success_step = 1000
            
        result_file = os.path.join(self.logger.result_directory, "case" + str(self.case) + ".txt")
        with open(result_file, "w") as out_file:
            out_file.write(
                "%s %.18e %.18e %.18e %.18e\n"
                % (
                    lang_goal,
                    avg_success,
                    avg_step,
                    avg_success_step,
                    avg_reward,
                )
            )
        print("\033[034m Language goal: {}, average steps: {}/{}, average reward: {}, average success: {}\033[0m".format(
            lang_goal, avg_step, avg_success_step, avg_reward, avg_success
        ))

class PickPlaceEvaluator(BaseEvaluator):
    def __init__(self, args):
        super().__init__(args)
        
    def init_networks(self):
        self.graspnet = Graspnet()
        self.placenet = Placenet()
        
        if self.args.direct_grounding:
            self.grasp_agent = CLIPGrasp(self.args)
            self.place_agent = CLIPPlace(self.args)
        else:
            # load vision-language-action model
            if not self.args.adaptive:
                if not self.args.lang_emb:
                    self.agent = ViLGP3D(action_dim=7, args=self.args)
                else:
                    self.agent = LangEmbViLGP3D(action_dim=7, args=self.args)
            else:
                self.agent = AdaptViLGP3D(action_dim=7, args=self.args)
                
            if self.args.load_model:
                self.logger.load_sl_checkpoint(self.agent.vilg3d, self.args.model_path, self.args.evaluate)
                
    def init_feature_fields(self):
        self.grasp_field_builder = FeatureField(
            num_cam=3,
            query_threshold=[0.4],
            grid_size=0.004,
            boundaries=WORKSPACE_LIMITS,
            feat_backbone=self.args.feat_backbone,
            device=self.device
        )
        
        self.place_field_builder = FeatureField(
            num_cam=3,
            query_threshold=[0.4],
            grid_size=0.004,
            boundaries=WORKSPACE_LIMITS,
            feat_backbone=self.args.feat_backbone,
            device=self.device
        )
        
    def process_episode(self, episode, file_path):
        episode_reward = 0
        episode_steps = 0
        reset = False
        reset_times = 0
        grasp_done = False
        place_done = False

        # Reset environment
        while not reset:
            self.env.reset(workspace=self.args.workspace)
            # load grasp scenarios
            reset, grasp_lang_goal, _ = self.env.add_object_push_from_pickplace_file(file_path, mode="grasp")
            print(f"\033[032m Reset environment of episode {episode}, grasp language goal {grasp_lang_goal}\033[0m")

            # remove objects out of workspace for grasp 
            out_of_workspace = []
            for obj_id in self.env.obj_ids["rigid"]:
                pos, _, _ = self.env.obj_info(obj_id)
                if pos[0] < GRASP_WORKSPACE_LIMITS[0][0] or pos[0] > GRASP_WORKSPACE_LIMITS[0][1] \
                    or pos[1] < GRASP_WORKSPACE_LIMITS[1][0] or pos[1] > GRASP_WORKSPACE_LIMITS[1][1]:
                    print("\033[031m Delete objects out of workspace!\033[0m")
                    self.env.remove_object_id(obj_id)    

            # load place scenarios
            reset, _, place_lang_goal = self.env.add_object_push_from_pickplace_file(file_path, mode="place")
            print(f"\033[032m Reset environment of episode {episode}, place language goal {place_lang_goal}\033[0m")

            # remove objects out of workspace for place 
            out_of_workspace = []
            for obj_id in self.env.obj_ids["rigid"]:
                if obj_id >= 19:
                    pos, _, _ = self.env.obj_info(obj_id)
                    if pos[0] < PLACE_WORKSPACE_LIMITS[0][0] or pos[0] > PLACE_WORKSPACE_LIMITS[0][1] \
                        or pos[1] < PLACE_WORKSPACE_LIMITS[1][0] or pos[1] > PLACE_WORKSPACE_LIMITS[1][1]:
                        print("\033[031m Delete objects out of workspace!\033[0m")
                        self.env.remove_object_id(obj_id)

            if not reset:
                reset_times += 1
            if reset_times >= 3:
                break
                
        if reset_times >= 3:
            return
            
        while not grasp_done:
            # check if one of the target objects is in the workspace:
            for obj_id in self.env.target_obj_ids:
                pos, _, _ = self.env.obj_info(obj_id)
                if pos[0] < GRASP_WORKSPACE_LIMITS[0][0] or pos[0] > GRASP_WORKSPACE_LIMITS[0][1] \
                    or pos[1] < GRASP_WORKSPACE_LIMITS[1][0] or pos[1] > GRASP_WORKSPACE_LIMITS[1][1]:
                    self.env.target_obj_ids.remove(obj_id)
                    
            if len(self.env.target_obj_ids) == 0:
                print("\033[031m Target objects are not in the scene!\033[0m")
                break     

            self.env.bounds = PP_WORKSPACE_LIMITS
            self.env.pixel_size = PP_PIXEL_SIZE   
                
            # Get visual data
            color_images, depth_images, pcd = utils.get_multi_view_images_w_pointcloud(
                self.env,
                visualize=self.args.visualize,
                single_view=self.args.single_view,
                diff_views=self.args.diff_views
            )

            # get grasp pcd
            grasp_pcd = utils.filter_pcd(pcd, GRASP_WORKSPACE_LIMITS, visualize=self.args.visualize)
            # get place pcd
            place_pcd = utils.filter_pcd(pcd, PLACE_WORKSPACE_LIMITS, visualize=self.args.visualize)            

            
            # Get heightmap and bboxes
            color_heightmap, depth_heightmap, mask_heightmap = utils.get_true_heightmap(self.env)
            bbox_ids, bbox_images, bbox_sizes, bbox_centers, bbox_positions = utils.get_true_bboxes(
                self.env, color_heightmap, depth_heightmap, mask_heightmap
            )
            bbox_ids, remain_bbox_images, bbox_sizes, bbox_centers, bbox_positions = utils.preprocess_bboxes(
                bbox_ids, bbox_images, bbox_sizes, bbox_centers, bbox_positions
            )

            # graspnet
            # Note: simply replace object poses to object bounding boxes maybe ok
            with torch.no_grad():
                grasp_pose_set, grasp_pose_dict, gg = self.graspnet.grasp_detection(grasp_pcd, self.env.get_true_object_poses(), visualize=self.args.visualize)  # 1.19s
            print("Number of grasping poses", len(grasp_pose_set))
            if len(grasp_pose_set) == 0:
                print("\033[031m No grasp poses!\033[0m")   
                break

            grasp_pts = utils.generate_points_for_feature_extraction(grasp_pcd, visualize=self.args.visualize)

            feature_list = ['clip_feats', 'clip_sims']
            grasp_pts, feat_dict = self.grasp_field_builder.generate_feature_field(
                color_images, 
                depth_images, 
                self.extrinsics, 
                self.intrinsics, 
                grasp_lang_goal, 
                feature_list, 
                grasp_pts, 
                last_text_feature=False, 
                visualize=self.args.visualize) 

            # preprocess
            sampled_pts, sampled_clip_feats, sampled_clip_sims, grasps, grasp_pose_set = utils.preprocess_pp_unified(
                grasp_pts, 
                feat_dict, 
                grasp_pose_set, 
                sample_action=self.args.sample_grasp, 
                sample_num=self.args.sample_num, 
                visualize=self.args.visualize)

            if len(grasp_pose_set) == 1:
                action_idx = 0
            else:
                if self.args.direct_grounding:
                    with torch.no_grad():
                        action_idx = self.grasp_agent.select_action_knn_greedy(sampled_pts, sampled_clip_sims, grasps)
                else:
                    with torch.no_grad():
                        if self.args.normalize:
                            sampled_pts = utils.normalize_pos(sampled_pts, GRASP_WORKSPACE_LIMITS.T, device=sampled_pts.device)
                            grasps[:, :, :3] = utils.normalize_pos(grasps[:, :, :3], GRASP_WORKSPACE_LIMITS.T, device=grasps.device)
                        if self.args.workspace_shift:
                            sampled_pts = sampled_pts + torch.Tensor([[0, PP_SHIFT_Y, 0]]).to(sampled_pts.device)
                            grasps[:, :, :3] = grasps[:, :, :3] + torch.Tensor([[0, PP_SHIFT_Y, 0]]).to(grasps.device)
                            
                                                            
                        if not self.args.lang_emb:
                            logits, action_idx = self.agent.select_action(sampled_pts, sampled_clip_feats, sampled_clip_sims, grasps)
                        else:
                            logits, action_idx = self.agent.select_action(sampled_pts, sampled_clip_feats, grasps, grasp_lang_goal)

                        num = np.max((logits.shape[1], 10))
                        topk = np.argsort(logits[0])[-num:]
                        gg = gg[topk]
                                                
                        z = 1e-4
                        # gg.scores = (logits[0] - logits[0].min() + z) / (logits[0].max() - logits[0].min() + z)
                        gg.scores = (logits[0][topk] - logits[0][topk].min() + z) / (logits[0][topk].max() - logits[0][topk].min() + z)
                        
                        if self.args.visualize:
                            # !! Note that if sample grasp, ggs should be sampled too!!
                            print("predicted logits of agent!!!")
                            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                            o3d.visualization.draw_geometries([frame, pcd, *gg.to_open3d_geometry_list()])
                            
            action = grasp_pose_set[action_idx]
            grasp_success, grasped_obj_id, pos_dist = self.env.grasp(action, follow_place=True)
            if not grasp_success:
                reward = -1
            else:
                if grasped_obj_id in self.env.target_obj_ids:
                    reward = 2
                    grasp_done = True
                else:
                    reward = 0
                    
            if grasped_obj_id not in self.env.target_obj_ids:
                place_success = self.env.place_out_of_workspace()

            episode_steps += 1
            self.iteration += 1
            episode_reward += reward
            print("\033[034m Episode stage of grasp: {}, total numsteps: {}, reward: {}\033[0m".format(episode, self.iteration, round(reward, 2)))

            # record
            self.logger.reward_logs.append(reward)
            self.logger.executed_action_logs.append(action)
            self.logger.write_to_log('reward', self.logger.reward_logs)
            self.logger.write_to_log('executed_action', self.logger.executed_action_logs)
            
            if grasp_done or episode_steps == self.args.max_episode_step:
                break

        if grasp_done:    
            while not place_done:
                # check if one of the target objects is in the workspace:
                for obj_id in self.env.reference_obj_ids:
                    pos, _, _ = self.env.obj_info(obj_id)
                    if pos[0] < PLACE_WORKSPACE_LIMITS[0][0] or pos[0] > PLACE_WORKSPACE_LIMITS[0][1] \
                        or pos[1] < PLACE_WORKSPACE_LIMITS[1][0] or pos[1] > PLACE_WORKSPACE_LIMITS[1][1]:
                        self.env.reference_obj_ids.remove(obj_id)
            
                if len(self.env.reference_obj_ids) == 0:
                    print("\033[031m Reference objects are not in the scene!\033[0m")
                    break

                # get reference object information
                ref_obj_centers = []
                ref_regions = []
                for obj_id in self.env.reference_obj_ids:
                    if obj_id in bbox_ids:
                        bbox_idx = bbox_ids.index(obj_id)
                        ref_obj_centers.append(bbox_centers[bbox_idx])
                        obj_idx = self.env.reference_obj_ids.index(obj_id)
                        ref_regions.append(self.env.reference_obj_dirs[obj_idx])
                
                # !!! filter place box here !!!
                # bbox_ids, remain_bbox_images, bbox_sizes, bbox_centers, bbox_positions
                place_bbox_sizes = []
                place_bbox_centers = []
                target_obj_size = [5, 5]
                for i in range(len(bbox_ids)):
                    obj_id = bbox_ids[i]
                    if obj_id >= 19:
                        place_bbox_sizes.append(bbox_sizes[i])
                        place_bbox_centers.append(bbox_centers[i])
                    if obj_id == grasped_obj_id:
                        target_obj_size = bbox_sizes[i]
                        
                # placenet, generate all feasible places                 
                all_place_valid_mask = utils.generate_all_place_dist(
                    self.env.reference_obj_ids[0], 
                    self.env.obj_labels[self.env.reference_obj_ids[0]][0], 
                    self.env.reference_obj_dirs[0], 
                    None, 
                    self.env.obj_labels, 
                    bbox_ids, 
                    bbox_centers, 
                    bbox_sizes, 
                    grasped_obj_size=target_obj_size, 
                    pixel_size=PP_PIXEL_SIZE, 
                    visualize=self.args.visualize)
                
                place_pixels, place_pose_set, pp, valid_places_list = self.placenet.place_generation_return_gt(
                    depth_heightmap, 
                    place_bbox_centers, 
                    place_bbox_sizes, 
                    ref_obj_centers, 
                    ref_regions, 
                    valid_place_mask=all_place_valid_mask, 
                    grasped_obj_size=target_obj_size, 
                    sample_num_each_object=3, 
                    workspace_limits=PP_WORKSPACE_LIMITS, 
                    pixel_size=PP_PIXEL_SIZE, 
                    action_variance=self.args.action_var)
                
                
                if len(valid_places_list) == 0:
                    print("\033[031m Nonvalid place for this scene!\033[0m")
                    break
                
                place_pts = utils.generate_points_for_feature_extraction(place_pcd, cut_table=False, visualize=self.args.visualize)

                # generated feature
                feature_list = ['clip_feats', 'clip_sims']
                # pts, feat_dict = field_builer.generate_feature_field(color_images, depth_images, extrinsics, intrinsics, lang_goal, feature_list, args.visualize)  # 7.33s
                place_pts, feat_dict = self.place_field_builder.generate_feature_field(
                    color_images, depth_images, 
                    self.extrinsics, 
                    self.intrinsics, 
                    place_lang_goal, 
                    feature_list, 
                    place_pts, 
                    last_text_feature=False, 
                    visualize=self.args.visualize) 
                    
                # preprocess
                sampled_pts, sampled_clip_feats, sampled_clip_sims, places, place_pose_set = utils.preprocess_pp_unified(
                    place_pts, 
                    feat_dict, 
                    place_pose_set, 
                    sample_num=self.args.sample_num, 
                    sample_action=self.args.sample_place, 
                    visualize=self.args.visualize)
                

                if len(place_pose_set) == 1:
                    action_idx = 0
                else:
                    if self.args.direct_grounding:
                        with torch.no_grad():
                            # action_idx = place_agent.select_action_greedy(sampled_pts, sampled_clip_sims, places)
                            action_idx = self.place_agent.select_action_knn_greedy(sampled_pts, sampled_clip_sims, places)
                    else:
                        with torch.no_grad():
                            if self.args.normalize:
                                sampled_pts = utils.normalize_pos(sampled_pts, PLACE_WORKSPACE_LIMITS.T, device=sampled_pts.device)
                                places[:, :, :3] = utils.normalize_pos(places[:, :, :3], PLACE_WORKSPACE_LIMITS.T, device=places.device)
                            if self.args.workspace_shift:
                                sampled_pts = sampled_pts - torch.Tensor([[0, PP_SHIFT_Y, 0]]).to(sampled_pts.device)
                                places[:, :, :3] = places[:, :, :3] - torch.Tensor([[0, PP_SHIFT_Y, 0]]).to(places.device)
                                
                            if not self.args.lang_emb:
                                logits, action_idx = self.agent.select_action(sampled_pts, sampled_clip_feats, sampled_clip_sims, places)
                            else:
                                logits, action_idx = self.agent.select_action(sampled_pts, sampled_clip_feats, places, place_lang_goal)

                            num = np.max((logits.shape[1], 1))
                            topk = np.argsort(logits[0])[-num:]
                            pp = pp[topk]
                                                    
                            z = 1e-4
                            pp.scores = (logits[0][topk] - logits[0][topk].min() + z) / (logits[0][topk].max() - logits[0][topk].min() + z)
                            
                            if self.args.visualize:
                                # !! Note that if sample place, pps should be sampled too!!
                                print("predicted logits of agent!!!")
                                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
                                o3d.visualization.draw_geometries([frame, place_pcd, *pp.to_open3d_geometry_list()])

                action = place_pose_set[action_idx]
                if action_idx in valid_places_list:
                    place_done = True
                    reward = 2
                else:
                    reward = 0
                # episode_steps += 1
                self.iteration += 1
                episode_reward += reward
                print("\033[034m Episode stage of place: {}, total numsteps: {}, reward: {}\033[0m".format(episode, self.iteration, round(reward, 2)))

                # record
                self.logger.reward_logs.append(reward)
                self.logger.executed_action_logs.append(action)
                self.logger.write_to_log('reward', self.logger.reward_logs)
                self.logger.write_to_log('executed_action', self.logger.executed_action_logs)
                
                break
                
        # Save episode results
        done = grasp_done and place_done
        self.logger.episode_grasp_success_logs.append(grasp_done)
        self.logger.episode_place_success_logs.append(place_done)
        self.logger.episode_success_logs.append(done)
        self.logger.episode_step_logs.append(episode_steps)
        
        # Write logs
        self.logger.write_to_log('episode_grasp_success', self.logger.episode_grasp_success_logs)
        self.logger.write_to_log('episode_place_success', self.logger.episode_place_success_logs)
        self.logger.write_to_log('episode_step', self.logger.episode_step_logs)
        self.logger.write_to_log('episode_success', self.logger.episode_success_logs)
        
        print("\033[034m Episode: {}, total numsteps: {}, episode steps: {}, episode reward: {}, success: {}\033[0m".format(
            episode, self.iteration, episode_steps, round(episode_reward, 2), done))

        if episode == self.args.num_episode - 1:
            self.save_case_results(grasp_lang_goal, place_lang_goal)
        
    def save_case_results(self, grasp_lang_goal, place_lang_goal):
        avg_success = sum(self.logger.episode_success_logs)/len(self.logger.episode_success_logs)
        avg_grasp_success = sum(self.logger.episode_grasp_success_logs)/len(self.logger.episode_grasp_success_logs)
        avg_place_success = sum(self.logger.episode_place_success_logs)/len(self.logger.episode_place_success_logs)
        avg_step = sum(self.logger.episode_step_logs)/len(self.logger.episode_step_logs)
        
        success_steps = []
        for i in range(len(self.logger.episode_success_logs)):
            if self.logger.episode_success_logs[i]:
                success_steps.append(self.logger.episode_step_logs[i])
        if len(success_steps) > 0:
            avg_success_step = sum(success_steps) / len(success_steps)
        else:
            avg_success_step = 1000

        result_file = os.path.join(self.logger.result_directory, "case" + str(self.case) + ".txt")
        with open(result_file, "w") as out_file:
            out_file.write(
                "%s %s %.18e %.18e %.18e %.18e %.18e\n"
                % (
                    grasp_lang_goal,
                    place_lang_goal,
                    avg_success,
                    avg_grasp_success,
                    avg_place_success,
                    avg_step,
                    avg_success_step,
                )
            )
        print("\033[034m Language goal: {}+{}, average steps: {}/{}, average success: {}\033[0m".format(grasp_lang_goal, place_lang_goal, avg_step, avg_success_step, avg_success))

    def evaluate(self):
        if os.path.exists(self.args.testing_case_dir):
            filelist = os.listdir(self.args.testing_case_dir)
            filelist.sort(key=lambda x:int(x[4:6]))
        else:
            filelist = []
            
        if self.args.testing_case is not None:
            filelist = [self.args.testing_case]
            
        for f in filelist:
            f = os.path.join(self.args.testing_case_dir, f)
            
            # Reset logs for new case
            self.logger.episode_reward_logs = []
            self.logger.episode_step_logs = []
            self.logger.episode_success_logs = []
            self.logger.create_log("episode_grasp_success_logs")
            self.logger.create_log("episode_place_success_logs")
            
            # Process episodes
            for episode in range(self.args.num_episode):
                self.process_episode(episode, f)
                
            self.case += 1 