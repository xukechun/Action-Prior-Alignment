import os
import time
import argparse
import numpy as np
import random
import datetime
from regex import F
import torch

import utils.utils as utils
from env.constants import WORKSPACE_LIMITS
from env.environment_sim import Environment
from helpers.logger import Logger
from action_generator.grasp_detetor import Graspnet
from feature_extractor.feature_field_builder import FeatureField
from models.clip_agent import CLIPGrasp
from models.mb_agent import MBGrasp
from helpers.dataset import CLIPActionDataset


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', action='store', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=234, metavar='N',
                    help='random seed (default: 1234)')

    parser.add_argument('--evaluate', dest='evaluate', action='store_true', default=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
    parser.add_argument('--sample_grasp', dest='sample_grasp', action='store_true', default=False)
    parser.add_argument('--log_suffix', action='store', type=str, default=None)

    parser.add_argument('--feat_backbone', action='store', type=str, default='clip')
    parser.add_argument('--agent', action='store', type=str, default='mb')
    parser.add_argument('--sample_num', action='store', type=int, default=500)
    parser.add_argument('--num_obj', action='store', type=int, default=15)
    parser.add_argument('--num_episode', action='store', type=int, default=5000)
    parser.add_argument('--max_episode_step', type=int, default=8)


    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    
    # set device and seed
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # parameters
    num_obj = args.num_obj
    num_episode = args.num_episode

    # load environment
    env = Environment(gui=False)
    env.seed(args.seed)
    # env_sim = Environment(gui=False)
    # load logger
    logger = Logger(suffix=args.log_suffix)
    # load graspnet
    graspnet = Graspnet()
    # load feature field builer
    field_builer = FeatureField(num_cam=3, query_threshold=[0.4], grid_size=0.004, boundaries=WORKSPACE_LIMITS, feat_backbone=args.feat_backbone, device=args.device)
    
    if args.agent == "clip":
        # load clip agent
        agent = CLIPGrasp(args)
    elif args.agent == "mb":
        agent = MBGrasp(args)

    extrinsics, intrinsics = utils.get_camera_configs(env)
    logger.save_camera_configs(extrinsics, intrinsics)

    # data initialization
    data = CLIPActionDataset()

    iteration = 0
    updates = 0
    
    # collect data with clip agent
    for episode in range(num_episode):
        episode_reward = 0
        episode_steps = 0
        done = False
        reset = False

        while not reset:
            env.reset()
            # env_sim.reset()
            lang_goal = env.generate_lang_goal()
            if episode < 400:
                warmup_num_obj = 8
                reset = env.add_objects(warmup_num_obj, WORKSPACE_LIMITS)
            else:
                reset = env.add_objects(num_obj, WORKSPACE_LIMITS)
            # reset &= env_sim.add_objects(num_obj, WORKSPACE_LIMITS)
            print(f"\033[032m Reset environment of episode {episode}, language goal {lang_goal}\033[0m")

        while not done:
            # check if one of the target objects is in the workspace:
            out_of_workspace = []
            for obj_id in env.target_obj_ids:
                pos, _, _ = env.obj_info(obj_id)
                if pos[0] < WORKSPACE_LIMITS[0][0] or pos[0] > WORKSPACE_LIMITS[0][1] \
                    or pos[1] < WORKSPACE_LIMITS[1][0] or pos[1] > WORKSPACE_LIMITS[1][1]:
                    out_of_workspace.append(obj_id)

            if len(out_of_workspace) == len(env.target_obj_ids):
                print("\033[031m Target objects are not in the scene!\033[0m")
                break     
            
            color_images, depth_images, pcd = utils.get_multi_view_images_w_pointcloud(env, visualize=args.visualize)
            
            # graspnet
            # Note: simply replace object poses to object bounding boxes maybe ok
            with torch.no_grad():
                grasp_pose_set, grasp_pose_dict, _ = graspnet.grasp_detection(pcd, env.get_true_object_poses(), visualize=args.visualize)  # 1.19s
            print("Number of grasping poses", len(grasp_pose_set))
            if len(grasp_pose_set) == 0:
                break

            pts = utils.generate_points_for_feature_extraction(pcd, visualize=args.visualize)

            # generated feature
            # for vision-language tasks
            feature_list = ['clip_feats', 'clip_sims']
            
            pts, feat_dict = field_builer.generate_feature_field(color_images, depth_images, extrinsics, intrinsics, lang_goal, feature_list, pts, last_text_feature=False, visualize=args.visualize)  # 2s

            # preprocess
            sampled_pts, sampled_clip_feats, sampled_clip_sims, grasps, grasp_pose_set = utils.preprocess_pp_unified(pts, feat_dict, grasp_pose_set, sample_action=args.sample_grasp, sample_num=args.sample_num, visualize=args.visualize)

            if args.agent == "clip":
                if len(grasp_pose_set) == 1:
                    action_idx = 0
                else:
                    with torch.no_grad():
                        action_idx = agent.select_action_greedy(sampled_pts, sampled_clip_sims, grasps)

            elif args.agent == "mb":
                action_idx = agent.select_action_greedy(grasp_pose_set, env.get_target_object_poses())

            action = grasp_pose_set[action_idx]

            reward, done = env.step(action)
            episode_steps += 1
            iteration += 1
            episode_reward += reward
            print("\033[034m Episode: {}, total numsteps: {}, reward: {}\033[0m".format(episode, iteration, round(reward, 2), done))

            if done:
                # data collection
                data.add(episode, episode_steps, lang_goal, sampled_pts.detach().cpu().numpy()[0], sampled_clip_feats.detach().cpu().numpy()[0], sampled_clip_sims.detach().cpu().numpy()[0], grasps.detach().cpu().numpy()[0], action_idx, reward, done)

            # record
            logger.reward_logs.append(reward)
            logger.executed_action_logs.append(action)
            logger.write_to_log('reward', logger.reward_logs)
            logger.write_to_log('executed_action', logger.executed_action_logs)
            
            if done or episode_steps == args.max_episode_step:
                break

        if (episode + 1) % 1000 == 0:
            timestamp = time.time()
            timestamp_value = datetime.datetime.fromtimestamp(timestamp)
            name = 'train_' + timestamp_value.strftime('%Y_%m_%d_%H_%M_%S_') + str(len(data.data['sequence'])) + '.npy'
            data.save(name)
            
        logger.episode_reward_logs.append(episode_reward)
        logger.episode_step_logs.append(episode_steps)
        logger.episode_success_logs.append(done)
        logger.write_to_log('episode_reward', logger.episode_reward_logs)
        logger.write_to_log('episode_step', logger.episode_step_logs)
        logger.write_to_log('episode_success', logger.episode_success_logs)
        print("\033[034m Episode: {}, total numsteps: {}, episode steps: {}, episode reward: {}, success: {}\033[0m".format(episode, iteration, episode_steps, round(episode_reward, 2), done))