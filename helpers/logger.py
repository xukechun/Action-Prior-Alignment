import os
import time
import datetime
import cv2
import open3d as o3d
import torch
import numpy as np

class Logger:
    def __init__(self, case_dir=None, case=None, suffix=None, resume_logger=None):
        
        if resume_logger is None:
            # Create directory to save data
            timestamp = time.time()
            timestamp_value = datetime.datetime.fromtimestamp(timestamp)
            if case is not None:
                name = case.split("/")[-1].split(".")[0] + "-"
                name = name[:-1]
            elif case_dir is not None:
                name = "test"
            else:
                name = "train"
            if suffix is not None:
                self.base_directory = os.path.join(
                    os.path.abspath("logs"), timestamp_value.strftime("%Y-%m-%d-%H-%M-%S") + "-" + name + "-" + suffix)
            else:
                self.base_directory = os.path.join(
                    os.path.abspath("logs"), timestamp_value.strftime("%Y-%m-%d-%H-%M-%S") + "-" + name)
                
            print("Creating data logging session: %s" % (self.base_directory))
            
        else:
            # resume from given log directory
            self.base_directory = resume_logger
            print("Resuming data logging session: %s" % (self.base_directory))
        
        
        self.color_images_directory = os.path.join(
            self.base_directory, "data", "color-images"
        )
        self.depth_images_directory = os.path.join(
            self.base_directory, "data", "depth-images"
        )
        self.mesh_directory = os.path.join(
            self.base_directory, "data", "meshes"
        )

        self.camera_config_directory = os.path.join(self.base_directory, "data", "cameras")
        self.visualizations_directory = os.path.join(self.base_directory, "visualizations")
        self.transitions_directory = os.path.join(self.base_directory, "transitions")
        self.checkpoints_directory = os.path.join(self.base_directory, "checkpoints")

        self.reward_logs = []
        self.episode_reward_logs = []
        self.episode_step_logs = []
        self.episode_success_logs = []
        self.executed_action_logs = []

        if not os.path.exists(self.color_images_directory):
            os.makedirs(self.color_images_directory)
        if not os.path.exists(self.depth_images_directory):
            os.makedirs(self.depth_images_directory)
        if not os.path.exists(self.mesh_directory):
            os.makedirs(self.mesh_directory)
        if not os.path.exists(self.camera_config_directory):
            os.makedirs(self.camera_config_directory)
        if not os.path.exists(self.visualizations_directory):
            os.makedirs(self.visualizations_directory)
        if not os.path.exists(self.transitions_directory):
            os.makedirs(self.transitions_directory)
        if not os.path.exists(self.checkpoints_directory):
            os.makedirs(self.checkpoints_directory)

        if case is not None or case_dir is not None:
            self.result_directory = os.path.join(self.base_directory, "results")
            if not os.path.exists(self.result_directory):
                os.makedirs(self.result_directory)

    def create_dir(self, dir_name):
        dir = os.path.join(self.base_directory, dir_name)
        if not os.path.exists(dir):
            os.makedirs(dir)    
        return dir

    def create_log(self, log_name):
        setattr(self, log_name, [])

    def save_multi_view_images(self, iteration, color_images, depth_images):
        # each pair of color and depth images is generated from one camera view
        for i in range(len(color_images)):
            color = color_images[i]
            depth = depth_images[i]
            color_image = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(self.color_images_directory, "color_%06d_%d.png" % (iteration, i)), color_image)
            
            # for visualization
            # depth_image = np.round(depth * 20000).astype(np.uint16)  # Save depth in 1e-3 meters (i.e. mm)
            depth_image = np.round(depth * 1000).astype(np.uint16)  # Save depth in 1e-3 meters (i.e. mm)
            cv2.imwrite(os.path.join(self.depth_images_directory, "depth_%06d_%d.png" % (iteration, i)), depth_image)
        
    def save_visualizations(self, iteration, geometries, suffix="action"):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        for geometry in geometries:
            vis.add_geometry(geometry)
            vis.update_geometry(geometry)
        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image(os.path.join(self.visualizations_directory, "vis_%06d_%s.png" % (iteration, suffix)), do_render=True)
        vis.destroy_window()

    def save_camera_configs(self, multi_camera_extrinsics, multi_camera_intrinsics):
        for i in range(len(multi_camera_extrinsics)):
            camera_extrinsics = multi_camera_extrinsics[i]
            camera_intrinsics = multi_camera_intrinsics[i]
            np.save(os.path.join(self.camera_config_directory, "camera_extrinsics_%d.npy" % i) , camera_extrinsics)
            np.save(os.path.join(self.camera_config_directory, "camera_intrinsics_%d.npy" % i) , camera_intrinsics)

    def save_meshes(self, iteration, meshes, mesh_names):
        for mesh, mesh_name in zip(meshes, mesh_names):
            mesh.export(os.path.join(self.mesh_directory, "%06d.%s.obj" % (iteration, mesh_name)))

    def write_to_log(self, log_name, log):
        np.savetxt(
            os.path.join(self.transitions_directory, "%s.log.txt" % log_name), log, delimiter=" "
        )
    
    def save_net_arch(self, net):
        with open(os.path.join(self.base_directory, "net.txt"), "w") as net_arch_file:
            net_arch_file.write(str(net))   
            
    def save_configs(self, configs):
        with open(os.path.join(self.base_directory, "config.txt"), "w") as config_file:
            config_file.write(str(configs))
      
    def save_adaptive_checkpoint(self, model, datatime, suffix="", ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = "sac_checkpoint_{}_{}.pth".format(datatime, suffix)
            ckpt_path = os.path.join(self.checkpoints_directory, ckpt_path)
        print('Saving models to {}'.format(ckpt_path))
        # torch.save(model.state_dict(), ckpt_path)
        torch.save({'feature_state_dict': model.vilg_fusion.state_dict(),
                    'policy_state_dict': model.policy.state_dict(),
                    'residual_policy_state_dict': model.residual_policy.state_dict(),
                    'critic_state_dict': model.critic.state_dict(),
                    'critic_target_state_dict': model.critic_target.state_dict(),
                    'critic_optimizer_state_dict': model.critic_optim.state_dict(),
                    'policy_optimizer_state_dict': model.policy_optim.state_dict()}, ckpt_path)

    # Save model parameters
    def save_sl_checkpoint(self, model, datatime, suffix="", ckpt_path=None):
        if ckpt_path is None:
            ckpt_path = "sl_checkpoint_{}_{}.pth".format(datatime, suffix)
            ckpt_path = os.path.join(self.checkpoints_directory, ckpt_path)
        print('Saving models to {}'.format(ckpt_path))
        torch.save(model.state_dict(), ckpt_path)

    # Load model parameters
    def load_sl_checkpoint(self, model, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint)
            model.eval() if evaluate else model.train()
            
    def load_base_sl_checkpoint(self, model, ckpt_path, evaluate=False):
        print('Loading base models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint_dict = torch.load(ckpt_path)
            model_dict = model.state_dict()
            model_dict.update(checkpoint_dict)
            model.load_state_dict(model_dict)
            model.eval() if evaluate else model.train()
