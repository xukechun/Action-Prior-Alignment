import time
import datetime
import glob
import pybullet as pb
import pybullet_data
from pybullet_utils import bullet_client
import numpy as np
import trimesh
import os
from operator import itemgetter
from scipy.spatial.transform import Rotation as R
import utils.utils as utils
import env.cameras as cameras
from env.constants import PIXEL_SIZE, WORKSPACE_LIMITS, LANG_TEMPLATES, LABEL, GENERAL_LABEL, COLOR_SHAPE, FUNCTION, LABEL_DIR_MAP, KEYWORD_DIR_MAP, ALL_LABEL, ALL_LABEL_DIR_MAP

class Environment:
    def __init__(self, gui=True, time_step=1 / 240):
        """Creates environment with PyBullet.

        Args:
        gui: show environment with PyBullet's built-in display viewer
        time_step: PyBullet physics simulation step speed. Default is 1 / 240.
        """

        self.time_step = time_step
        self.gui = gui
        self.pixel_size = PIXEL_SIZE
        self.case_dir = "cases/"
        self.obj_ids = {"fixed": [], "rigid": []}
        # initialize task relevent object lists
        self.target_obj_ids = []
        self.reference_obj_ids = []
        self.obj_labels = {}
        self.obj_dirs = {}
        self.test_file_name = None
        # self.agent_cams = cameras.RealSenseD435.CONFIG
        self.agent_cams = cameras.RealSenseL515.CONFIG
        self.oracle_cams = cameras.Oracle.CONFIG
        self.bounds = WORKSPACE_LIMITS
        self.home_joints = np.array([0, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi
        self.ik_rest_joints = np.array([0, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        # self.drop_joints0 = np.array([0.5, -0.8, 0.5, -0.2, -0.5, 0]) * np.pi
        self.drop_joints0 = np.array([-0.5, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi
        self.drop_joints1 = np.array([1, -0.5, 0.5, -0.5, -0.5, 0]) * np.pi

        # Start PyBullet.
        # pb = bullet_client.BulletClient(connection_mode=pb.GUI, "options=opengl2")
        # self._client_id = pb._client
        self._client_id = pb.connect(pb.GUI if gui else pb.DIRECT)
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.setTimeStep(time_step)

        if gui:
            target = pb.getDebugVisualizerCamera()[11]
            pb.resetDebugVisualizerCamera(
                cameraDistance=1.5, cameraYaw=90, cameraPitch=-25, cameraTargetPosition=target,
            )
            

    @property
    def is_static(self):
        """Return true if objects are no longer moving."""
        v = [
            np.linalg.norm(pb.getBaseVelocity(i, physicsClientId=self._client_id)[0])
            for i in self.obj_ids["rigid"]
        ]
        return all(np.array(v) < 5e-3)

    @property
    def info(self):
        """Environment info variable with object poses, dimensions, and colors."""

        info = {}  # object id : (position, rotation, dimensions)
        for obj_ids in self.obj_ids.values():
            for obj_id in obj_ids:
                pos, rot = pb.getBasePositionAndOrientation(
                    obj_id, physicsClientId=self._client_id
                )
                dim = pb.getVisualShapeData(obj_id, physicsClientId=self._client_id)[0][3]
                info[obj_id] = (pos, rot, dim)
        return info

    def obj_info(self, obj_id):
        """Environment info variable with object poses, dimensions, and colors."""

        pos, rot = pb.getBasePositionAndOrientation(
            obj_id, physicsClientId=self._client_id
        )
        dim = pb.getVisualShapeData(obj_id, physicsClientId=self._client_id)[0][3]
        info = (pos, rot, dim)
        return info    

    def generate_lang_goal(self):
        
        prob = np.array([0.4, 0.2, 0.1, 0.1, 0.2])
        template_id = np.random.choice(a=range(len(LANG_TEMPLATES)), size=1, p=prob)[0]

        if template_id == 0:
            id = np.random.choice(range(len(LABEL)), 1)[0]
            keyword = LABEL[id]
            self.target_obj_dir = [LABEL_DIR_MAP[id]]
            self.target_obj_lst = self.target_obj_dir
        else:
            if template_id == 1:
                id = np.random.choice(range(len(GENERAL_LABEL)), 1)[0]
                keyword = GENERAL_LABEL[id]
                self.target_obj_dir = KEYWORD_DIR_MAP[keyword]
            elif template_id == 2:
                id = np.random.choice(range(len(COLOR_SHAPE)), 1)[0]
                keyword = COLOR_SHAPE[id]
                self.target_obj_dir = KEYWORD_DIR_MAP[keyword]
            elif template_id == 3:
                id = np.random.choice(range(len(COLOR_SHAPE)), 1)[0]
                keyword = COLOR_SHAPE[id]
                self.target_obj_dir = KEYWORD_DIR_MAP[keyword]
            elif template_id == 4:
                id = np.random.choice(range(len(FUNCTION)), 1)[0]
                keyword = FUNCTION[id]
                self.target_obj_dir = KEYWORD_DIR_MAP[keyword]
            
            if len(self.target_obj_dir) > 3:
                batch = np.random.choice(range(len(self.target_obj_dir)), 2, replace=False) 
                self.target_obj_lst = list(itemgetter(*batch)(self.target_obj_dir))
            else:
                self.target_obj_lst = self.target_obj_dir

        self.lang_goal = LANG_TEMPLATES[template_id].format(keyword=keyword)
                    
        pb.addUserDebugText(text=self.lang_goal, textPosition=[0.8, -0.2, 0], textColorRGB=[0, 0, 1], textSize=2)
        
        return self.lang_goal

    def generate_place_lang_goal(self, obj_bbox_ids, obj_bbox_centers, obj_bbox_sizes, pp_file=False):
        self.lang_goal, ref_obj_ids, ref_obj_centers, ref_regions, valid_mask = utils.generate_place_inst(self.obj_labels, self.obj_dirs, obj_bbox_ids, obj_bbox_centers, obj_bbox_sizes, self.pixel_size)
        self.reference_obj_ids = ref_obj_ids
        
        if self.lang_goal is not None:
            obj_name = self.obj_labels[ref_obj_ids[0]][0]
            dir_phrase = ref_regions[0]

            if not pp_file:
                # insert at the begining
                with open(self.case_dir + self.test_file_name, "r") as file:
                    current_content = file.read()

                new_content = self.lang_goal + "\n" + obj_name + "\n" + dir_phrase + "\n" + current_content

                with open(self.case_dir + self.test_file_name, "w") as file:
                    file.write(new_content)
                
            else:
                # insert after grasp
                with open(self.case_dir + self.test_file_name, "r") as file:
                    current_content = file.readlines()

                insert_line_number = 17
                current_content.insert(insert_line_number, self.lang_goal + "\n" + obj_name + "\n" + dir_phrase + "\n")
                with open(self.case_dir + self.test_file_name, "w") as file:
                    file.writelines(current_content)
        
        return self.lang_goal, ref_obj_ids, ref_obj_centers, ref_regions, valid_mask
  
    def get_target_id(self):
        return self.target_obj_ids
    
    def get_reference_id(self):
        return self.reference_obj_ids

    def add_object_id(self, obj_id, category="rigid"):
        """List of (fixed, rigid) objects in env."""
        self.obj_ids[category].append(obj_id)

    def remove_object_id(self, obj_id, category="rigid"):
        """List of (fixed, rigid) objects in env."""
        self.obj_ids[category].remove(obj_id)
        
        if obj_id in self.target_obj_ids:
            self.target_obj_ids.remove(obj_id)
        if obj_id in self.reference_obj_ids:
            self.reference_obj_ids.remove(obj_id)
        if obj_id in self.obj_labels.keys():
            del self.obj_labels[obj_id]
        
        pb.removeBody(obj_id)

    def save_objects(self):
        """Save states of all rigid objects. If this is unstable, could use saveBullet."""
        success = False
        while not success:
            success = self.wait_static()
        object_states = []
        for obj in self.obj_ids["rigid"]:
            pos, orn = pb.getBasePositionAndOrientation(obj)
            linVel, angVel = pb.getBaseVelocity(obj)
            object_states.append((pos, orn, linVel, angVel))
        return object_states

    def restore_objects(self, object_states):
        """Restore states of all rigid objects. If this is unstable, could use restoreState along with saveBullet."""
        for idx, obj in enumerate(self.obj_ids["rigid"]):
            pos, orn, linVel, angVel = object_states[idx]
            pb.resetBasePositionAndOrientation(obj, pos, orn)
            pb.resetBaseVelocity(obj, linVel, angVel)
        success = self.wait_static()
        return success

    def wait_static(self, timeout=3):
        """Step simulator asynchronously until objects settle."""
        pb.stepSimulation()
        t0 = time.time()
        while (time.time() - t0) < timeout:
            if self.is_static:
                return True
            pb.stepSimulation()
        print(f"Warning: Wait static exceeded {timeout} second timeout. Skipping.")
        return False

    def reset(self, workspace="raw"):
        self.obj_ids = {"fixed": [], "rigid": []}
        self.target_obj_ids = []
        self.reference_obj_ids = []
        self.obj_labels = {}
        self.obj_dirs = {}
        
        pb.resetSimulation()
        pb.setGravity(0, 0, -9.8)

        # Temporarily disable rendering to load scene faster.
        if self.gui:
            pb.configureDebugVisualizer(pb.COV_ENABLE_RENDERING, 0)

        # Load workspace
        self.plane = pb.loadURDF(
            "plane.urdf", basePosition=(0, 0, -0.0005), useFixedBase=True,
        )
        if workspace == "raw":
            self.workspace = pb.loadURDF(
                "assets/workspace/workspace.urdf", basePosition=(0.5, 0, 0), useFixedBase=True,
            )
        elif workspace == "extend":
            self.workspace = pb.loadURDF(
                "assets/workspace/pp_workspace.urdf", basePosition=(0.5, 0, 0), useFixedBase=True,
            )
                        
        pb.changeDynamics(
            self.plane,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )
        pb.changeDynamics(
            self.workspace,
            -1,
            lateralFriction=1.1,
            restitution=0.5,
            linearDamping=0.5,
            angularDamping=0.5,
        )

        # Load UR5e
        self.ur5e = pb.loadURDF(
            "assets/ur5e/ur5e.urdf", basePosition=(0, 0, 0), useFixedBase=True,
        )
        self.ur5e_joints = []
        for i in range(pb.getNumJoints(self.ur5e)):
            info = pb.getJointInfo(self.ur5e, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "ee_fixed_joint":
                self.ur5e_ee_id = joint_id
            if joint_type == pb.JOINT_REVOLUTE:
                self.ur5e_joints.append(joint_id)
        pb.enableJointForceTorqueSensor(self.ur5e, self.ur5e_ee_id, 1)

        self.setup_gripper()

        # Move robot to home joint configuration.
        success = self.go_home()

        # !!! Note that here we should get the eelink pose, NOT tcp pose !!!
        home_pos, home_rot = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        self.initial_ee_pose = np.hstack((np.array(home_pos), np.array(home_rot)))
        
        self.close_gripper()
        self.open_gripper()

        if not success:
            print("Simulation is wrong!")
            exit()

        # Re-enable rendering.
        if self.gui:
            pb.configureDebugVisualizer(
                pb.COV_ENABLE_RENDERING, 1, physicsClientId=self._client_id
            )

    def setup_gripper(self):
        """Load end-effector: gripper"""
        ee_position, _ = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        self.ee = pb.loadURDF(
            "assets/ur5e/gripper/robotiq_2f_85.urdf",
            ee_position,
            pb.getQuaternionFromEuler((0, -np.pi / 2, 0)),
        )
        self.ee_tip_z_offset = 0.1625
        self.gripper_angle_open = 0.03
        self.gripper_angle_close = 0.8
        self.gripper_angle_close_threshold = 0.73
        self.gripper_mimic_joints = {
            "left_inner_finger_joint": -1,
            "left_inner_knuckle_joint": -1,
            "right_outer_knuckle_joint": -1,
            "right_inner_finger_joint": -1,
            "right_inner_knuckle_joint": -1,
        }
        for i in range(pb.getNumJoints(self.ee)):
            info = pb.getJointInfo(self.ee, i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            if joint_name == "finger_joint":
                self.gripper_main_joint = joint_id
            elif joint_name == "dummy_center_fixed_joint":
                self.ee_tip_id = joint_id
            elif "finger_pad_joint" in joint_name:
                pb.changeDynamics(
                    self.ee, joint_id, lateralFriction=0.9
                )
                self.ee_finger_pad_id = joint_id
            elif joint_type == pb.JOINT_REVOLUTE:
                self.gripper_mimic_joints[joint_name] = joint_id
                # Keep the joints static
                pb.setJointMotorControl2(
                    self.ee, joint_id, pb.VELOCITY_CONTROL, targetVelocity=0, force=0,
                )
        self.ee_constraint = pb.createConstraint(
            parentBodyUniqueId=self.ur5e,
            parentLinkIndex=self.ur5e_ee_id,
            childBodyUniqueId=self.ee,
            childLinkIndex=-1,
            jointType=pb.JOINT_FIXED,
            jointAxis=(0, 0, 1),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, -0.02),
            childFrameOrientation=pb.getQuaternionFromEuler((0, -np.pi / 2, 0)),
            physicsClientId=self._client_id,
        )
        pb.changeConstraint(self.ee_constraint, maxForce=10000)
        pb.enableJointForceTorqueSensor(self.ee, self.gripper_main_joint, 1)

        # Set up mimic joints in robotiq gripper: left
        c = pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["left_inner_finger_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=1, erp=0.8, maxForce=10000)
        c = pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["left_inner_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=10000)
        # Set up mimic joints in robotiq gripper: right
        c = pb.createConstraint(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee,
            self.gripper_mimic_joints["right_inner_finger_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=1, erp=0.8, maxForce=10000)
        c = pb.createConstraint(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            self.ee,
            self.gripper_mimic_joints["right_inner_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[1, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=10000)
        # Set up mimic joints in robotiq gripper: connect left and right
        c = pb.createConstraint(
            self.ee,
            self.gripper_main_joint,
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            jointType=pb.JOINT_GEAR,
            jointAxis=[0, 1, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=self._client_id,
        )
        pb.changeConstraint(c, gearRatio=-1, erp=0.8, maxForce=1000)

    def step(self, pose=None):
        """Execute action with specified primitive.

        Args:
            action: action to execute.

        Returns:
            obs, done
        """
        done = False
        if pose is not None:
            success, grasped_obj_id, pos_dist = self.grasp(pose)
            # !!! Note that here we do not punish the failed grasp, just encourage to execute the nearest grasp to the targets
            if grasped_obj_id in self.target_obj_ids:
                reward = 1
                done = True
            else:
                max_pos_dist = np.sqrt((self.bounds[0][1]-self.bounds[0][0]) ** 2 + (self.bounds[1][1]-self.bounds[1][0]) ** 2)
                reward = - pos_dist / max_pos_dist

        # Step simulator asynchronously until objects settle.
        while not self.is_static:
            pb.stepSimulation()

        return reward, done

    def seed(self, seed=None):
        self._random = np.random.RandomState(seed)
        return seed

    def render_camera(self, config):
        """Render RGB-D image with specified camera configuration."""
        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = pb.getMatrixFromQuaternion(config["rotation"])
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = config["position"] + lookdir
        focal_len = config["intrinsics"][0, 0]
        znear, zfar = config["zrange"]
        viewm = pb.computeViewMatrix(config["position"], lookat, updir)
        fovh = (config["image_size"][0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = config["image_size"][1] / config["image_size"][0]
        projm = pb.computeProjectionMatrixFOV(fovh, aspect_ratio, znear, zfar)

        # Render with OpenGL camera settings.
        _, _, color, depth, segm = pb.getCameraImage(
            width=config["image_size"][1],
            height=config["image_size"][0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            shadow=0,
            flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pb.ER_BULLET_HARDWARE_OPENGL,
        )

        # Get color image.
        color_image_size = (config["image_size"][0], config["image_size"][1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel
        if config["noise"]:
            color = np.int32(color)
            color += np.int32(self._random.normal(0, 3, color.shape))
            color = np.uint8(np.clip(color, 0, 255))

        # Get depth image.
        depth_image_size = (config["image_size"][0], config["image_size"][1])
        zbuffer = np.array(depth).reshape(depth_image_size)
        depth = zfar + znear - (2.0 * zbuffer - 1.0) * (zfar - znear)
        depth = (2.0 * znear * zfar) / depth
        if config["noise"]:
            depth += self._random.normal(0, 0.003, depth_image_size)

        # Get segmentation image.
        segm = np.uint8(segm).reshape(depth_image_size)

        return color, depth, segm

    def __del__(self):
        pb.disconnect()

    def get_link_pose(self, body, link):
        result = pb.getLinkState(body, link)
        return result[4], result[5]

    def add_objects(self, num_obj, workspace_limits, test_data=False):
        """Randomly dropped objects to the workspace"""
        self.num_obj = num_obj
        mesh_list = glob.glob("assets/simplified_objects/*.urdf")

        # get target object
        target_mesh_list = []
        for target_obj in self.target_obj_lst:
            target_mesh_file = "assets/simplified_objects/" + target_obj + ".urdf"
            target_mesh_list.append(target_mesh_file)
        for obj in self.target_obj_dir:
            obj_mesh_file = "assets/simplified_objects/" + obj + ".urdf"
            mesh_list.remove(obj_mesh_file)

        obj_mesh_ind = np.random.randint(0, len(mesh_list), size=num_obj-len(self.target_obj_lst))
        # obj_mesh_color = color_space[np.asarray(range(num_obj)) % 10, :]

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        body_ids = []
        self.target_obj_ids = []

        if test_data:
            timestamp = time.time()
            timestamp_value = datetime.datetime.fromtimestamp(timestamp)
            self.test_file_name = timestamp_value.strftime("%Y-%m-%d-%H-%M-%S") + ".txt"
        else:
            self.test_file_name = "temp.txt"
        
        with open(self.case_dir + self.test_file_name, "w") as out_file:
            out_file.write("%s\n" % self.lang_goal)
            for i in range(len(target_mesh_list)):
                out_file.write(f"{i} ") 
            out_file.write("\n")
            
            # add target objects
            for target_mesh_file in target_mesh_list:
                drop_x = (
                    (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample()
                    + workspace_limits[0][0]
                    + 0.1
                )
                drop_y = (
                    (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample()
                    + workspace_limits[1][0]
                    + 0.1
                )
                object_position = [drop_x, drop_y, 0.2]
                object_orientation = [
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample(),
                ]
                body_id = pb.loadURDF(
                    target_mesh_file, object_position, pb.getQuaternionFromEuler(object_orientation)
                )
                # pb.changeVisualShape(body_id, -1, rgbaColor=object_color)
                body_ids.append(body_id)
                self.target_obj_ids.append(body_id)
                self.add_object_id(body_id)
                self.wait_static()

                out_file.write(
                    "%s %.18e %.18e %.18e %.18e %.18e %.18e\n"
                    % (
                        target_mesh_file,
                        object_position[0],
                        object_position[1],
                        object_position[2],
                        object_orientation[0],
                        object_orientation[1],
                        object_orientation[2],
                    )
                )

            # add other objects
            for object_idx in range(len(obj_mesh_ind)):
                curr_mesh_file = mesh_list[obj_mesh_ind[object_idx]]
                
                drop_x = (
                    (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample()
                    + workspace_limits[0][0]
                    + 0.1
                )
                drop_y = (
                    (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample()
                    + workspace_limits[1][0]
                    + 0.1
                )
                object_position = [drop_x, drop_y, 0.2]
                object_orientation = [
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample(),
                ]
                body_id = pb.loadURDF(
                    curr_mesh_file, object_position, pb.getQuaternionFromEuler(object_orientation)
                )
                body_ids.append(body_id)
                self.add_object_id(body_id)
                self.wait_static()

                out_file.write(
                    "%s %.18e %.18e %.18e %.18e %.18e %.18e\n"
                    % (
                        curr_mesh_file,
                        object_position[0],
                        object_position[1],
                        object_position[2],
                        object_orientation[0],
                        object_orientation[1],
                        object_orientation[2],
                    )
                )

        return body_ids, True

    def add_objects_for_place(self, num_obj, workspace_limits, test_data=False):
        """Randomly dropped objects to the workspace"""
        self.num_obj = num_obj
        mesh_list = glob.glob("assets/simplified_objects/*.urdf")

        obj_mesh_ind = np.random.randint(0, len(mesh_list), size=num_obj)

        # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
        body_ids = []
        self.obj_labels = {}
        self.obj_dirs = {}

        if test_data:
            timestamp = time.time()
            timestamp_value = datetime.datetime.fromtimestamp(timestamp)
            self.test_file_name = timestamp_value.strftime("%Y-%m-%d-%H-%M-%S") + ".txt"
        else:
            self.test_file_name = "temp.txt"
        
        # get drop positions
        drop_xys = utils.generate_drop_positions(num_obj, workspace_limits, min_dist=0.05) # 0.1 before
        if len(drop_xys) < num_obj:
            return body_ids, False
                
        with open(self.case_dir + self.test_file_name, "a") as out_file:

            # add objects
            for object_idx in range(len(obj_mesh_ind)):
                curr_mesh_file = mesh_list[obj_mesh_ind[object_idx]]
                
                curr_dir_label = curr_mesh_file.split('/')[-1].split('.')[0]
                
                drop_x, drop_y = drop_xys[object_idx]
                object_position = [drop_x, drop_y, 0.1]
                
                object_orientation = [
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample(),
                    2 * np.pi * np.random.random_sample(),
                ]
                body_id = pb.loadURDF(
                    curr_mesh_file, object_position, pb.getQuaternionFromEuler(object_orientation)
                )
                body_ids.append(body_id)
                self.add_object_id(body_id)
                self.wait_static()

                self.obj_dirs[body_id] = curr_dir_label
                self.obj_labels[body_id] = []

                # for objects that are have unique meaningful labels
                if curr_dir_label in ALL_LABEL_DIR_MAP:
                    curr_ind = ALL_LABEL_DIR_MAP.index(curr_dir_label)
                    curr_obj_label = ALL_LABEL[curr_ind] 
                    self.obj_labels[body_id].append(curr_obj_label)

                # for objects that have general meaningful labels
                for key in KEYWORD_DIR_MAP.keys():
                    if curr_dir_label in KEYWORD_DIR_MAP[key]:
                        self.obj_labels[body_id].append(key)
                        break
                
                if len(self.obj_labels[body_id]) == 0:
                    self.obj_labels.pop(body_id)
                
                out_file.write(
                    "%s %.18e %.18e %.18e %.18e %.18e %.18e\n"
                    % (
                        curr_mesh_file,
                        object_position[0],
                        object_position[1],
                        object_position[2],
                        object_orientation[0],
                        object_orientation[1],
                        object_orientation[2],
                    )
                )

        return body_ids, True

    def add_dummy_object_to_place(self, mesh_file, position):
        """Add the setting object into the workspace as the placement result"""
        body_ids = []
        self.obj_labels = {}
        self.obj_dirs = {}

        self.test_file_name = "temp.txt"
                
        with open(self.case_dir + self.test_file_name, "a") as out_file:
            dir_label = mesh_file.split('/')[-1].split('.')[0]
            
            object_position = [position[0], position[1], 0.05]
            object_orientation = [0., 0., 0.]
            
            body_id = pb.loadURDF(
                mesh_file, object_position, pb.getQuaternionFromEuler(object_orientation)
            )
            body_ids.append(body_id)
            self.add_object_id(body_id)
            self.wait_static()

            self.obj_dirs[body_id] = dir_label
            self.obj_labels[body_id] = []

            # for objects that are have unique meaningful labels
            if dir_label in ALL_LABEL_DIR_MAP:
                ind = ALL_LABEL_DIR_MAP.index(dir_label)
                obj_label = ALL_LABEL[ind] 
                self.obj_labels[body_id].append(obj_label)

            # for objects that have general meaningful labels
            for key in KEYWORD_DIR_MAP.keys():
                if dir_label in KEYWORD_DIR_MAP[key]:
                    self.obj_labels[body_id].append(key)
                    break
            
            if len(self.obj_labels[body_id]) == 0:
                self.obj_labels.pop(body_id)
            
            out_file.write(
                "%s %.18e %.18e %.18e %.18e %.18e %.18e\n"
                % (
                    mesh_file,
                    object_position[0],
                    object_position[1],
                    object_position[2],
                    object_orientation[0],
                    object_orientation[1],
                    object_orientation[2],
                )
            )

        return body_ids, True        

    def add_objects_w_target_pose(self, obj_mesh_file, workspace_limits):

        body_ids = []
    
        # add object with target pose
        with open(self.case_dir + "temp.txt", "w") as out_file:
            # add object with target pose
            drop_x = (
                (workspace_limits[0][1] - workspace_limits[0][0] - 0.2) * np.random.random_sample()
                + workspace_limits[0][0]
                + 0.1
            )
            drop_y = (
                (workspace_limits[1][1] - workspace_limits[1][0] - 0.2) * np.random.random_sample()
                + workspace_limits[1][0]
                + 0.1
            )
            object_position = [drop_x, drop_y, 0.2]
            object_orientation = [
                2 * np.pi * np.random.random_sample(),
                2 * np.pi * np.random.random_sample(),
                2 * np.pi * np.random.random_sample(),
            ]
            body_id = pb.loadURDF(
                obj_mesh_file, object_position, pb.getQuaternionFromEuler(object_orientation)
            )
            # pb.changeVisualShape(body_id, -1, rgbaColor=object_color)
            body_ids.append(body_id)
            self.add_object_id(body_id)
            self.wait_static()

            out_file.write(
                "%s %.18e %.18e %.18e %.18e %.18e %.18e\n"
                % (
                    obj_mesh_file,
                    object_position[0],
                    object_position[1],
                    object_position[2],
                    object_orientation[0],
                    object_orientation[1],
                    object_orientation[2],
                )
            )

        return body_ids, True

    def add_object_push_from_file(self, file_name, switch=None):
        success = True
        # Read data
        with open(file_name, "r") as preset_file:
            file_content = preset_file.readlines()
            self.lang_goal = file_content[0].split('\n')[0]
            target_obj = file_content[1].split()
            self.target_obj_ids = [4 + int(i) for i in target_obj]
            num_obj = len(file_content) - 2
            obj_files = []
            obj_positions = []
            obj_orientations = []
            for object_idx in range(num_obj):
                file_content_curr_object = file_content[object_idx+2].split()
                obj_file = file_content_curr_object[0]
                obj_files.append(obj_file)
                obj_positions.append(
                    [
                        float(file_content_curr_object[1]),
                        float(file_content_curr_object[2]),
                        float(file_content_curr_object[3]),
                    ]
                )
                obj_orientations.append(
                    [
                        float(file_content_curr_object[4]),
                        float(file_content_curr_object[5]),
                        float(file_content_curr_object[6]),
                    ]
                )

        # Import objects
        for object_idx in range(num_obj):
            curr_mesh_file = obj_files[object_idx]
            object_position = [
                obj_positions[object_idx][0],
                obj_positions[object_idx][1],
                obj_positions[object_idx][2],
            ]
            object_orientation = [
                obj_orientations[object_idx][0],
                obj_orientations[object_idx][1],
                obj_orientations[object_idx][2],
            ]
            body_id = pb.loadURDF(
                curr_mesh_file,
                object_position,
                pb.getQuaternionFromEuler(object_orientation),
                flags=pb.URDF_ENABLE_SLEEPING
            )
            self.add_object_id(body_id)
            success &= self.wait_static()
            success &= self.wait_static()

        # give time to stop
        for _ in range(5):
            pb.stepSimulation()

        return success, self.lang_goal

    def add_object_push_from_place_file(self, file_name, switch=None, variance=False):
        success = True
        # Read data
        with open(file_name, "r") as preset_file:
            file_content = preset_file.readlines()
            self.lang_goal = file_content[0].split('\n')[0]
            target_obj_label = file_content[1].split('\n')[0]
            target_dir = file_content[2].split('\n')[0]
            num_obj = len(file_content) - 3
            obj_files = []
            obj_positions = []
            obj_orientations = []
            for object_idx in range(num_obj):
                file_content_curr_object = file_content[object_idx+3].split()
                obj_file = file_content_curr_object[0]
                obj_files.append(obj_file)
                if not variance:
                    obj_positions.append(
                        [
                            float(file_content_curr_object[1]),
                            float(file_content_curr_object[2]),
                            float(file_content_curr_object[3]),
                        ]
                    )
                else:
                    obj_positions.append(
                        [
                            float(file_content_curr_object[1]) + 0.05 * np.random.uniform(-1, 1),
                            float(file_content_curr_object[2]) + 0.05 * np.random.uniform(-1, 1),
                            float(file_content_curr_object[3]),
                        ]
                    ) 
                obj_orientations.append(
                    [
                        float(file_content_curr_object[4]),
                        float(file_content_curr_object[5]),
                        float(file_content_curr_object[6]),
                    ]
                )

        # Import objects
        self.obj_labels = {}
        self.reference_obj_labels = []
        self.reference_obj_ids = []
        self.reference_obj_dirs = []
        for object_idx in range(num_obj):
            curr_mesh_file = obj_files[object_idx]
            object_position = [
                obj_positions[object_idx][0],
                obj_positions[object_idx][1],
                obj_positions[object_idx][2],
            ]
            object_orientation = [
                obj_orientations[object_idx][0],
                obj_orientations[object_idx][1],
                obj_orientations[object_idx][2],
            ]
            body_id = pb.loadURDF(
                curr_mesh_file,
                object_position,
                pb.getQuaternionFromEuler(object_orientation),
                flags=pb.URDF_ENABLE_SLEEPING
            )
            self.add_object_id(body_id)
            success &= self.wait_static()
            success &= self.wait_static()

            self.obj_labels[body_id] = []
            
            # check if the object is target
            curr_dir_label = curr_mesh_file.split('/')[-1].split('.')[0]
            # for objects that are have unique meaningful labels
            if curr_dir_label in ALL_LABEL_DIR_MAP:
                curr_ind = ALL_LABEL_DIR_MAP.index(curr_dir_label)
                curr_obj_label = ALL_LABEL[curr_ind] 
                self.obj_labels[body_id].append(curr_obj_label)

                if curr_obj_label == target_obj_label:
                    self.reference_obj_ids.append(body_id)
                    self.reference_obj_dirs.append(target_dir)
                    self.reference_obj_labels.append(target_obj_label)

            # for objects that have general meaningful labels
            for key in KEYWORD_DIR_MAP.keys():
                if curr_dir_label in KEYWORD_DIR_MAP[key]:
                    self.obj_labels[body_id].append(key)
                    if len(self.obj_labels[body_id]) == 1 and key == target_obj_label:
                        self.reference_obj_ids.append(body_id)
                        self.reference_obj_dirs.append(target_dir)
                        self.reference_obj_labels.append(target_obj_label)
                    break
                
            if len(self.obj_labels[body_id]) == 0:
                self.obj_labels.pop(body_id)
        
        # give time to stop
        for _ in range(5):
            pb.stepSimulation()

        return success, self.lang_goal

    def add_object_push_from_pickplace_file(self, file_name, mode="grasp"):
        success = True
        grasp_lang_goal = None
        place_lang_goal = None
        if mode == "grasp":
            # Read grasp data
            with open(file_name, "r") as preset_file:
                file_content = preset_file.readlines()
                grasp_lang_goal = file_content[0].split('\n')[0]
                target_obj = file_content[1].split()
                self.target_obj_ids = [4 + int(i) for i in target_obj]
                grasp_num_obj = 15
                obj_files = []
                obj_positions = []
                obj_orientations = []
                for object_idx in range(grasp_num_obj):
                    file_content_curr_object = file_content[object_idx + 2].split()
                    obj_file = file_content_curr_object[0]
                    obj_files.append(obj_file)
                    obj_positions.append(
                        [
                            float(file_content_curr_object[1]),
                            float(file_content_curr_object[2]),
                            float(file_content_curr_object[3]),
                        ]
                    )
                    obj_orientations.append(
                        [
                            float(file_content_curr_object[4]),
                            float(file_content_curr_object[5]),
                            float(file_content_curr_object[6]),
                        ]
                    )

            # Import objects
            for object_idx in range(grasp_num_obj):
                curr_mesh_file = obj_files[object_idx]
                object_position = [
                    obj_positions[object_idx][0],
                    obj_positions[object_idx][1],
                    obj_positions[object_idx][2],
                ]
                object_orientation = [
                    obj_orientations[object_idx][0],
                    obj_orientations[object_idx][1],
                    obj_orientations[object_idx][2],
                ]
                body_id = pb.loadURDF(
                    curr_mesh_file,
                    object_position,
                    pb.getQuaternionFromEuler(object_orientation),
                    flags=pb.URDF_ENABLE_SLEEPING
                )
                self.add_object_id(body_id)
                success &= self.wait_static()
                success &= self.wait_static()

            # give time to stop
            for _ in range(5):
                pb.stepSimulation()

        elif mode == "place":
            success = True
            # Read data
            with open(file_name, "r") as preset_file:
                file_content = preset_file.readlines()
                place_lang_goal = file_content[17].split('\n')[0]
                target_obj_label = file_content[18].split('\n')[0]
                target_dir = file_content[19].split('\n')[0]
                place_num_obj = 8
                obj_files = []
                obj_positions = []
                obj_orientations = []
                for object_idx in range(place_num_obj):
                    file_content_curr_object = file_content[object_idx + 20].split()
                    obj_file = file_content_curr_object[0]
                    obj_files.append(obj_file)
                    obj_positions.append(
                        [
                            float(file_content_curr_object[1]),
                            float(file_content_curr_object[2]),
                            float(file_content_curr_object[3]),
                        ]
                    )
                    obj_orientations.append(
                        [
                            float(file_content_curr_object[4]),
                            float(file_content_curr_object[5]),
                            float(file_content_curr_object[6]),
                        ]
                    )

            # Import objects
            self.obj_labels = {}
            self.reference_obj_labels = []
            self.reference_obj_ids = []
            self.reference_obj_dirs = []
            for object_idx in range(place_num_obj):
                curr_mesh_file = obj_files[object_idx]
                object_position = [
                    obj_positions[object_idx][0],
                    obj_positions[object_idx][1],
                    obj_positions[object_idx][2],
                ]
                object_orientation = [
                    obj_orientations[object_idx][0],
                    obj_orientations[object_idx][1],
                    obj_orientations[object_idx][2],
                ]
                body_id = pb.loadURDF(
                    curr_mesh_file,
                    object_position,
                    pb.getQuaternionFromEuler(object_orientation),
                    flags=pb.URDF_ENABLE_SLEEPING
                )
                self.add_object_id(body_id)
                success &= self.wait_static()
                success &= self.wait_static()

                self.obj_labels[body_id] = []
                
                # check if the object is target
                curr_dir_label = curr_mesh_file.split('/')[-1].split('.')[0]
                # for objects that are have unique meaningful labels
                if curr_dir_label in ALL_LABEL_DIR_MAP:
                    curr_ind = ALL_LABEL_DIR_MAP.index(curr_dir_label)
                    curr_obj_label = ALL_LABEL[curr_ind] 
                    self.obj_labels[body_id].append(curr_obj_label)

                    if curr_obj_label == target_obj_label:
                        self.reference_obj_ids.append(body_id)
                        self.reference_obj_dirs.append(target_dir)
                        self.reference_obj_labels.append(target_obj_label)

                # for objects that have general meaningful labels
                for key in KEYWORD_DIR_MAP.keys():
                    if curr_dir_label in KEYWORD_DIR_MAP[key]:
                        self.obj_labels[body_id].append(key)
                        if len(self.obj_labels[body_id]) == 1 and key == target_obj_label:
                            self.reference_obj_ids.append(body_id)
                            self.reference_obj_dirs.append(target_dir)
                            self.reference_obj_labels.append(target_obj_label)
                        break
                    
                if len(self.obj_labels[body_id]) == 0:
                    self.obj_labels.pop(body_id)
            
            # give time to stop
            for _ in range(5):
                pb.stepSimulation()

        return success, grasp_lang_goal, place_lang_goal

    def get_target_object_poses(self):
        transforms = dict()
        for obj_id in self.target_obj_ids:
            transform = self.get_true_object_pose(obj_id)
            transforms[obj_id] = transform
        return transforms    

    def get_reference_object_poses(self):
        transforms = dict()
        for obj_id in self.reference_obj_ids:
            transform = self.get_true_object_pose(obj_id)
            transforms[obj_id] = transform
        return transforms    
        
    def get_true_object_pose(self, obj_id):
        pos, ort = pb.getBasePositionAndOrientation(obj_id)
        position = np.array(pos).reshape(3, 1)
        rotation = pb.getMatrixFromQuaternion(ort)
        rotation = np.array(rotation).reshape(3, 3)
        transform = np.eye(4)
        transform[:3, :] = np.hstack((rotation, position))
        return transform   

    def get_true_object_poses(self):
        transforms = dict()
        for obj_id in self.obj_ids["rigid"]:
            transform = self.get_true_object_pose(obj_id)
            transforms[obj_id] = transform
        return transforms

    # ---------------------------------------------------------------------------
    # Robot Movement Functions
    # ---------------------------------------------------------------------------

    def go_home(self):
        return self.move_joints(self.home_joints)

    def move_joints(self, target_joints, speed=0.01, timeout=3):
        """Move UR5e to target joint configuration."""
        t0 = time.time()
        while (time.time() - t0) < timeout:
            current_joints = np.array(
                [
                    pb.getJointState(self.ur5e, i, physicsClientId=self._client_id)[0]
                    for i in self.ur5e_joints
                ]
            )
            pos, _ = self.get_link_pose(self.ee, self.ee_tip_id)
            
            if pos[2] < 0.005:
                print(f"Warning: move_joints tip height is {pos[2]}. Skipping.")
                return False
            diff_joints = target_joints - current_joints
            if all(np.abs(diff_joints) < 0.05):
                # give time to stop
                for _ in range(5):
                    pb.stepSimulation()
                return True

            # Move with constant velocity
            norm = np.linalg.norm(diff_joints)
            v = diff_joints / norm if norm > 0 else 0
            step_joints = current_joints + v * speed
            pb.setJointMotorControlArray(
                bodyIndex=self.ur5e,
                jointIndices=self.ur5e_joints,
                controlMode=pb.POSITION_CONTROL,
                targetPositions=step_joints,
                positionGains=np.ones(len(self.ur5e_joints)),
            )
            pb.stepSimulation()
        print(f"Warning: move_joints exceeded {timeout} second timeout. Skipping.")
        return False

    def move_ee_pose(self, pose, speed=0.01):
        """Move UR5e to target end effector pose."""
        target_joints = self.solve_ik(pose)
        return self.move_joints(target_joints, speed)

    def solve_ik(self, pose):
        """Calculate joint configuration with inverse kinematics."""
        joints = pb.calculateInverseKinematics(
            bodyUniqueId=self.ur5e,
            endEffectorLinkIndex=self.ur5e_ee_id,
            targetPosition=pose[0],
            targetOrientation=pose[1],
            lowerLimits=[-6.283, -6.283, -3.141, -6.283, -6.283, -6.283],
            upperLimits=[6.283, 6.283, 3.141, 6.283, 6.283, 6.283],
            jointRanges=[12.566, 12.566, 6.282, 12.566, 12.566, 12.566],
            restPoses=np.float32(self.ik_rest_joints).tolist(),
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        joints = np.array(joints, dtype=np.float32)
        # joints[2:] = (joints[2:] + np.pi) % (2 * np.pi) - np.pi
        return joints

    def straight_move(self, pose0, pose1, rot, speed=0.01, max_force=300, detect_force=False, is_push=False):
        """Move every 1 cm, keep the move in a straight line instead of a curve. Keep level with rot"""
        step_distance = 0.01  # every 1 cm
        vec = np.float32(pose1) - np.float32(pose0)
        length = np.linalg.norm(vec)
        vec = vec / length
        n_push = np.int32(np.floor(length / step_distance))  # every 1 cm
        success = True
        for n in range(n_push):
            target = pose0 + vec * n * step_distance
            success &= self.move_ee_pose((target, rot), speed)
            if detect_force:
                force = np.sum(
                    np.abs(np.array(pb.getJointState(self.ur5e, self.ur5e_ee_id)[2]))
                )
                if force > max_force:
                    target = target - vec * 2 * step_distance
                    self.move_ee_pose((target, rot), speed)
                    print(f"Force is {force}, exceed the max force {max_force}")
                    return False    
        if is_push:
            speed /= 5
        success &= self.move_ee_pose((pose1, rot), speed)
        return success

    def push(self, pose0, pose1, speed=0.002, verbose=True):
        """Execute pushing primitive.

        Args:
            pose0: SE(3) starting pose.
            pose1: SE(3) ending pose.
            speed: the speed of the planar push.

        Returns:
            success: robot movement success if True.
        """

        # close the gripper
        self.close_gripper(is_slow=False)

        # Adjust push start and end positions.
        pos0 = np.array(pose0, dtype=np.float32)
        pos1 = np.array(pose1, dtype=np.float32)
        pos0[2] += self.ee_tip_z_offset
        pos1[2] += self.ee_tip_z_offset
        vec = pos1 - pos0
        length = np.linalg.norm(vec)
        vec = vec / length
        over0 = np.array((pos0[0], pos0[1], pos0[2] + 0.05))
        over0h = np.array((pos0[0], pos0[1], pos0[2] + 0.2))
        over1 = np.array((pos1[0], pos1[1], pos1[2] + 0.05))
        over1h = np.array((pos1[0], pos1[1], pos1[2] + 0.2))

        # Align against push direction.
        theta = np.arctan2(vec[1], vec[0]) + np.pi / 2
        rot = pb.getQuaternionFromEuler([np.pi / 2, np.pi / 2, theta])

        # Execute push.
        success = self.move_joints(self.ik_rest_joints)
        if success:
            success = self.move_ee_pose((over0h, rot))
        if success:
            success = self.straight_move(over0h, over0, rot, detect_force=True)
        if success:
            success = self.straight_move(over0, pos0, rot, detect_force=True)
        if success:
            success = self.straight_move(pos0, pos1, rot, speed, detect_force=True, is_push=True)
        if success:
            success = self.straight_move(pos1, over1, rot, speed)
        if success:
            success = self.straight_move(over1, over1h, rot)
        self.go_home()

        if verbose:
            print(f"Push from {pose0} to {pose1}, {success}")

        return success


    def grasp(self, pose, speed=0.005, follow_place=False):
        """Execute grasping primitive.

        Args:
            pose: SE(3) grasping pose.

        Returns:
            success: robot movement success if True.
        """

        # Handle unexpected behavior
        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9, spinningFriction=0.1
        )

        pose = np.array(pose, dtype=np.float32)
        rot = pose[-4:]
        pos = pose[:3]
        transform = np.eye(4)
        transform[:3, :3] = R.from_quat(rot).as_matrix()
        transform[:3, 3] = pos

        # ee link in tip
        ee_tip_transform = np.array([[0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [1, 0, 0, -self.ee_tip_z_offset],
                                    [0, 0, 0, 1]])
        
        # visualization for debug
        # ee_pos, ee_rot = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        # tip_pos, tip_rot = self.get_link_pose(self.ee, self.ee_tip_id)
        # ee_axis = DebugAxes()
        # ee_axis.update(ee_pos, ee_rot)
        # tip_axis = DebugAxes()
        # tip_axis.update(tip_pos, tip_rot)

        # transform from tip to ee link
        ee_transform = transform @ ee_tip_transform

        pos = (ee_transform[:3, 3]).T
        pos[2] = max(pos[2] - 0.02, self.bounds[2][0])
        over = np.array((pos[0], pos[1], pos[2] + 0.2))
        rot = R.from_matrix(ee_transform[:3, :3]).as_quat()

        # Execute 6-dof grasping.
        grasped_obj_id = None
        min_pos_dist = None
        self.open_gripper()
        success = self.move_joints(self.ik_rest_joints)
        if success:
            success = self.move_ee_pose((over, rot))
        if success:
            success = self.straight_move(over, pos, rot, speed, detect_force=True)
        if success:
            self.close_gripper()
            success = self.straight_move(pos, over, rot, speed)
            success &= self.is_gripper_closed
            
            if success: # get grasp object id
                max_height = -0.0001
                for i in self.obj_ids["rigid"]:
                    height = self.info[i][0][2]
                    if height >= max_height:
                        grasped_obj_id = i
                        max_height = height

                # compute the object distance to the targets
                pos_dists = []
                for target_obj_id in self.target_obj_ids:
                    pos_dist = np.linalg.norm(np.array(self.info[grasped_obj_id][0][:2]) - np.array(self.info[target_obj_id][0][:2]))
                    pos_dists.append(pos_dist)
                min_pos_dist = min(pos_dists)
            
        if not success:
            # compute the grasp distance to the target
            pos_dists = []
            for target_obj_id in self.target_obj_ids:
                pos_dist = np.linalg.norm(np.array(pos[:2]) - np.array(self.info[target_obj_id][0][:2]))
                pos_dists.append(pos_dist)
            min_pos_dist = min(pos_dists)                

        if not follow_place:
            if success:
                success = self.move_joints(self.drop_joints1)
                # success &= self.is_gripper_closed
                self.open_gripper(is_slow=True)
            
            self.go_home()

        print(f"Grasp at {pose}, the grasp {success}")

        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9
        )

        return success, grasped_obj_id, min_pos_dist

    # place out of the workspace / trash
    def place_out_of_workspace(self):
        """Execute placing primitive.

        Returns:
            success: robot movement success if True.
        """
        success = self.move_joints(self.drop_joints0)
        success &= self.is_gripper_closed
        self.open_gripper(is_slow=True)
        self.go_home() 

        print(f"Move the object to the trash bin, the place {success}")
        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9
        )
        return success
    
    def place(self, pose, speed=0.005):
        """Execute placing primitive.

        Args:
            pose: SE(3) placing pose.

        Returns:
            success: robot movement success if True.
        """

        # Handle unexpected behavior
        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9, spinningFriction=0.1
        )

        pose = np.array(pose, dtype=np.float32)
        rot = pose[-4:]
        pos = pose[:3]
        transform = np.eye(4)
        transform[:3, :3] = R.from_quat(rot).as_matrix()
        transform[:3, 3] = pos

        # ee link in tip
        ee_tip_transform = np.array([[0, 0, -1, 0],
                                    [0, 1, 0, 0],
                                    [1, 0, 0, -self.ee_tip_z_offset],
                                    [0, 0, 0, 1]])
        
        # visualization for debug
        # ee_pos, ee_rot = self.get_link_pose(self.ur5e, self.ur5e_ee_id)
        # tip_pos, tip_rot = self.get_link_pose(self.ee, self.ee_tip_id)
        # ee_axis = DebugAxes()
        # ee_axis.update(ee_pos, ee_rot)
        # tip_axis = DebugAxes()
        # tip_axis.update(tip_pos, tip_rot)

        # transform from tip to ee link
        ee_transform = transform @ ee_tip_transform

        pos = (ee_transform[:3, 3]).T
        pos[2] = max(pos[2] - 0.02, self.bounds[2][0])
        over = np.array((pos[0], pos[1], pos[2] + 0.2))
        rot = R.from_matrix(ee_transform[:3, :3]).as_quat()

        # Execute 6-dof placing.
        self.close_gripper()
        success = self.move_ee_pose((over, rot))

        if success:
            success = self.straight_move(over, pos, rot, speed/4, detect_force=True)
            self.open_gripper(is_slow=True)
            self.go_home() 
            
        print(f"Place at {pose}, the place {success}")

        pb.changeDynamics(
            self.ee, self.ee_finger_pad_id, lateralFriction=0.9
        )

        return success


    def open_gripper(self, is_slow=False):
        self._move_gripper(self.gripper_angle_open, is_slow=is_slow)

    def close_gripper(self, is_slow=True):
        self._move_gripper(self.gripper_angle_close, is_slow=is_slow)

    @property
    def is_gripper_closed(self):
        gripper_angle = pb.getJointState(
            self.ee, self.gripper_main_joint, physicsClientId=self._client_id
        )[0]
        return gripper_angle < self.gripper_angle_close_threshold

    def _move_gripper(self, target_angle, timeout=3, is_slow=False):
        t0 = time.time()
        prev_angle = pb.getJointState(
            self.ee, self.gripper_main_joint, physicsClientId=self._client_id
        )[0]

        if is_slow:
            pb.setJointMotorControl2(
                self.ee,
                self.gripper_main_joint,
                pb.VELOCITY_CONTROL,
                targetVelocity=1 if target_angle > 0.5 else -1,
                maxVelocity=1 if target_angle > 0.5 else -1,
                force=3,
                physicsClientId=self._client_id,
            )
            pb.setJointMotorControl2(
                self.ee,
                self.gripper_mimic_joints["right_outer_knuckle_joint"],
                pb.VELOCITY_CONTROL,
                targetVelocity=1 if target_angle > 0.5 else -1,
                maxVelocity=1 if target_angle > 0.5 else -1,
                force=3,
                physicsClientId=self._client_id,
            )
            for _ in range(10):
                pb.stepSimulation()
            while (time.time() - t0) < timeout:
                current_angle = pb.getJointState(self.ee, self.gripper_main_joint)[0]
                diff_angle = abs(current_angle - prev_angle)
                if diff_angle < 1e-4:
                    break
                prev_angle = current_angle
                for _ in range(10):
                    pb.stepSimulation()
        # maintain the angles
        pb.setJointMotorControl2(
            self.ee,
            self.gripper_main_joint,
            pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=3.1,
        )
        pb.setJointMotorControl2(
            self.ee,
            self.gripper_mimic_joints["right_outer_knuckle_joint"],
            pb.POSITION_CONTROL,
            targetPosition=target_angle,
            force=3.1,
        )
        for _ in range(10):
            pb.stepSimulation()

class DebugAxes(object):
    """
    Visualization of local frame: red for x axis, green for y axis, blue for z axis
    """
    def __init__(self):
        self.uids = [-1, -1, -1]

    def update(self, pos, orn):
        """
        Args:
        - pos: len=3, position in world frame
        - orn: len=4, quaternion (x, y, z, w), world frame
        """
        pos = np.asarray(pos)
        rot3x3 = R.from_quat(orn).as_matrix()
        axis_x, axis_y, axis_z = rot3x3.T
        self.uids[0] = pb.addUserDebugLine(pos, pos + axis_x * 0.05, [1, 0, 0], replaceItemUniqueId=self.uids[0])
        self.uids[1] = pb.addUserDebugLine(pos, pos + axis_y * 0.05, [0, 1, 0], replaceItemUniqueId=self.uids[1])
        self.uids[2] = pb.addUserDebugLine(pos, pos + axis_z * 0.05, [0, 0, 1], replaceItemUniqueId=self.uids[2])

if __name__ == "__main__":
    env = Environment()
    env.reset()

    print(pb.getPhysicsEngineParameters(env._client_id))

    time.sleep(1)
    # env.add_object_push_from_file("hard-cases/temp.txt", switch=None)

    # push_start = [4.280000000000000471e-01, -3.400000000000000244e-02, 0.01]
    # push_end = [5.020000000000000018e-01, -3.400000000000000244e-02, 0.01]
    # env.push(push_start, push_end)
    # time.sleep(1)

    env.render_camera(env.oracle_cams[0])

    for i in range(16):
        best_rotation_angle = np.deg2rad(90 - i * (360.0 / 16))
        primitive_position = [0.6, 0, 0.01]
        primitive_position_end = [
            primitive_position[0] + 0.1 * np.cos(best_rotation_angle),
            primitive_position[1] + 0.1 * np.sin(best_rotation_angle),
            0.01,
        ]
        env.push(primitive_position, primitive_position_end, speed=0.0002)
        env._pb.addUserDebugLine(primitive_position, primitive_position_end, lifeTime=0)

        # angle = np.deg2rad(i * 360 / 16)
        # pos = [0.5, 0, 0.05]
        # env.grasp(pos, angle)

        time.sleep(1)
