import os
import sys
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
import pickle
from matplotlib import cm
import matplotlib.pyplot as plt
import cv2
import mcubes
import trimesh
from torch import nn
from torchvision import transforms
import torch.nn.modules.utils as nn_utils
import math
import timm
import types
from pathlib import Path
from typing import Union, List, Tuple
from PIL import Image
from transformers import AutoModel, CLIPImageProcessor
import open3d as o3d
from einops import rearrange

from .clip import clip, tokenize
from utils.draw_utils import draw_keypoints, aggr_point_cloud_from_data
from utils.corr_utils import compute_similarity_tensor, compute_similarity_tensor_multi
from utils.my_utils import  depth2fgpcd, fps_np, find_indices, depth2normal
from utils.utils import get_pca_map, resize
# torch.cuda.set_device(0)

class ViTExtractor:
    """ This class facilitates extraction of features, descriptors, and saliency maps from a ViT.
    We use the following notation in the documentation of the module's methods:
    B - batch size
    h - number of heads. usually takes place of the channel dimension in pytorch's convention BxCxHxW
    p - patch size of the ViT. either 8 or 16.
    t - number of tokens. equals the number of patches + 1, e.g. HW / p**2 + 1. Where H and W are the height and width
    of the input image.
    d - the embedding dimension in the ViT.
    """

    def __init__(self, model_type: str = 'dino_vits8', stride: int = 4, model: nn.Module = None, device: str = 'cuda'):
        """
        :param model_type: A string specifying the type of model to extract from.
                          [dino_vits8 | dino_vits16 | dino_vitb8 | dino_vitb16 | vit_small_patch8_224 |
                          vit_small_patch16_224 | vit_base_patch8_224 | vit_base_patch16_224]
        :param stride: stride of first convolution layer. small stride -> higher resolution.
        :param model: Optional parameter. The nn.Module to extract from instead of creating a new one in ViTExtractor.
                      should be compatible with model_type.
        """
        self.model_type = model_type
        self.device = device
        if model is not None:
            self.model = model
        else:
            self.model = ViTExtractor.create_model(model_type)

        self.model = ViTExtractor.patch_vit_resolution(self.model, stride=stride)
        self.model.eval()
        self.model.to(self.device)
        self.p = self.model.patch_embed.patch_size
        if type(self.p)==tuple:
            self.p = self.p[0]
        self.stride = self.model.patch_embed.proj.stride

        self.mean = (0.485, 0.456, 0.406) if "dino" in self.model_type else (0.5, 0.5, 0.5)
        self.std = (0.229, 0.224, 0.225) if "dino" in self.model_type else (0.5, 0.5, 0.5)

        self._feats = []
        self.hook_handlers = []
        self.load_size = None
        self.num_patches = None

    @staticmethod
    def create_model(model_type: str) -> nn.Module:
        """
        :param model_type: a string specifying which model to load. [dino_vits8 | dino_vits16 | dino_vitb8 |
                           dino_vitb16 | vit_small_patch8_224 | vit_small_patch16_224 | vit_base_patch8_224 |
                           vit_base_patch16_224]
        :return: the model
        """
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        if 'v2' in model_type:
            model = torch.hub.load('facebookresearch/dinov2', model_type)
        elif 'dino' in model_type:
            model = torch.hub.load('facebookresearch/dino:main', model_type)
        else:  # model from timm -- load weights from timm to dino model (enables working on arbitrary size images).
            temp_model = timm.create_model(model_type, pretrained=True)
            model_type_dict = {
            'vit_small_patch16_224': 'dino_vits16',
            'vit_small_patch8_224': 'dino_vits8',
            'vit_base_patch16_224': 'dino_vitb16',
            'vit_base_patch8_224': 'dino_vitb8'
            }
            model = torch.hub.load('facebookresearch/dino:main', model_type_dict[model_type])
            temp_state_dict = temp_model.state_dict()
            del temp_state_dict['head.weight']
            del temp_state_dict['head.bias']
            model.load_state_dict(temp_state_dict)
        return model

    @staticmethod
    def _fix_pos_enc(patch_size: int, stride_hw: Tuple[int, int]):
        """
        Creates a method for position encoding interpolation.
        :param patch_size: patch size of the model.
        :param stride_hw: A tuple containing the new height and width stride respectively.
        :return: the interpolation method
        """
        def interpolate_pos_encoding(self, x: torch.Tensor, w: int, h: int) -> torch.Tensor:
            npatch = x.shape[1] - 1
            N = self.pos_embed.shape[1] - 1
            if npatch == N and w == h:
                return self.pos_embed
            class_pos_embed = self.pos_embed[:, 0]
            patch_pos_embed = self.pos_embed[:, 1:]
            dim = x.shape[-1]
            # compute number of tokens taking stride into account
            w0 = 1 + (w - patch_size) // stride_hw[1]
            h0 = 1 + (h - patch_size) // stride_hw[0]
            assert (w0 * h0 == npatch), f"""got wrong grid size for {h}x{w} with patch_size {patch_size} and 
                                            stride {stride_hw} got {h0}x{w0}={h0 * w0} expecting {npatch}"""
            # we add a small number to avoid floating point error in the interpolation
            # see discussion at https://github.com/facebookresearch/dino/issues/8
            w0, h0 = w0 + 0.1, h0 + 0.1
            patch_pos_embed = nn.functional.interpolate(
                patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
                mode='bicubic',
                align_corners=False, recompute_scale_factor=False
            )
            assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        return interpolate_pos_encoding

    @staticmethod
    def patch_vit_resolution(model: nn.Module, stride: int) -> nn.Module:
        """
        change resolution of model output by changing the stride of the patch extraction.
        :param model: the model to change resolution for.
        :param stride: the new stride parameter.
        :return: the adjusted model
        """
        patch_size = model.patch_embed.patch_size
        if type(patch_size) == tuple:
            patch_size = patch_size[0]
        if stride == patch_size:  # nothing to do
            return model

        stride = nn_utils._pair(stride)
        assert all([(patch_size // s_) * s_ == patch_size for s_ in
                    stride]), f'stride {stride} should divide patch_size {patch_size}'

        # fix the stride
        model.patch_embed.proj.stride = stride
        # fix the positional encoding code
        model.interpolate_pos_encoding = types.MethodType(ViTExtractor._fix_pos_enc(patch_size, stride), model)
        return model

    def preprocess(self, image_path: Union[str, Path],
                   load_size: Union[int, Tuple[int, int]] = None, patch_size: int = 14) -> Tuple[torch.Tensor, Image.Image]:
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        def divisible_by_num(num, dim):
            return num * (dim // num)
        pil_image = Image.open(image_path).convert('RGB')
        if load_size is not None:
            pil_image = transforms.Resize(load_size, interpolation=transforms.InterpolationMode.LANCZOS)(pil_image)

            width, height = pil_image.size
            new_width = divisible_by_num(patch_size, width)
            new_height = divisible_by_num(patch_size, height)
            pil_image = pil_image.resize((new_width, new_height), resample=Image.LANCZOS)
            
        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        prep_img = prep(pil_image)[None, ...]
        return prep_img, pil_image

    def preprocess_pil(self, pil_image):
        """
        Preprocesses an image before extraction.
        :param image_path: path to image to be extracted.
        :param load_size: optional. Size to resize image before the rest of preprocessing.
        :return: a tuple containing:
                    (1) the preprocessed image as a tensor to insert the model of shape BxCxHxW.
                    (2) the pil image in relevant dimensions
        """
        prep = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        prep_img = prep(pil_image)[None, ...]
        return prep_img

    def _get_hook(self, facet: str):
        """
        generate a hook method for a specific block and facet.
        """
        if facet in ['attn', 'token']:
            def _hook(model, input, output):
                self._feats.append(output)
            return _hook

        if facet == 'query':
            facet_idx = 0
        elif facet == 'key':
            facet_idx = 1
        elif facet == 'value':
            facet_idx = 2
        else:
            raise TypeError(f"{facet} is not a supported facet.")

        def _inner_hook(module, input, output):
            input = input[0]
            B, N, C = input.shape
            qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
            self._feats.append(qkv[facet_idx]) #Bxhxtxd
        return _inner_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet == 'token':
                    self.hook_handlers.append(block.register_forward_hook(self._get_hook(facet)))
                elif facet == 'attn':
                    self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_hook(facet)))
                elif facet in ['key', 'query', 'value']:
                    self.hook_handlers.append(block.attn.register_forward_hook(self._get_hook(facet)))
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(self, batch: torch.Tensor, layers: List[int] = 11, facet: str = 'key') -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                  if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                  if facet is 'attn' has shape Bxhxtxt
                  if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        self.load_size = (H, W)
        self.num_patches = (1 + (H - self.p) // self.stride[0], 1 + (W - self.p) // self.stride[1])
        return self._feats

    def _log_bin(self, x: torch.Tensor, hierarchy: int = 2) -> torch.Tensor:
        """
        create a log-binned descriptor.
        :param x: tensor of features. Has shape Bxhxtxd.
        :param hierarchy: how many bin hierarchies to use.
        """
        B = x.shape[0]
        num_bins = 1 + 8 * hierarchy

        bin_x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)  # Bx(t-1)x(dxh)
        bin_x = bin_x.permute(0, 2, 1)
        bin_x = bin_x.reshape(B, bin_x.shape[1], self.num_patches[0], self.num_patches[1])
        # Bx(dxh)xnum_patches[0]xnum_patches[1]
        sub_desc_dim = bin_x.shape[1]

        avg_pools = []
        # compute bins of all sizes for all spatial locations.
        for k in range(0, hierarchy):
            # avg pooling with kernel 3**kx3**k
            win_size = 3 ** k
            avg_pool = torch.nn.AvgPool2d(win_size, stride=1, padding=win_size // 2, count_include_pad=False)
            avg_pools.append(avg_pool(bin_x))

        bin_x = torch.zeros((B, sub_desc_dim * num_bins, self.num_patches[0], self.num_patches[1])).to(self.device)
        for y in range(self.num_patches[0]):
            for x in range(self.num_patches[1]):
                part_idx = 0
                # fill all bins for a spatial location (y, x)
                for k in range(0, hierarchy):
                    kernel_size = 3 ** k
                    for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                        for j in range(x - kernel_size, x + kernel_size + 1, kernel_size):
                            if i == y and j == x and k != 0:
                                continue
                            if 0 <= i < self.num_patches[0] and 0 <= j < self.num_patches[1]:
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                           :, :, i, j]
                            else:  # handle padding in a more delicate way than zero padding
                                temp_i = max(0, min(i, self.num_patches[0] - 1))
                                temp_j = max(0, min(j, self.num_patches[1] - 1))
                                bin_x[:, part_idx * sub_desc_dim: (part_idx + 1) * sub_desc_dim, y, x] = avg_pools[k][
                                                                                                           :, :, temp_i,
                                                                                                           temp_j]
                            part_idx += 1
        bin_x = bin_x.flatten(start_dim=-2, end_dim=-1).permute(0, 2, 1).unsqueeze(dim=1)
        # Bx1x(t-1)x(dxh)
        return bin_x

    def extract_descriptors(self, batch: torch.Tensor, layer: int = 11, facet: str = 'key',
                            bin: bool = False, include_cls: bool = False) -> torch.Tensor:
        """
        extract descriptors from the model
        :param batch: batch to extract descriptors for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token']
        :param bin: apply log binning to the descriptor. default is False.
        :return: tensor of descriptors. Bx1xtxd' where d' is the dimension of the descriptors.
        """
        assert facet in ['key', 'query', 'value', 'token'], f"""{facet} is not a supported facet for descriptors. 
                                                             choose from ['key' | 'query' | 'value' | 'token'] """
        self._extract_features(batch, [layer], facet)
        x = self._feats[0]
        if facet == 'token':
            x.unsqueeze_(dim=1) #Bx1xtxd
        if not include_cls:
            x = x[:, :, 1:, :]  # remove cls token
        else:
            assert not bin, "bin = True and include_cls = True are not supported together, set one of them False."
        if not bin:
            desc = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1).unsqueeze(dim=1)  # Bx1xtx(dxh)
        else:
            desc = self._log_bin(x)
        return desc

    def extract_saliency_maps(self, batch: torch.Tensor) -> torch.Tensor:
        """
        extract saliency maps. The saliency maps are extracted by averaging several attention heads from the last layer
        in of the CLS token. All values are then normalized to range between 0 and 1.
        :param batch: batch to extract saliency maps for. Has shape BxCxHxW.
        :return: a tensor of saliency maps. has shape Bxt-1
        """
        assert self.model_type == "dino_vits8", f"saliency maps are supported only for dino_vits model_type."
        self._extract_features(batch, [11], 'attn')
        head_idxs = [0, 2, 4, 5]
        curr_feats = self._feats[0] #Bxhxtxt
        cls_attn_map = curr_feats[:, head_idxs, 0, 1:].mean(dim=1) #Bx(t-1)
        temp_mins, temp_maxs = cls_attn_map.min(dim=1)[0], cls_attn_map.max(dim=1)[0]
        cls_attn_maps = (cls_attn_map - temp_mins) / (temp_maxs - temp_mins)  # normalize to range [0,1]
        return cls_attn_maps
     
def project_points_coords(pts, Rt, K):
    """
    :param pts:  [pn,3]
    :param Rt:   [rfn,3,4]
    :param K:    [rfn,3,3]
    :return:
        coords:         [rfn,pn,2]
        invalid_mask:   [rfn,pn]
        depth:          [rfn,pn,1]
    """
    pn = pts.shape[0]
    hpts = torch.cat([pts,torch.ones([pn,1],device=pts.device,dtype=torch.float32)],1)
    srn = Rt.shape[0]
    KRt = K @ Rt # rfn,3,4
    last_row = torch.zeros([srn,1,4],device=pts.device,dtype=torch.float32)
    last_row[:,:,3] = 1.0
    H = torch.cat([KRt,last_row],1) # rfn,4,4
    pts_cam = H[:,None,:,:] @ hpts[None,:,:,None]
    pts_cam = pts_cam[:,:,:3,0]
    depth = pts_cam[:,:,2:]
    invalid_mask = torch.abs(depth)<1e-4
    depth[invalid_mask] = 1e-3
    pts_2d = pts_cam[:,:,:2]/depth
    return pts_2d, ~(invalid_mask[...,0]), depth

def interpolate_feats(feats, points, h=None, w=None, padding_mode='zeros', align_corners=False, inter_mode='bilinear'):
    """

    :param feats:   b,f,h,w
    :param points:  b,n,2
    :param h:       float
    :param w:       float
    :param padding_mode:
    :param align_corners:
    :param inter_mode:
    :return: feats_inter: b,n,f
    """
    b, _, ch, cw = feats.shape
    if h is None and w is None:
        h, w = ch, cw
    x_norm = points[:, :, 0] / (w - 1) * 2 - 1
    y_norm = points[:, :, 1] / (h - 1) * 2 - 1
    points_norm = torch.stack([x_norm, y_norm], -1).unsqueeze(1)    # [srn,1,n,2]
    feats_inter = F.grid_sample(feats, points_norm, mode=inter_mode, padding_mode=padding_mode, align_corners=align_corners).squeeze(2)      # srn,f,n
    feats_inter = feats_inter.permute(0,2,1)
    return feats_inter

def create_init_grid(boundaries, step_size):
    x_lower, x_upper = boundaries['x_lower'], boundaries['x_upper']
    y_lower, y_upper = boundaries['y_lower'], boundaries['y_upper']
    z_lower, z_upper = boundaries['z_lower'], boundaries['z_upper']
    x = torch.arange(x_lower, x_upper, step_size, dtype=torch.float32) + step_size / 2
    y = torch.arange(y_lower, y_upper, step_size, dtype=torch.float32) + step_size / 2
    z = torch.arange(z_lower, z_upper, step_size, dtype=torch.float32) + step_size / 2
    xx, yy, zz = torch.meshgrid(x, y, z)
    coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    return coords, xx.shape

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model

def vis_tracking_pts(database, match_pts, sel_time):
    img_id = database.get_img_ids()[2]
    
    img = database.get_image(img_id, sel_time)[..., ::-1]
    
    color_cands = [(31,119,180), # in BGR
                (255,127,14),
                (44,160,44),
                (214,39,40),
                (148,103,189),]
    
    N = match_pts.shape[0]
    colors = color_cands[:N]
    
    Ks = database.get_K(img_id)
    pose = database.get_pose(img_id)
    
    fx = Ks[0, 0]
    fy = Ks[1, 1]
    cx = Ks[0, 2]
    cy = Ks[1, 2]
    
    match_pts = np.concatenate([match_pts, np.ones([N, 1])], axis=-1) # [N, 4]
    match_pts = np.matmul(pose, match_pts.T)[:3].T # [N, 3]
    
    match_pts_2d = match_pts[:, :2] / match_pts[:, 2:] # [N, 2]
    match_pts_2d[:, 0] = match_pts_2d[:, 0] * fx + cx
    match_pts_2d[:, 1] = match_pts_2d[:, 1] * fy + cy
    
    match_pts_2d = match_pts_2d.astype(np.int32)
    img = draw_keypoints(img, match_pts_2d, colors, radius=5)
    
    return img

def vis_tracking_multimodal_pts(database, match_pts_list, conf_list, sel_time, mask, view_idx = 0):
    # :param match_pts_list: list of [num_pts, 3]
    # :mask: [num_view, H, W, NQ] numpy array
    img_id = database.get_img_ids()[view_idx]
    
    img = database.get_image(img_id, sel_time)[..., ::-1]
    
    color_cands = [(31,119,180), # in BGR
                (255,127,14),
                (44,160,44),
                (214,39,40),
                (148,103,189),]
    
    Ks = database.get_K(img_id)
    pose = database.get_pose(img_id)
    
    fx = Ks[0, 0]
    fy = Ks[1, 1]
    cx = Ks[0, 2]
    cy = Ks[1, 2]
    
    for i, match_pts in enumerate(match_pts_list):
        # topk = min(5,match_pts.shape[0])
        # conf = conf_list[i]
        # topk_conf_idx = np.argpartition(conf, -topk)[-topk:]
        num_pts = match_pts.shape[0]
        # colors = color_cands[:num_pts]
        cmap = cm.get_cmap('viridis')
        colors = (cmap(np.linspace(0, 1, num_pts))[:, :3] * 255).astype(np.int32)[::-1, ::-1]
        match_pts = np.concatenate([match_pts, np.ones([num_pts, 1])], axis=-1) # [num_pts, 4]
        match_pts = np.matmul(pose, match_pts.T)[:3].T # [num_pts, 3]
        
        match_pts_2d = match_pts[:, :2] / match_pts[:, 2:] # [num_pts, 2]
        match_pts_2d[:, 0] = match_pts_2d[:, 0] * fx + cx
        match_pts_2d[:, 1] = match_pts_2d[:, 1] * fy + cy
        
        match_pts_2d = match_pts_2d.astype(np.int32)
        match_pts_2d = match_pts_2d.reshape(num_pts, 2)
        # img = draw_keypoints(img, match_pts_2d[topk_conf_idx], colors[topk_conf_idx], radius=5)
        img = draw_keypoints(img, match_pts_2d, colors, radius=5)
    
    # visualize the mask
    # mask = onehot2instance(mask) # [num_view, H, W]
    # mask = mask / mask.max() # [num_view, H, W]
    # num_view, H, W = mask.shape
    # cmap = cm.get_cmap('jet')
    # mask_vis = cmap(mask.reshape(-1)).reshape(num_view, H, W, 4)[..., :3] # [num_view, H, W, 3]
    # mask_vis = (mask_vis * 255).astype(np.uint8)
    # mask_vis = mask_vis[view_idx]
    
    # img = cv2.addWeighted(img, 0.5, mask_vis, 0.5, 0)
    
    return img

def octree_subsample(sim_vol, que_pts, last_res, topK):
    # :param sim_vol: [nd, N] # larger means more similar
    # :param que_pts: [nd, 3]
    # :param last_res: float
    # :param topK: float
    # :return child_que_pts = que_pts[alpha_mask] # [n_pts * 8, 3]
    assert sim_vol.shape[0] == que_pts.shape[0]
    sim_vol_topk, sim_vol_topk_idx = torch.topk(sim_vol, topK, dim=0, largest=True, sorted=False) # [topK, N]
    sim_vol_topk_idx = sim_vol_topk_idx.reshape(-1) # [topK*N]
    sim_vol_topk_idx = torch.unique(sim_vol_topk_idx) # [n_pts], n_pts <= topK*N
    sel_que_pts = que_pts[sim_vol_topk_idx] # [n_pts, 3]
    curr_res = last_res / 2
    
    child_offsets = torch.tensor([[curr_res, curr_res, curr_res],
                                  [curr_res, curr_res, -curr_res],
                                  [curr_res, -curr_res, curr_res],
                                  [curr_res, -curr_res, -curr_res],
                                  [-curr_res, curr_res, curr_res],
                                  [-curr_res, curr_res, -curr_res],
                                  [-curr_res, -curr_res, curr_res],
                                  [-curr_res, -curr_res, -curr_res]], dtype=torch.float32, device=que_pts.device) # [8, 3]
    child_que_pts = [sel_que_pts + child_offsets[i] for i in range(8)] # [n_pts * 8, 3]
    child_que_pts = torch.cat(child_que_pts, dim=0) # [n_pts * 8, 3]
    return child_que_pts, curr_res

def extract_kypts_gpu(sim_vol, que_pts, match_metric='sum'):
    # :param sim_vol: [n_pts, N] numpy array
    # :param que_pts: [n_pts, 3] numpy array
    # :return: [N, 3] numpy array
    N = sim_vol.shape[1]
    if type(sim_vol) is not torch.Tensor:
        sim_vol_tensor = torch.from_numpy(sim_vol).to("cuda:0") # [n_pts, N]
        que_pts_tensor = torch.from_numpy(que_pts).to("cuda:0") # [n_pts, 3]
    else:
        sim_vol_tensor = sim_vol
        que_pts_tensor = que_pts
    if match_metric == 'max':
        raise NotImplementedError
    elif match_metric == 'sum':
        # scale = 0.05
        # sim_vol_tensor = torch.exp(-sim_vol_tensor*scale)
        # match_pts_tensor = torch.zeros([N, 3]).cuda() # [N, 3]
        # for j in range(N):
        #     match_pts_tensor[j] = torch.sum(que_pts_tensor * sim_vol_tensor[:, j].unsqueeze(-1), dim=0) / torch.sum(sim_vol_tensor[:, j])
        
        # vectorized version
        match_pts_tensor = torch.sum(que_pts_tensor.unsqueeze(1) * sim_vol_tensor.unsqueeze(-1), dim=0) / torch.sum(sim_vol_tensor, dim=0).unsqueeze(-1) # [N, 3]
        # conf = sim_vol_tensor / torch.sum(sim_vol_tensor, dim=0).unsqueeze(0) # [n_pts, N]
        # conf = conf.max(dim=0)[0] # [N]
    return match_pts_tensor # , conf

def instance2onehot(instance, N = None):
    # :param instance: [**dim] numpy array uint8, val from 0 to N-1
    # :return: [**dim, N] numpy array bool
    if N is None:
        N = instance.max() + 1
    if type(instance) is np.ndarray:
        assert instance.dtype == np.uint8
        # assert instance.min() == 0
        H, W = instance.shape
        out = np.zeros(instance.shape + (N,), dtype=bool)
        for i in range(N):
            out[:, :, i] = (instance == i)
    elif type(instance) is torch.Tensor:
        assert instance.dtype == torch.uint8
        # assert instance.min() == 0
        out = torch.zeros(instance.shape + (N,), dtype=torch.bool, device=instance.device)
        for i in range(N):
            out[..., i] = (instance == i)
    return out

def onehot2instance(one_hot_mask):
    # :param one_hot_mask: [**dim, N] numpy array float32 or bool (probalistic or not)
    # :return: [**dim] numpy array uint8, val from 0 to N-1
    if type(one_hot_mask) == np.ndarray:
        return np.argmax(one_hot_mask, axis=-1).astype(np.uint8)
    elif type(one_hot_mask) == torch.Tensor:
        return torch.argmax(one_hot_mask, dim=-1).to(dtype=torch.uint8)
    else:
        raise NotImplementedError

class Fusion():
    def __init__(self, num_cam, feat_backbone=['dinov2'], device='cuda:0'):
        self.device = device
        
        # hyper-parameters
        self.mu = 0.02
        
        # curr_obs_torch is a dict contains:
        # - 'dino_feats': (K, patch_h, patch_w, feat_dim) torch tensor, dino features
        # - 'depth': (K, H, W) torch tensor, depth images
        # - 'pose': (K, 4, 4) torch tensor, poses of the images
        # - 'K': (K, 3, 3) torch tensor, intrinsics of the images
        self.curr_obs_torch = {}
        self.H = -1
        self.W = -1
        self.num_cam = num_cam
        
        # load feature extractor
        self.feat_backbone = feat_backbone
        if 'clip' in self.feat_backbone:
            self.clip_feat_extractor, _ = clip.load('ViT-L/14@336px', device=self.device)
        print(f"Loaded feature extractor {feat_backbone}")
        
        # # load GroundedSAM model
        # # config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'  # change the path of the model config file
        # config_file = 'models/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.cfg.py'
        # # grounded_checkpoint = 'ckpts/groundingdino_swint_ogc.pth'  # change the path of the model
        # grounded_checkpoint = 'models/ckpts/groundingdino_swinb_cogcoor.pth'
        # sam_checkpoint = 'models/ckpts/sam_vit_h_4b8939.pth'
        # self.ground_dino_model = load_model(config_file, grounded_checkpoint, device=self.device)

        # self.sam_model = SamPredictor(build_sam(checkpoint=sam_checkpoint))
        # self.sam_model.model = self.sam_model.model.to(self.device)
        
    def eval(self, pts, return_names=['dino_feats', 'mask'], return_inter=False):
        # :param pts: (N, 3) torch tensor in world frame
        # :param return_names: a set of {'dino_feats', 'mask'}
        # :return: output: dict contains:
        #          - 'dist': (N) torch tensor, dist to the closest point on the surface
        #          - 'dino_feats': (N, f) torch tensor, the features of the points
        #          - 'mask': (N, NQ) torch tensor, the query masks of the points
        #          - 'valid_mask': (N) torch tensor, whether the point is valid
        try:
            assert len(self.curr_obs_torch) > 0
        except:
            print('Please call update() first!')
            exit()
        assert type(pts) == torch.Tensor
        assert len(pts.shape) == 2
        assert pts.shape[1] == 3
        
        # transform pts to camera pixel coords and get the depth
        pts_2d, valid_mask, pts_depth = project_points_coords(pts, self.curr_obs_torch['pose'], self.curr_obs_torch['K'])
        # get grid point depths
        pts_depth = pts_depth[...,0] # [rfn,pn]
        
        # get interpolated depth and features
        # get surface depths
        inter_depth = interpolate_feats(self.curr_obs_torch['depth'].unsqueeze(1),
                                        pts_2d,
                                        h = self.H,
                                        w = self.W,
                                        padding_mode='zeros',
                                        align_corners=True,
                                        inter_mode='nearest')[...,0] # [rfn,pn,1]
        # inter_normal = interpolate_feats(self.curr_obs_torch['normals'].permute(0,3,1,2),
        #                                 pts_2d,
        #                                 h = self.H,
        #                                 w = self.W,
        #                                 padding_mode='zeros',
        #                                 align_corners=True,
        #                                 inter_mode='bilinear') # [rfn,pn,3]
        
        # compute the distance to the closest point on the surface
        dist = inter_depth - pts_depth # [rfn,pn]
        dist_valid = (inter_depth > -0.0001) & valid_mask & (dist > -self.mu) # [rfn,pn]
        
        # distance-based weight
        dist_weight = torch.exp(torch.clamp(self.mu-torch.abs(dist), max=0) / self.mu) # [rfn,pn]
        
        # # normal-based weight
        # fxfy = [torch.Tensor([self.curr_obs_torch['K'][i,0,0].item(), self.curr_obs_torch['K'][i,1,1].item()]) for i in range(self.num_cam)] # [rfn, 2]
        # fxfy = torch.stack(fxfy, dim=0).to(self.device) # [rfn, 2]
        # view_dir = pts_2d / fxfy[:, None, :] # [rfn,pn,2]
        # view_dir = torch.cat([view_dir, torch.ones_like(view_dir[...,0:1])], dim=-1) # [rfn,pn,3]
        # view_dir = view_dir / torch.norm(view_dir, dim=-1, keepdim=True) # [rfn,pn,3]
        # dist_weight = torch.abs(torch.sum(view_dir * inter_normal, dim=-1)) # [rfn,pn]
        # dist_weight = dist_weight * dist_valid.float() # [rfn,pn]
        
        dist = torch.clamp(dist, min=-self.mu, max=self.mu) # [rfn,pn]
        
        # # weighted distance
        # dist = (dist * dist_weight).sum(0) / (dist_weight.sum(0) + 1e-6) # [pn]
        
        # valid-weighted distance
        dist = (dist * dist_valid.float()).sum(0) / (dist_valid.float().sum(0) + 1e-6) # [pn]
        
        dist_all_invalid = (dist_valid.float().sum(0) == 0) # [pn]
        dist[dist_all_invalid] = 1e3
        
        outputs = {'dist': dist,
                   'valid_mask': ~dist_all_invalid}
        
        for k in return_names:
            inter_k = interpolate_feats(self.curr_obs_torch[k].permute(0,3,1,2),
                                        pts_2d, # here h, w are of images
                                        h = self.H,
                                        w = self.W,
                                        padding_mode='zeros',
                                        align_corners=True,
                                        inter_mode='bilinear') # [rfn,pn,k_dim]
            
            # weighted sum
            # val = (inter_k * dist_weight.unsqueeze(-1)).sum(0) / (dist_weight.float().sum(0).unsqueeze(-1) + 1e-6) # [pn,k_dim]
            
            # # valid-weighted sum
            val = (inter_k * dist_valid.float().unsqueeze(-1) * dist_weight.unsqueeze(-1)).sum(0) / (dist_valid.float().sum(0).unsqueeze(-1) + 1e-6) # [pn,k_dim]
            val[dist_all_invalid] = 0.0
            
            outputs[k] = val
            if return_inter:
                outputs[k+'_inter'] = inter_k
            else:
                del inter_k
        
        return outputs

    def eval_dist(self, pts):
        # this version does not clamp the distance or change the invalid points to 1e3
        # this is for grasper planner to find the grasping pose that does not penalize the depth
        # :param pts: (N, 3) torch tensor in world frame
        # :return: output: dict contains:
        #          - 'dist': (N) torch tensor, dist to the closest point on the surface
        try:
            assert len(self.curr_obs_torch) > 0
        except:
            print('Please call update() first!')
            exit()
        assert type(pts) == torch.Tensor
        assert len(pts.shape) == 2
        assert pts.shape[1] == 3
        
        # transform pts to camera pixel coords and get the depth
        pts_2d, valid_mask, pts_depth = project_points_coords(pts, self.curr_obs_torch['pose'], self.curr_obs_torch['K'])
        pts_depth = pts_depth[...,0] # [rfn,pn]
        
        # get interpolated depth and features
        inter_depth = interpolate_feats(self.curr_obs_torch['depth'].unsqueeze(1),
                                        pts_2d,
                                        h = self.H,
                                        w = self.W,
                                        padding_mode='zeros',
                                        align_corners=True,
                                        inter_mode='nearest')[...,0] # [rfn,pn,1]
        
        # compute the distance to the closest point on the surface
        dist = inter_depth - pts_depth # [rfn,pn]
        dist_valid = (inter_depth > -0.0001) & valid_mask # [rfn,pn]
        
        # valid-weighted distance
        dist = (dist * dist_valid.float()).sum(0) / (dist_valid.float().sum(0) + 1e-6) # [pn]
        
        dist_all_invalid = (dist_valid.float().sum(0) == 0) # [pn]
        
        outputs = {'dist': dist,
                   'valid_mask': ~dist_all_invalid}
        
        return outputs
        
    def batch_eval(self, pts, return_names=['dino_feats', 'mask']):
        batch_pts = 60000
        outputs = {}
        for i in tqdm(range(0, pts.shape[0], batch_pts)):
            st_idx = i
            ed_idx = min(i + batch_pts, pts.shape[0])
            out = self.eval(pts[st_idx:ed_idx], return_names=return_names)
            for k in out:
                if k not in outputs:
                    outputs[k] = [out[k]]
                else:
                    outputs[k].append(out[k])
        
        # concat the outputs
        for k in outputs:
            if outputs[k][0] is not None:
                outputs[k] = torch.cat(outputs[k], dim=0)
            else:
                outputs[k] = None
        return outputs
    
    def extract_dist_vol(self, boundaries):
        step = 0.002
        init_grid, grid_shape = create_init_grid(boundaries, step)
        init_grid = init_grid.to(self.device, dtype=torch.float32)
        
        batch_pts = 10000
        
        dist_vol = torch.zeros(init_grid.shape[0], dtype=torch.float32, device=self.device)
        valid_mask = torch.zeros(init_grid.shape[0], dtype=torch.bool, device=self.device)
        
        for i in range(0, init_grid.shape[0], batch_pts):
            st_idx = i
            ed_idx = min(i + batch_pts, init_grid.shape[0])
            out = self.eval(init_grid[st_idx:ed_idx], return_names={})
            
            dist_vol[st_idx:ed_idx] = out['dist']
            valid_mask[st_idx:ed_idx] = out['valid_mask']
        return {'init_grid': init_grid,
                'grid_shape': grid_shape,
                'dist': dist_vol,
                'valid_mask': valid_mask,}

    def extract_multiscale_clip_features(self, imgs, text, params):
        """Extract high resolution patch-level CLIP features for given images"""
        # !!! Don't use Center Crop, center crop will affect the feature interpolation !!!
        feat_dim = 768 # ViT-L/14@336px
        # skip_center_crop = True
        # K, H, W, _ = imgs.shape
        
        # Note that do not use center crop here, center crop will affect the feature fusion
        input_res = self.clip_feat_extractor.visual.input_resolution
        preprocess = T.Compose([
            T.Resize(input_res, interpolation=Image.BICUBIC),
            lambda image: image.convert("RGB"),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
        # To crop each image to several low resolution images i.e. image grids
        # batch_size = int(params['batch_size'])
        row_size = int(params['row_size'])
        column_size = int(params['column_size'])
        row_grid_num = int(imgs.shape[1] // row_size)
        column_grid_num = int(imgs.shape[2] // column_size)
        patch_h = int(input_res // 14)
        patch_w = int(column_size / row_size * input_res // 14)

        grids_list = []
        for img in imgs:
            grids = None
            for i in range(row_grid_num):
                for j in range(column_grid_num):
                    grid = img[i*row_size : (i+1)*row_size, j*column_size : (j+1)*column_size, :]
                    grid = Image.fromarray(grid)
                    grid = preprocess(grid) # shape = [3, input_res[0], input_res[1]], i.e. [3, 336, 336]
                    grid = grid.unsqueeze(0) # shape = [1, 3, input_res[0], input_res[1]], i.e. [1, 3, 336, 336]
                    if grids == None:
                        grids = grid
                    else:
                        grids = torch.cat((grids, grid), dim=0) # shape = [n_grids, 3, input_res[0], input_res[1]]
            grids = grids.to(self.device)
            grids_list.append(grids)

        grids_concat = torch.cat(grids_list, dim=0)
        with torch.no_grad():
            grids_concat_features = self.clip_feat_extractor.get_patch_encodings(grids_concat)
        grids_concat_features = torch.split(grids_concat_features, grids_list[0].shape[0], dim=0)

        features = []
        for idx, grids in enumerate(grids_list):

            grid_features = grids_concat_features[idx]
            grid_features = grid_features.reshape((grid_features.shape[0], patch_h, patch_w, feat_dim)).float() # shape = [n_grids, patch_h, patch_w, dim]

            # to concat the grid features to the raw image
            all_row_features = []
            for i in range(0, grids.shape[0], column_grid_num):
                single_row_features = []
                for j in range(0, column_grid_num):
                    single_row_features.append(grid_features[i+j])
                single_row_feature = torch.cat(single_row_features, dim=1) # shape = [patch_h, patch_w*j, dim]
                all_row_features.append(single_row_feature) # shape = [i, patch_h, patch_w*j, dim]
            feature = torch.cat(all_row_features, dim=0) # shape = [patch_h*i, patch_w*j, dim]
            feature = feature.unsqueeze(0) # shape = [1, patch_h*i, patch_w*j, dim]
            features.append(feature)    
        features = torch.cat(features, dim=0)
        
        # Get similarity of color and text
        color_embs = features / features.norm(dim=-1, keepdim=True)
        
        if text is not None:
            if not params['last_text_feat']:
                with torch.no_grad():
                    tokens = tokenize(text).to(self.device)
                    text_feature = self.clip_feat_extractor.encode_text(tokens).detach()
            else:
                text_feature = self.curr_obs_torch['text_feat']

            text_embs = text_feature / text_feature.norm(dim=-1, keepdim=True)
            text_embs = text_embs.float()

            # Compute similarities
            sims = color_embs @ text_embs.T
            sims = sims.repeat((1, 1, 1, 3))
            # print('time for color and text simliarity: ', time.time() - t3)
        else:
            text_feature = None
            sims = None

        return features, sims, text_feature

    def extract_clip_features(self, imgs, text, params):
        """Extract dense patch-level CLIP features for given images"""
        # !!! Don't use Center Crop, center crop will affect the feature interpolation !!!
        skip_center_crop = True
        feat_dim = 768 # ViT-L/14@336px
        patch_h = params['patch_h']
        patch_w = params['patch_w']

        K, H, W, _ = imgs.shape
        imgs_tensor = torch.zeros((K, 3, patch_h * 14, patch_w * 14), device=self.device)

        # To resize the image to a size that fits the patch size
        preprocess = T.Compose([
            T.Resize((patch_h * 14, patch_w * 14), interpolation=Image.BICUBIC),
            lambda image: image.convert("RGB"),
            T.CenterCrop((patch_h * 14, patch_w * 14)),
            T.ToTensor(),
            T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        # Patch the preprocess if we want to skip center crop
        if skip_center_crop:
            # Check there is exactly one center crop transform
            is_center_crop = [isinstance(t, T.CenterCrop) for t in preprocess.transforms]
            assert (
                sum(is_center_crop) == 1
            ), "There should be exactly one CenterCrop transform"
            # Create new preprocess without center crop
            preprocess = T.Compose(
                [t for t in preprocess.transforms if not isinstance(t, T.CenterCrop)]
            )
            print("Skipping center crop")

        for j in range(K):
            img = Image.fromarray(imgs[j])
            imgs_tensor[j] = preprocess(img)[:3]
        with torch.no_grad():
            features = self.clip_feat_extractor.get_patch_encodings(imgs_tensor)
            features = features.reshape((K, patch_h, patch_w, feat_dim)).float()
    
        # Get similarity of color and text
        color_embs = features / features.norm(dim=-1, keepdim=True)
        tokens = tokenize(text).to(self.device)
        text_embs = self.clip_feat_extractor.encode_text(tokens)
        text_embs /= text_embs.norm(dim=-1, keepdim=True)
        text_embs = text_embs.float()

        # Compute similarities
        sims = color_embs @ text_embs.T
        sims = sims.repeat((1, 1, 1, 3))    

        return features, sims        

    def extract_dinov2_features(self, imgs, params):
        K, H, W, _ = imgs.shape
        
        patch_h = params['patch_h']
        patch_w = params['patch_w']
        # feat_dim = 384 # vits14
        # feat_dim = 768 # vitb14
        feat_dim = 1024 # vitl14
        # feat_dim = 1536 # vitg14
        
        transform = T.Compose([
            # T.GaussianBlur(9, sigma=(0.1, 2.0)),
            T.Resize((patch_h * 14, patch_w * 14)),
            T.CenterCrop((patch_h * 14, patch_w * 14)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        imgs_tensor = torch.zeros((K, 3, patch_h * 14, patch_w * 14), device=self.device)

        # transform = T.Compose([
        #     T.Resize((720 * 840 // 1280, 840)),
        #     T.Pad(padding=(0, (840 - 720 * 840 // 1280) // 2)),
        #     T.ToTensor(),
        #     T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # ])        
        # imgs_tensor = torch.zeros((K, 3, 840, 840), device=self.device)
        
        for j in range(K):
            img = Image.fromarray(imgs[j])
            imgs_tensor[j] = transform(img)[:3]
        with torch.no_grad():
            # features_dict = self.dinov2_feat_extractor.forward_features(imgs_tensor)
            # features = features_dict['x_norm_patchtokens']
            # features = features.reshape((K, patch_h, patch_w, feat_dim))

            features = self.dinov2_feat_extractor.extract_descriptors(imgs_tensor.to(self.device), self.dinov2_layer, self.dinov2_facet)
            patch_size = self.dinov2_feat_extractor.model.patch_embed.patch_size[0]
            num_patch_h = int(patch_size / self.dinov2_stride * (imgs_tensor.shape[-2] // patch_size - 1) + 1)
            num_patch_w = int(patch_size / self.dinov2_stride * (imgs_tensor.shape[-1] // patch_size - 1) + 1)
            features = features.reshape((K, num_patch_h, num_patch_w, feat_dim))
            
        return features

    def extract_feature_correspondence(self, features1, features2):
        # apply co pca
        features = torch.cat((features1, features2), dim=0)
        features_pca, _ = self.apply_pca_colormap_return_proj(features, target_dim=128, normalize=True)
        
        features1_pca, features2_pca = features_pca.split((features1.shape[0], features2.shape[0]), dim=0)

        distances = torch.cdist(features1_pca, features2_pca)
        nearest_patch_dists, nearest_patch_indices = torch.min(distances, dim=1)

        return nearest_patch_dists, nearest_patch_indices

    def extract_image_dinov2_correspondence(self, mask1, mask2, image1, image2, features1, features2, mask=False):
        def polar_color_map(image_shape):
            h, w = image_shape[:2]
            x = np.linspace(-1, 1, w)
            y = np.linspace(-1, 1, h)
            xx, yy = np.meshgrid(x, y)

            # Find the center of the mask
            mask=mask2.cpu()
            mask_center = np.array(np.where(mask > 0))
            mask_center = np.round(np.mean(mask_center, axis=1)).astype(int)
            mask_center_y, mask_center_x = mask_center

            # Calculate distance and angle based on mask_center
            xx_shifted, yy_shifted = xx - x[mask_center_x], yy - y[mask_center_y]
            max_radius = np.sqrt(h**2 + w**2) / 2
            radius = np.sqrt(xx_shifted**2 + yy_shifted**2) * max_radius
            angle = np.arctan2(yy_shifted, xx_shifted) / (2 * np.pi) + 0.5

            angle = 0.2 + angle * 0.6  # Map angle to the range [0.25, 0.75]
            radius = np.where(radius <= max_radius, radius, max_radius)  # Limit radius values to the unit circle
            radius = 0.2 + radius * 0.6 / max_radius  # Map radius to the range [0.1, 1]

            return angle, radius

        WITH_MASK = True
        TARGET_DIM=16
        if WITH_MASK:
            TARGET_DIM=16
            mask1_t = torch.nn.functional.interpolate(torch.Tensor(mask1)[None, None, ...], (features1.shape[0], features1.shape[1]), mode='bilinear')[0,0]
            mask2_t = torch.nn.functional.interpolate(torch.Tensor(mask2)[None, None, ...], (features2.shape[0], features2.shape[1]), mode='bilinear')[0,0]
            OBJ_ID = 4
            scale_factor = 0.95
            mask1_t = (mask1_t > (OBJ_ID*scale_factor)).bool().reshape(-1)
            mask2_t = (mask2_t > (OBJ_ID*scale_factor)).bool().reshape(-1)
            
            features1_masked = features1.reshape(-1, features1.shape[-1])[mask1_t]
            features2_masked = features2.reshape(-1, features2.shape[-1])[mask2_t]
            
            features_masked = torch.cat((features1_masked, features2_masked), dim=0)
            features_pca, pca_matrix = self.apply_pca_colormap_return_proj(features_masked, target_dim=TARGET_DIM, normalize=True)
            features1_reshaped = features1.reshape(-1, features1.shape[-1])
            features2_reshaped = features2.reshape(-1, features2.shape[-1])
            features_reshaped = torch.cat((features1_reshaped, features2_reshaped), dim=0)
            features_reshaped = features_reshaped @ pca_matrix
            f1pca, f2pca = features_reshaped.split((features1_reshaped.shape[0], features2_reshaped.shape[0]), dim=0)
            
            f1pca = f1pca.reshape(features1.shape[-3], features1.shape[-2], TARGET_DIM)[..., :3].detach().cpu().numpy()
            f2pca = f2pca.reshape(features2.shape[-3], features2.shape[-2], TARGET_DIM)[..., :3].detach().cpu().numpy()
            cv2.imwrite('f1mask.png', f1pca * 255)
            cv2.imwrite('f2mask.png', f2pca * 255)
        
        else:
            TARGET_DIM=16
            features1_reshaped = features1.reshape(-1, features1.shape[-1])
            features2_reshaped = features2.reshape(-1, features2.shape[-1])
            features_reshaped = torch.cat((features1_reshaped, features2_reshaped), dim=0)
            features_pca, pca_matrix = self.apply_pca_colormap_return_proj(features_reshaped, target_dim=TARGET_DIM, normalize=True)
            features1_pca, features2_pca = features_pca.split((features1_reshaped.shape[0], features2_reshaped.shape[0]), dim=0)
            
            f1pca = features1_pca.reshape(features1.shape[-3], features1.shape[-2], TARGET_DIM)[..., :3].detach().cpu().numpy()
            f2pca = features2_pca.reshape(features2.shape[-3], features2.shape[-2], TARGET_DIM)[..., :3].detach().cpu().numpy()
            
            cv2.imwrite('f1.png', f1pca * 255)
            cv2.imwrite('f2.png', f2pca * 255)
        

        features1 = features1 / features1.norm(dim=-1, keepdim=True)
        features2 = features2 / features2.norm(dim=-1, keepdim=True)
        
        
        # resize the image to the shape of the feature map
        resized_image1 = resize(image1, features1.shape[2], resize=True, to_pil=False)
        resized_image2 = resize(image2, features2.shape[2], resize=True, to_pil=False)

        if mask: # mask the features
            resized_mask1 = F.interpolate(mask1.cuda().unsqueeze(0).unsqueeze(0).float(), size=features1.shape[:2], mode='nearest')
            resized_mask2 = F.interpolate(mask2.cuda().unsqueeze(0).unsqueeze(0).float(), size=features2.shape[:2], mode='nearest')
            features1 = features1 * resized_mask1.repeat(1, features1.shape[1], 1, 1)
            features2 = features2 * resized_mask2.repeat(1, features2.shape[1], 1, 1)
            # set where mask==0 a very large number
            features1[(features1.sum(1)==0).repeat(1, features1.shape[1], 1, 1)] = 100000
            features2[(features2.sum(1)==0).repeat(1, features2.shape[1], 1, 1)] = 100000

        features1_2d = features1.reshape(features1.shape[1], -1).permute(1, 0).cpu().detach().numpy()
        features2_2d = features2.reshape(features2.shape[1], -1).permute(1, 0).cpu().detach().numpy()

        features1_2d = torch.tensor(features1_2d).to("cuda")
        features2_2d = torch.tensor(features2_2d).to("cuda")
        resized_image1 = torch.tensor(resized_image1).to("cuda").float()
        resized_image2 = torch.tensor(resized_image2).to("cuda").float()

        mask1 = F.interpolate(mask1.cuda().unsqueeze(0).unsqueeze(0).float(), size=resized_image1.shape[:2], mode='nearest').squeeze(0).squeeze(0)
        mask2 = F.interpolate(mask2.cuda().unsqueeze(0).unsqueeze(0).float(), size=resized_image2.shape[:2], mode='nearest').squeeze(0).squeeze(0)

        # Mask the images
        resized_image1 = resized_image1 * mask1.unsqueeze(-1).repeat(1, 1, 3)
        resized_image2 = resized_image2 * mask2.unsqueeze(-1).repeat(1, 1, 3)
        # Normalize the images to the range [0, 1]
        resized_image1 = (resized_image1 - resized_image1.min()) / (resized_image1.max() - resized_image1.min())
        resized_image2 = (resized_image2 - resized_image2.min()) / (resized_image2.max() - resized_image2.min())

        angle, radius = polar_color_map(resized_image2.shape)

        angle_mask = angle * mask2.cpu().numpy()
        radius_mask = radius * mask2.cpu().numpy()

        hsv_mask = np.zeros(resized_image2.shape, dtype=np.float32)
        hsv_mask[:, :, 0] = angle_mask
        hsv_mask[:, :, 1] = radius_mask
        hsv_mask[:, :, 2] = 1

        rainbow_mask2 = cv2.cvtColor((hsv_mask * 255).astype(np.uint8), cv2.COLOR_HSV2BGR) / 255

        # Apply the rainbow mask to image2
        rainbow_image2 = rainbow_mask2 * mask2.cpu().numpy()[:, :, None]

        # Create a white background image
        background_color = np.array([1, 1, 1], dtype=np.float32)
        background_image = np.ones(resized_image2.shape, dtype=np.float32) * background_color

        # Apply the rainbow mask to image2 only in the regions where mask2 is 1
        rainbow_image2 = np.where(mask2.cpu().numpy()[:, :, None] == 1, rainbow_mask2, background_image)
        
        nearest_patches = []

        distances = torch.cdist(features1_2d, features2_2d)
        nearest_patch_indices = torch.argmin(distances, dim=1)
        nearest_patches = torch.index_select(torch.tensor(rainbow_mask2).cuda().reshape(-1, 3), 0, nearest_patch_indices)

        nearest_patches_image = nearest_patches.reshape(resized_image1.shape)
        rainbow_image2 = torch.tensor(rainbow_image2).to("cuda")

        # TODO: upsample the nearest_patches_image to the resolution of the original image
        # nearest_patches_image = F.interpolate(nearest_patches_image.permute(2,0,1).unsqueeze(0), size=256, mode='bilinear').squeeze(0).permute(1,2,0)
        # rainbow_image2 = F.interpolate(rainbow_image2.permute(2,0,1).unsqueeze(0), size=256, mode='bilinear').squeeze(0).permute(1,2,0)

        nearest_patches_image = (nearest_patches_image).cpu().numpy()
        resized_image2 = (rainbow_image2).cpu().numpy()

        fig_colormap, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.axis('off')
        ax2.axis('off')
        ax1.imshow(resized_image2)
        ax2.imshow(nearest_patches_image)
        fig_colormap.savefig(f'{save_path}/{category[0]}/{pair_idx}_colormap.png')
        plt.close(fig_colormap)
        
        return nearest_patches_image, resized_image2

    def extract_features(self, imgs, params):
        # :param imgs (K, H, W, 3) np array, color images
        # :param params: dict contains:
        #                - 'patch_h', 'patch_w': int, the size of the patch
        # :return features: (K, patch_h, patch_w, feat_dim) np array, features of the images
        if 'dinov2' in self.feat_backbone:
            return self.extract_dinov2_features(imgs, params)
        elif 'dino' in self.feat_backbone:
            K, H, W, _ = imgs.shape
            
            transform = T.Compose([
                # T.GaussianBlur(9, sigma=(0.1, 2.0)),
                T.Resize((H, W)),
                T.CenterCrop((H, W)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            
            imgs_tensor = torch.zeros((K, 3, H, W), device=self.device)
            for j in range(K):
                img = Image.fromarray(imgs[j])
                imgs_tensor[j] = transform(img)[:3]
            with torch.no_grad():
                features = self.feat_extractor.extract_descriptors(imgs_tensor)
            return features
        else:
            raise NotImplementedError
       
    def update(self, obs, text, last_text_feat=False, visualize=False):
        # :param obs: dict contains:
        #             - 'color': (K, H, W, 3) np array, color image
        #             - 'depth': (K, H, W) np array, depth image
        #             - 'pose': (K, 4, 4) np array, camera pose
        #             - 'K': (K, 3, 3) np array, camera intrinsics
        self.num_cam = obs['color'].shape[0]
        color = obs['color']
        params = {
            'patch_h': color.shape[1] // 14,
            'patch_w': color.shape[2] // 14,
            'row_size': color.shape[1] / 3,
            'column_size': color.shape[2] /4,
            'batch_size': 12,
            'last_text_feat': last_text_feat,
        }

        if 'clip' in self.feat_backbone:
            # maskclip
            # color_features, color_text_sim = self.extract_clip_features(color, text, params)
            color_features, color_text_sim, text_feature = self.extract_multiscale_clip_features(color, text, params)

            self.curr_obs_torch['clip_feats'] = color_features
            self.curr_obs_torch['clip_sims'] = color_text_sim
            self.curr_obs_torch['text_feat'] = text_feature
            features = color_features

            # if text is not None and visualize:
            #     # visualize similarity of clip features
            #     color_text_sim = color_text_sim[..., 0]
            #     plt.figure()
            #     cmap = plt.get_cmap("turbo")
            #     for i in range(color_text_sim.shape[0]):
            #         sim = color_text_sim[i]
            #         sim_norm = (sim - sim.min()) / (sim.max() - sim.min())
            #         heatmap = cmap(sim_norm.detach().cpu().numpy())
                    
            #         plt.subplot(1, 2, 1)
            #         plt.imshow(color[i])
            #         plt.axis("off")

            #         plt.subplot(1, 2, 2)
            #         sim_norm = (sim - sim.min()) / (sim.max() - sim.min())
            #         heatmap = cmap(sim_norm.detach().cpu().numpy())                
            #         plt.imshow(heatmap)
            #         plt.axis("off")

            #         plt.tight_layout()
            #         plt.show()
            
        if 'dino' in self.feat_backbone:
            features = self.extract_features(color, params)
            self.curr_obs_torch['dino_feats'] = features
            self.curr_obs_torch['text_feat'] = None
            
        elif 'dinov2' in self.feat_backbone:
            features = self.extract_features(color, params)
            self.curr_obs_torch['dinov2_feats'] = features
            self.curr_obs_torch['text_feat'] = None
            
        # if visualize:
        #     # pca feature
        #     plt.figure()
        #     for i, feat in enumerate(features):
        #         pca_feat = get_pca_map(feat, color.shape[1:3])
        #         pca_feat = (pca_feat * 255).astype(int)

        #         plt.subplot(1, 2, 1)
        #         plt.imshow(color[i])
        #         plt.axis("off")

        #         plt.subplot(1, 2, 2)               
        #         plt.imshow(pca_feat)
        #         plt.axis("off")

        #         plt.tight_layout()
        #         plt.show()
        
        self.curr_obs_torch['color'] = color
        normals_np = [depth2normal(obs['depth'][i], obs['K'][i]) for i in range(self.num_cam)]
        normals_np = np.stack(normals_np, axis=0)
        self.curr_obs_torch['normals'] = torch.from_numpy(normals_np).to(self.device, dtype=torch.float32)
        self.curr_obs_torch['color_tensor'] = torch.from_numpy(color).to(self.device, dtype=torch.float32) / 255.0
        self.curr_obs_torch['depth'] = torch.from_numpy(obs['depth']).to(self.device, dtype=torch.float32)
        self.curr_obs_torch['pose'] = torch.from_numpy(obs['pose']).to(self.device, dtype=torch.float32)
        self.curr_obs_torch['K'] = torch.from_numpy(obs['K']).to(self.device, dtype=torch.float32)
        
        _, self.H, self.W = obs['depth'].shape
    
    def compute_alignment(self, v_i, v_j, inst_n, inst_m):
        # :param v_i, view index i
        # :param v_j, view index j
        # :param inst_n, instance index in view i
        # :param inst_m, instance index in view j
        # :return: visiblity score (# of visible points)
        
        # extract information from view i
        mask_i = self.curr_obs_torch['mask'][v_i] # [H,W]
        mask_i_inst_n = (mask_i == inst_n)
        depth_i = self.curr_obs_torch['depth'][v_i] # [H,W]
        pose_i = self.curr_obs_torch['pose'][v_i]
        pose_i_homo = torch.cat([pose_i, torch.tensor([[0, 0, 0, 1]], device=self.device, dtype=torch.float32)], dim=0)
        pose_i_inv = torch.inverse(pose_i_homo)
        K_i = self.curr_obs_torch['K'][v_i]
        depth_i = self.curr_obs_torch['depth'][v_i]
        camera_param_i = [K_i[0, 0].item(),
                            K_i[1, 1].item(),
                            K_i[0, 2].item(),
                            K_i[1, 2].item(),]
        
        pcd_i_np = depth2fgpcd(depth_i.cpu().numpy(), mask_i_inst_n.cpu().numpy(), camera_param_i)
        pcd_i = torch.from_numpy(pcd_i_np).to(self.device, dtype=torch.float32)
        pcd_i = torch.cat([pcd_i.T, torch.ones((1, pcd_i.shape[0]), device=self.device, dtype=torch.float32)], dim=0)
        pcd_i = (pose_i_inv @ pcd_i)[:3].T
        
        # extract information from view j        
        mask_j = self.curr_obs_torch['mask'][v_j] # [H,W]
        mask_j_inst_m = (mask_j == inst_m)
        depth_j = self.curr_obs_torch['depth'][v_j] # [H,W]
        pose_j = self.curr_obs_torch['pose'][v_j]
        pose_j_homo = torch.cat([pose_j, torch.tensor([[0, 0, 0, 1]], device=self.device, dtype=torch.float32)], dim=0)
        pose_j_inv = torch.inverse(pose_j_homo)
        K_j = self.curr_obs_torch['K'][v_j]
        depth_j = self.curr_obs_torch['depth'][v_j]
        camera_param_j = [K_j[0, 0].item(),
                            K_j[1, 1].item(),
                            K_j[0, 2].item(),
                            K_j[1, 2].item(),]
        
        pcd_j_np = depth2fgpcd(depth_j.cpu().numpy(), mask_j_inst_m.cpu().numpy(), camera_param_j)
        pcd_j = torch.from_numpy(pcd_j_np).to(self.device, dtype=torch.float32)
        pcd_j = torch.cat([pcd_j.T, torch.ones((1, pcd_j.shape[0]), device=self.device, dtype=torch.float32)], dim=0)
        pcd_j = (pose_j_inv @ pcd_j)[:3].T
        
        # project i in j
        pts_2d_i_in_j, valid_mask, pts_depth_i_in_j = project_points_coords(pcd_i, pose_j[None], K_j[None]) # [1,N,2], [1,N], [1,N,1]
        pts_depth_i_in_j = pts_depth_i_in_j[...,0] # [1,N]
        inter_depth_i_in_j = interpolate_feats(depth_j[None,None],
                                               pts_2d_i_in_j,
                                               h = self.H,
                                               w = self.W,
                                               padding_mode='zeros',
                                               align_corners=True,
                                               inter_mode='nearest')[...,0] # [1,N]
        
        inter_mask_i_in_j = interpolate_feats(mask_j_inst_m[None,None].to(torch.float32),
                                              pts_2d_i_in_j,
                                              h=self.H,
                                              w=self.W,
                                              padding_mode='zeros',
                                              align_corners=True,
                                              inter_mode='nearest')[...,0] # [1,N]
        inter_mask_i_in_j = inter_mask_i_in_j > 0.5
        
        # project j in i
        pts_2d_j_in_i, valid_mask, pts_depth_j_in_i = project_points_coords(pcd_j, pose_i[None], K_i[None]) # [1,N,2], [1,N], [1,N,1]
        pts_depth_j_in_i = pts_depth_j_in_i[...,0] # [1,N]
        inter_depth_j_in_i = interpolate_feats(depth_i[None,None],
                                               pts_2d_j_in_i,
                                               h = self.H,
                                               w = self.W,
                                               padding_mode='zeros',
                                               align_corners=True,
                                               inter_mode='nearest')[...,0]
        inter_mask_j_in_i = interpolate_feats(mask_i_inst_n[None,None].to(torch.float32),
                                                pts_2d_j_in_i,
                                                h=self.H,
                                                w=self.W,
                                                padding_mode='zeros',
                                                align_corners=True,
                                                inter_mode='nearest')[...,0] # [1,N]

        inter_mask_j_in_i = inter_mask_j_in_i > 0.5

        num_vis = ((torch.abs(pts_depth_i_in_j - inter_depth_i_in_j) < 0.03 ) & inter_mask_i_in_j).sum().item() +\
                    ((torch.abs(pts_depth_j_in_i - inter_depth_j_in_i) < 0.03 ) & inter_mask_j_in_i).sum().item()
        
        return num_vis
    
    def add_mask_in_j_use_i_inst_n(self, v_i, v_j, inst_n):
        # :param v_i, view index i with inst_n
        # :param v_j, view index j with missing mask
                # :param v_i, view index i
        # :param v_j, view index j
        # :param inst_n, instance index in view i
        # :param inst_m, instance index in view j
        # :return: visiblity score (# of visible points)
        
        # extract information from view i
        mask_i = self.curr_obs_torch['mask'][v_i] # [H,W]
        mask_i_inst_n = (mask_i == inst_n)
        depth_i = self.curr_obs_torch['depth'][v_i] # [H,W]
        pose_i = self.curr_obs_torch['pose'][v_i]
        pose_i_homo = torch.cat([pose_i, torch.tensor([[0, 0, 0, 1]], device=self.device, dtype=torch.float32)], dim=0)
        pose_i_inv = torch.inverse(pose_i_homo)
        K_i = self.curr_obs_torch['K'][v_i]
        depth_i = self.curr_obs_torch['depth'][v_i]
        camera_param_i = [K_i[0, 0].item(),
                            K_i[1, 1].item(),
                            K_i[0, 2].item(),
                            K_i[1, 2].item(),]
        
        mask_i_inst_n = mask_i_inst_n.cpu().numpy().astype(np.uint8)
        mask_i_inst_n = (cv2.erode(mask_i_inst_n * 255, np.ones([10, 10], np.uint8), iterations=1) / 255).astype(bool)
        pcd_i_np = depth2fgpcd(depth_i.cpu().numpy(), mask_i_inst_n, camera_param_i)
        pcd_i = torch.from_numpy(pcd_i_np).to(self.device, dtype=torch.float32)
        pcd_i = torch.cat([pcd_i.T, torch.ones((1, pcd_i.shape[0]), device=self.device, dtype=torch.float32)], dim=0)
        pcd_i = (pose_i_inv @ pcd_i)[:3].T
        
        # extract information from view j        
        depth_j = self.curr_obs_torch['depth'][v_j] # [H,W]
        pose_j = self.curr_obs_torch['pose'][v_j]
        K_j = self.curr_obs_torch['K'][v_j]
        depth_j = self.curr_obs_torch['depth'][v_j]
        
        # project i in j
        pts_2d_i_in_j, valid_mask, pts_depth_i_in_j = project_points_coords(pcd_i, pose_j[None], K_j[None]) # [1,N,2], [1,N], [1,N,1]
        pts_depth_i_in_j = pts_depth_i_in_j[...,0] # [1,N]
        inter_depth_i_in_j = interpolate_feats(depth_j[None,None],
                                               pts_2d_i_in_j,
                                               h = self.H,
                                               w = self.W,
                                               padding_mode='zeros',
                                               align_corners=True,
                                               inter_mode='nearest')[...,0] # [1,N]
        pts_i_visible_in_j = ((torch.abs(pts_depth_i_in_j - inter_depth_i_in_j) < 0.02) & valid_mask)[0] # [N]
        
        pts_2d_i_in_j = pts_2d_i_in_j[0][pts_i_visible_in_j] # [M, 2]
        
        if pts_2d_i_in_j.shape[0] == 0:
            return None
        
        input_point = pts_2d_i_in_j.detach().cpu().numpy().astype(np.int32)
        input_label = np.ones((input_point.shape[0],), dtype=np.int32)

        self.sam_model.set_image(self.curr_obs_torch['color'][v_j])
        masks, _, _ = self.sam_model.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        return masks[0]
        
    def compute_alignment_with_consensus(self, v_i, inst_n, consensus_mask_dict, consensus_i):
        curr_view_inst_dict = consensus_mask_dict[consensus_i]
        avg_align = []
        for curr_view, curr_inst in curr_view_inst_dict.items():
            if curr_view == v_i or curr_inst == -1:
                continue
            avg_align.append(self.compute_alignment(curr_view, v_i, curr_inst, inst_n))
        avg_align = np.mean(avg_align)
        return avg_align
    
    def update_with_missing_label(self, consensus_mask_label, consensus_mask_dict, v_i):
        len_i = len(self.curr_obs_torch['mask_label'][v_i])
        mask_label_i = self.curr_obs_torch['mask_label'][v_i]
        len_curr = len(consensus_mask_label)
        assert len_i < len_curr
        for label_i, label in enumerate(mask_label_i):
            align_ls = []
            for consensus_i, consensus_label in enumerate(consensus_mask_label):
                if label != consensus_label:
                    # skip when the label is not the same
                    align_ls.append(-1)
                    continue
                align_ls.append(self.compute_alignment_with_consensus(v_i, label_i, consensus_mask_dict, consensus_i))
            matched_consensus_i = np.argmax(align_ls)
            try:
                assert v_i not in consensus_mask_dict[matched_consensus_i]
            except:
                print(f'{v_i} is already in {consensus_mask_dict} at {matched_consensus_i}')
                raise AssertionError
            consensus_mask_dict[matched_consensus_i][v_i] = label_i
        # add the missing label
        for consensus_i, consensus_label in enumerate(consensus_mask_label):
            if v_i in consensus_mask_dict[consensus_i]:
                continue
            consensus_mask_dict[consensus_i][v_i] = -1
            
            # # add mask
            # existing_match = consensus_mask_dict[consensus_i]
            # successful_adding = False
            # for v_j, inst_j in existing_match.items():
            #     if inst_j == -1:
            #         continue
            #     add_mask = self.add_mask_in_j_use_i_inst_n(v_j, v_i, inst_j)
            #     if add_mask is None:
            #         continue
            #     else:
            #         successful_adding = True
            #         break
            # if not successful_adding:
            #     consensus_mask_dict[consensus_i][v_i] = -1
            # else:
            #     consensus_mask_dict[consensus_i][v_i] = len(self.curr_obs_torch['mask_label'][v_i])
            #     self.curr_obs_torch['mask'][v_i][add_mask > 0.5] = len(self.curr_obs_torch['mask_label'][v_i])
            #     self.curr_obs_torch['mask_label'][v_i].append(consensus_label)
            
        return consensus_mask_label, consensus_mask_dict
    
    def update_with_equal_label(self, consensus_mask_label, consensus_mask_dict, v_i):
        # the instance number of view i
        len_i = len(self.curr_obs_torch['mask_label'][v_i])
        mask_label_i = self.curr_obs_torch['mask_label'][v_i]
        # the instance number of view 0
        len_curr = len(consensus_mask_label)
        assert len_i == len_curr
        # for each instance in view i
        for label_i, label in enumerate(mask_label_i):
            align_ls = []
            # compute its alignment of instances in view 0
            for consensus_i, consensus_label in enumerate(consensus_mask_label):
                if label != consensus_label:
                    # skip when the label is not the same
                    align_ls.append(-1)
                    continue
                # compute the alignment of instances with the same label in view 0
                align_ls.append(self.compute_alignment_with_consensus(v_i, label_i, consensus_mask_dict, consensus_i))
            max_align_ls = np.max(align_ls)
            if max_align_ls > 0:
                matched_consensus_i = np.argmax(align_ls)
                try:
                    assert v_i not in consensus_mask_dict[matched_consensus_i]
                except:
                    print(f'{v_i} is already in {consensus_mask_dict} at {matched_consensus_i}')
                    raise AssertionError
                consensus_mask_dict[matched_consensus_i][v_i] = label_i
        # assert whether all labels are matched
        for consensus_i, consensus_label in enumerate(consensus_mask_label):
            try:
                assert v_i in consensus_mask_dict[consensus_i]
            except:
                print(f'{v_i} is not in {consensus_mask_dict} at {consensus_i}')
                print(f'hint: curr view label {mask_label_i}')
                print(f'hint: curr consensus label {consensus_mask_label}')
                exit()
        return consensus_mask_label, consensus_mask_dict
    
    def update_with_additional_label(self, consensus_mask_label, consensus_mask_dict, v_i):
        len_i = len(self.curr_obs_torch['mask_label'][v_i])
        mask_label_i = self.curr_obs_torch['mask_label'][v_i]
        len_curr = len(consensus_mask_label)
        assert len_i > len_curr
        matched_label_idx_ls = []
        for consensus_i, consensus_label in enumerate(consensus_mask_label):
            align_ls = []
            for label_i, label in enumerate(mask_label_i):
                if label != consensus_label:
                    # skip when the label is not the same
                    align_ls.append(-1)
                    continue
                align_ls.append(self.compute_alignment_with_consensus(v_i, label_i, consensus_mask_dict, consensus_i))
            matched_label_i = np.argmax(align_ls)
            consensus_mask_dict[consensus_i][v_i] = matched_label_i
            matched_label_idx_ls.append(matched_label_i)
        
        # do nothing
        for inst_i in range(len_i):
            if inst_i in matched_label_idx_ls:
                continue
            consensus_mask_label.append(mask_label_i[inst_i])
            consensus_mask_dict[len_curr] = {v_i: inst_i}
            for prev_v_i in range(v_i):
                consensus_mask_dict[len_curr][prev_v_i] = -1
            len_curr += 1
        
        # # add additional label
        # for inst_i in range(len_i):
        #     if inst_i in matched_label_idx_ls:
        #         continue
        #     consensus_mask_label.append(mask_label_i[inst_i])
        #     consensus_mask_dict[len_curr] = {v_i: inst_i}
        #     for prev_v_i in range(v_i):
        #         add_mask = self.add_mask_in_j_use_i_inst_n(v_i, prev_v_i, inst_i)
        #         if add_mask is None:
        #             consensus_mask_dict[len_curr][prev_v_i] = -1
        #         else:
        #             consensus_mask_dict[len_curr][prev_v_i] = len(self.curr_obs_torch['mask_label'][prev_v_i])
        #             self.curr_obs_torch['mask'][prev_v_i][add_mask > 0.5] = len(self.curr_obs_torch['mask_label'][prev_v_i])
        #             self.curr_obs_torch['mask_label'][prev_v_i].append(mask_label_i[inst_i])
        #     len_curr += 1
                
        return consensus_mask_label, consensus_mask_dict

    def adjust_masks_using_consensus(self, consensus_mask_label, consensus_mask_dict, use_other):
        old_masks = self.curr_obs_torch['mask'].clone()
        new_masks = self.curr_obs_torch['mask'].clone()
        H, W = old_masks.shape[1:]
        for consensus_i, consensus_label in enumerate(consensus_mask_label):
            
            visible_views = []
            for v_i, label_i in consensus_mask_dict[consensus_i].items():
                if label_i != -1:
                    visible_views.append(v_i)
                    
            for v_i, label_i in consensus_mask_dict[consensus_i].items():
                if label_i == -1:
                    # continue
                    
                    # add additional label
                    for v_j in visible_views:
                        inst_j = consensus_mask_dict[consensus_i][v_j]
                        if use_other:
                            add_mask = self.add_mask_in_j_use_i_inst_n(v_j, v_i, inst_j)
                        else:
                            add_mask = None
                        if add_mask is None:
                            continue
                        consensus_mask_dict[consensus_i][v_i] = len(self.curr_obs_torch['mask_label'][v_i])
                        new_masks[v_i][add_mask > 0.5] = consensus_i
                        self.curr_obs_torch['mask_label'][v_i].append(consensus_label)
                        break
                else:
                    mask_i = old_masks[v_i]
                    mask_i_inst_label_i = (mask_i == label_i)
                    new_masks[v_i][mask_i_inst_label_i] = consensus_i
        self.curr_obs_torch['mask'] = new_masks
    
    def order_consensus(self, consensus_mask_label, consensus_mask_dict, queries):
        queries_wo_period = [query[:-1] if query[-1] == '.' else query for query in queries]
        # the order of consensus is the order of queries
        new_consesus_mask_label = [consensus_mask_label[0]]
        new_consesus_mask_dict = {0: consensus_mask_dict[0]}
        for query in queries_wo_period:
            # find all indices of query in consensus_mask_label
            for i, label in enumerate(consensus_mask_label):
                if label == query:
                    new_consesus_mask_label.append(label)
                    new_consesus_mask_dict[len(new_consesus_mask_label)-1] = consensus_mask_dict[i]
        return new_consesus_mask_label, new_consesus_mask_dict
                    
    def align_instance_mask_v2(self, queries, use_other=True):
        num_view = self.curr_obs_torch['color'].shape[0]
        assert num_view > 0
        consensus_mask_label = self.curr_obs_torch['mask_label'][0].copy()
        consensus_mask_dict = {} # map label idx to a dict, each dict contains the view index and the instance id
        for i, label in enumerate(consensus_mask_label):
            consensus_mask_dict[i] = {0: i}
            
            # if label not in consensus_mask_dict:
            #     consensus_mask_dict[label] = [{0: i},]
            # else:
            #     consensus_mask_dict[label].append({0: i})
        
        for i in range(1, num_view):
            len_i = len(self.curr_obs_torch['mask_label'][i])
            len_curr = len(consensus_mask_label)
            
            if len_i < len_curr:
                # some mask labels are missing in view i
                consensus_mask_label, consensus_mask_dict = self.update_with_missing_label(consensus_mask_label, consensus_mask_dict, i)
            elif len_i == len_curr:
                assert consensus_mask_label == self.curr_obs_torch['mask_label'][i]
                # all mask labels are matched
                consensus_mask_label, consensus_mask_dict = self.update_with_equal_label(consensus_mask_label, consensus_mask_dict, i)
            else:
                consensus_mask_label, consensus_mask_dict = self.update_with_additional_label(consensus_mask_label, consensus_mask_dict, i)
                consensus_mask_label, consensus_mask_dict = self.order_consensus(consensus_mask_label, consensus_mask_dict, queries)
        
        self.adjust_masks_using_consensus(consensus_mask_label, consensus_mask_dict, use_other = use_other)
        self.curr_obs_torch['consensus_mask_label'] = consensus_mask_label
    
    def align_with_prev_mask(self, mask):
        # :param new_mask: [num_cam, H, W, num_instance] torch tensor, the new detected mask
        out_mask = torch.zeros_like(mask).to(self.device, dtype=torch.bool)
        for cam_i in range(self.num_cam):
            for instance_i in range(len(self.track_ids)):
                mask_i = mask[cam_i, ..., instance_i]
                intersec_nums = (mask_i[..., None] & self.curr_obs_torch['mask'][cam_i].to(torch.bool)).sum(dim=(0,1)) # [num_instance]
                correct_label = intersec_nums.argmax()
                out_mask[cam_i, ..., correct_label] = mask_i
        out_mask = out_mask.to(self.device, dtype=torch.uint8)
        return out_mask
    
    def text_queries_for_inst_mask_no_track(self, queries, thresholds, merge_all = False):
        query_mask = torch.zeros((self.num_cam, self.H, self.W), device=self.device)
        labels = []
        for i in range(self.num_cam):
            mask, label = grounded_instance_sam_bacth_queries_np(self.curr_obs_torch['color'][i], queries, self.ground_dino_model, self.sam_model, thresholds, merge_all)
            labels.append(label)
            query_mask[i] = mask
        self.curr_obs_torch['mask'] = query_mask # [num_cam, H, W]

        # !!! what if the label is different for different camera views !!!
        self.curr_obs_torch['mask_label'] = labels # [num_cam, ] list of list
        _, idx = np.unique(labels[0], return_index=True)
        self.curr_obs_torch['semantic_label'] = list(np.array(labels[0])[np.sort(idx)]) # list of semantic label we have
        # verfiy the assumption that the mask label is the same for all cameras
        # for i in range(self.num_cam):
        #     try:
        #         assert self.curr_obs_torch['mask_label'][i] == self.curr_obs_torch['mask_label'][0]
        #     except:
        #         print('The mask label is not the same for all cameras!')
        #         print(self.curr_obs_torch['mask_label'])
        #         for j in range(self.num_cam):
        #             for k in range(len(self.curr_obs_torch['mask_label'][j])):
        #                 plt.subplot(1, len(self.curr_obs_torch['mask_label'][j]), k+1)
        #                 plt.imshow(self.curr_obs_torch['mask'][j].detach().cpu().numpy() == k)
        #             plt.show()
        #         raise AssertionError
        # align instance mask id to the first frame
        self.align_instance_mask_v2(queries)
        self.curr_obs_torch[f'mask'] = instance2onehot(self.curr_obs_torch[f'mask'].to(torch.uint8), len(self.curr_obs_torch['consensus_mask_label'])).to(dtype=torch.float32)
    
    def get_query_obj_pcd(self):
        color = self.curr_obs_torch['color']
        depth = self.curr_obs_torch['depth'].detach().cpu().numpy()
        mask = self.curr_obs_torch['mask'].detach().cpu().numpy()
        mask = (mask[..., 1:].sum(axis=-1) > 0)
        for i in range(self.num_cam):
            mask[i] = (cv2.erode((mask[i] * 255).astype(np.uint8), np.ones([2, 2], np.uint8), iterations=1) / 255).astype(bool)
        K = self.curr_obs_torch['K'].detach().cpu().numpy()
        pose = self.curr_obs_torch['pose'].detach().cpu().numpy() # [num_cam, 3, 4]
        pad = np.tile(np.array([[[0,0,0,1]]]), [pose.shape[0], 1, 1])
        pose = np.concatenate([pose, pad], axis=1)
        pcd = aggr_point_cloud_from_data(color, depth, K, pose, downsample=False, masks=mask)
        return pcd
    
    def vis_3d(self, pts, res, params):
        # :param pts: (N, 3) torch tensor in world frame
        # :param res: dict contains:
        #             - 'dist': (N) torch tensor, dist to the closest point on the surface
        #             - 'dino_feats': (N, f) torch tensor, the features of the points
        #             - 'query_masks': (N, nq) torch tensor, the query masks of the points
        #             - 'valid_mask': (N) torch tensor, whether the point is valid
        # :param params: dict for other useful params
        
        pts = pts[res['valid_mask']].cpu().numpy()
        dist = res['dist'][res['valid_mask']].cpu().numpy()
        # visualize dist
        dist_vol = go.Figure(data=[go.Scatter3d(x=pts[:,0],
                                                y=pts[:,1],
                                                z=pts[:,2],
                                                mode='markers',
                                                marker=dict(
                                                    size=2,
                                                    color=dist,
                                                    colorscale='Viridis',
                                                    colorbar=dict(thickness=20, ticklen=4),))],
                             layout=go.Layout(scene=dict(aspectmode='data'),))
        dist_vol.show()
        
        # # visualize features
        # features = res['dino_feats'][res['valid_mask']].cpu().numpy()
        # pca = params['pca']
        # features_pca = pca.transform(features)
        # for i in range(features_pca.shape[1]):
        #     features_pca[:, i] = (features_pca[:, i] - features_pca[:, i].min()) / (features_pca[:, i].max() - features_pca[:, i].min())
        # features_pca = (features_pca * 255).astype(np.uint8)
        # colors = []
        # for i in range(0, features_pca.shape[0], 1):
        #     colors.append(f'rgb({features_pca[i, 0]}, {features_pca[i, 1]}, {features_pca[i, 2]})')
        # features_vol = go.Figure(data=[go.Scatter3d(x=pts[:,0],
        #                                             y=pts[:,1],
        #                                             z=pts[:,2],
        #                                             mode='markers',
        #                                             marker=dict(
        #                                                 size=2,
        #                                                 color=colors,
        #                                                 colorbar=dict(thickness=20, ticklen=4),))],
        #                          layout=go.Layout(scene=dict(aspectmode='data'),))
        # features_vol.show()
        
        # # visualize masks
        # query_masks = res['query_masks'][res['valid_mask']].cpu().numpy()
        # NQ = res['query_masks'].shape[-1]
        # for i in range(NQ):
        #     mask_vol = go.Figure(data=[go.Scatter3d(x=pts[:,0],
        #                                             y=pts[:,1],
        #                                             z=pts[:,2],
        #                                             mode='markers',
        #                                             marker=dict(
        #                                                 size=2,
        #                                                 color=query_masks[:,i],
        #                                                 colorscale='Viridis',
        #                                                 colorbar=dict(thickness=20, ticklen=4),))],
        #                          layout=go.Layout(scene=dict(aspectmode='data'),))
        #     mask_vol.show()
    
    # visualize 3d field
    def interactive_corr(self, src_info, tgt_info, pts, res):
        # :param src_info: dict contains:
        #                  - 'color': (H, W, 3) np array, color image
        #                  - 'dino_feats': (H, W, f) torch tensor, dino features
        # :param tgt_info: dict contains:
        #                  - 'color': (K, H, W, 3) np array, color image
        #                  - 'pose': (K, 3, 4) torch tensor, pose of the camera
        #                  - 'K': (K, 3, 3) torch tensor, camera intrinsics
        # :param pts: (N, 3) torch tensor in world frame
        # :param res: dict contains:
        #             - 'dist': (N) torch tensor, dist to the closest point on the surface
        #             - 'dino_feats': (N, f) torch tensor, the features of the points
        #             - 'query_masks': (N, nq) torch tensor, the query masks of the points
        #             - 'valid_mask': (N) torch tensor, whether the point is valid
        num_tgt = len(tgt_info['color'])
        sim_scale = 0.05
        imshow_scale = 0.6
        
        viridis_cmap = cm.get_cmap('viridis')
        
        def drawHeatmap(event, x, y, flags, param):
            if event == cv2.EVENT_MOUSEMOVE:
                src_color_render_curr = draw_keypoints(src_info['color'], np.array([[x, y]]), colors=[(255, 0, 0)], radius=5)
                cv2.imshow('src', src_color_render_curr[..., ::-1])
                src_feat_tensor = src_info['dino_feats'][y, x]
                tgt_feat_sims_tensor = compute_similarity_tensor(res['dino_feats'], src_feat_tensor, scale=sim_scale, dist_type='l2') # [N]
                tgt_feat_sims_tensor = (tgt_feat_sims_tensor - tgt_feat_sims_tensor.min()) / (tgt_feat_sims_tensor.max() - tgt_feat_sims_tensor.min()) # [N]
                tgt_feat_sims = tgt_feat_sims_tensor.detach().cpu().numpy()
                tgt_feat_sim_imgs = (viridis_cmap(tgt_feat_sims)[..., :3] * 255)[..., ::-1]
                
                pts_2d, _, _ = project_points_coords(pts, tgt_info['pose'], tgt_info['K']) # [N, 2]
                pts_2d = pts_2d.detach().cpu().numpy()
                pts_2d = pts_2d[..., ::-1]
                
                max_sim_idx = tgt_feat_sims_tensor.argmax()
                match_pt_3d = pts[max_sim_idx][None]
                match_pt_2d, _, _ = project_points_coords(match_pt_3d, tgt_info['pose'], tgt_info['K'])
                match_pt_2d = match_pt_2d.detach().cpu().numpy()
                
                merge_imgs = []
                
                for idx in range(num_tgt):
                    heatmap = np.zeros_like(tgt_info['color'][idx])
                    heatmap = heatmap.reshape(self.H * self.W, 3)
                    pts_2d_i = pts_2d[idx].astype(np.int32)
                    valid_pts = (pts_2d_i[:, 0] >= 0) & (pts_2d_i[:, 0] < self.W) & (pts_2d_i[:, 1] >= 0) & (pts_2d_i[:, 1] < self.H)
                    pts_2d_flat_idx = np.ravel_multi_index(pts_2d_i[valid_pts].T, (self.H, self.W))
                    heatmap[pts_2d_flat_idx] = tgt_feat_sim_imgs[valid_pts]
                    heatmap = heatmap.reshape(self.H, self.W, 3).astype(np.uint8)
                    cv2.imshow(f'tgt_heatmap_{idx}', heatmap)
                    
                    tgt_imshow_curr = draw_keypoints(tgt_info['color'][idx], match_pt_2d[idx], colors=[(255, 0, 0)], radius=5)
                    # cv2.imshow(f'tgt_{idx}', tgt_imshow_curr[..., ::-1])
                    
                    merge_img = np.concatenate([heatmap, tgt_imshow_curr[..., ::-1]], axis=1)
                    merge_imgs.append(merge_img)
                    
                merge_imgs = np.concatenate(merge_imgs, axis=0)
                merge_imgs = cv2.resize(merge_imgs, (int(merge_imgs.shape[1] * imshow_scale), int(merge_imgs.shape[0] * imshow_scale)), interpolation=cv2.INTER_NEAREST)
                cv2.imshow('merge', merge_imgs)
                
        cv2.imshow('src', src_info['color'][..., ::-1])
        # for idx in range(num_tgt):
        #     cv2.imshow(f'tgt_{idx}', tgt_info['color'][idx][..., ::-1])
        
        cv2.setMouseCallback('src', drawHeatmap)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def interactive_corr_img(self, src_info, tgt_info):
        src_img = src_info['color'][..., ::-1]
        src_dino = src_info['dino_feats']
        
        tgt_img = tgt_info['color'][..., ::-1]
        tgt_dino = tgt_info['dino_feats'][None].permute(0, 3, 1, 2)
        
        viridis_cmap = cm.get_cmap('viridis')
        
        def drawHeatmap(event, x, y, flags, param):
            # feat_x = int(x / src_img.shape[1] * src_dino.shape[1])
            # feat_y = int(y / src_img.shape[0] * src_dino.shape[0])
            if event == cv2.EVENT_MOUSEMOVE:
                src_img_curr = draw_keypoints(src_img, np.array([[x, y]]), colors=[(255, 0, 0)], radius=5)
                cv2.imshow('src', src_img_curr)
                src_feat = src_dino[y, x]
                tgt_feat_sim = compute_similarity_tensor(tgt_dino, src_feat, scale=0.5, dist_type='l2')[0].detach().cpu().numpy()
                tgt_feat_sim = (tgt_feat_sim - tgt_feat_sim.min()) / (tgt_feat_sim.max() - tgt_feat_sim.min())
                tgt_feat_sim_img = (viridis_cmap(tgt_feat_sim)[..., :3] * 255)[..., ::-1]
                # tgt_feat_sim_img = cv2.resize(tgt_feat_sim_img, (tgt_img.shape[1], tgt_img.shape[0]), interpolation=cv2.INTER_NEAREST)
                tgt_color_render_curr = (tgt_feat_sim_img * tgt_feat_sim[..., None] + tgt_img * (1.0 - tgt_feat_sim[..., None])).astype(np.uint8)
                cv2.imshow(f'tgt', tgt_color_render_curr)
                cv2.imshow(f'heatmap', tgt_feat_sim_img.astype(np.uint8))
                # cv2.imshow(f'tgt_{idx}', tgt_feat_sim_img.astype(np.uint8))
        
        cv2.imshow('src', src_img)
        cv2.imshow('tgt', tgt_img)

        cv2.setMouseCallback('src', drawHeatmap)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def extract_mesh(self, pts, res, grid_shape):
        # :param pts: (N, 3) torch tensor in world frame
        # :param res: dict contains:
        #             - 'dist': (N) torch tensor, dist to the closest point on the surface
        #             - 'dino_feats': (N, f) torch tensor, the features of the points
        #             - 'query_masks': (N, nq) torch tensor, the query masks of the points
        #             - 'valid_mask': (N) torch tensor, whether the point is valid
        # :param grid_shape: (3) tuple, the shape of the grid
        dist = res['dist'].detach().cpu().numpy()
        dist = dist.reshape(grid_shape)
        smoothed_dist = mcubes.smooth(dist)
        vertices, triangles = mcubes.marching_cubes(smoothed_dist, 0) # 3d index of valid pts closet to the surface
        vertices = vertices.astype(np.int32)
        # vertices_flat = np.unravel_index(vertices, grid_shape)
        vertices_flat = np.ravel_multi_index(vertices.T, grid_shape) # flat index of valid pts cloest to the surface
        vertices_coords = pts.detach().cpu().numpy()[vertices_flat] # coordinates of valid pts
        
        return vertices_coords, triangles
    
    def extract_meshes(self, pts, res, grid_shape):
        # :param pts: (N, 3) torch tensor in world frame
        # :param res: dict contains:
        #             - 'dist': (N) torch tensor, dist to the closest point on the surface
        #             - 'mask': (N, nq) torch tensor, the query masks of the points
        #             - 'valid_mask': (N) torch tensor, whether the point is valid
        # :param grid_shape: (3) tuple, the shape of the grid
        vertices_list = []
        triangles_list = []
        mask_label = self.curr_obs_torch['mask_label'][0]
        num_instance = len(mask_label)
        
        mask = res['mask']
        mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-7) # (N, num_instances)
        for i in range(num_instance):
            mask_i = mask[:, i] > 0.6
            dist = res['dist'].clone()
            dist[~mask_i] = 1e3
            dist = dist.detach().cpu().numpy()
            dist = dist.reshape(grid_shape)
            smoothed_dist = mcubes.smooth(dist)
            vertices, triangles = mcubes.marching_cubes(smoothed_dist, 0)
            vertices = vertices.astype(np.int32)
            # vertices_flat = np.unravel_index(vertices, grid_shape)
            vertices_flat = np.ravel_multi_index(vertices.T, grid_shape)
            vertices_coords = pts.detach().cpu().numpy()[vertices_flat]
            
            vertices_list.append(vertices_coords)
            triangles_list.append(triangles)
        
        return vertices_list, triangles_list

    def create_mask_mesh(self, vertices, triangles, res):
        # :param vertices: (N, 3) numpy array, the vertices of the mesh
        # :param triangles: (M, 3) numpy array, the triangles of the mesh
        # :param res: dict contains:
        #             - 'dist': (N) torch tensor, dist to the closest point on the surface
        #             - 'dino_feats': (N, f) torch tensor, the features of the points
        #             - 'query_masks': (N, nq) torch tensor, the query masks of the points
        #             - 'valid_mask': (N) torch tensor, whether the point is valid
        query_masks = res['query_masks'].detach().cpu().numpy()
        mask_meshes = []
        for i in range(query_masks.shape[1]):
            vertices_color = trimesh.visual.interpolate(query_masks[:,i], color_map='viridis')
            mask_meshes.append(trimesh.Trimesh(vertices=vertices, faces=triangles[..., ::-1], vertex_colors=vertices_color))
        return mask_meshes

    def create_instance_mask_mesh(self, vertices, triangles, res):
        # :param vertices: (N, 3) numpy array, the vertices of the mesh
        # :param triangles: (M, 3) numpy array, the triangles of the mesh
        # :param res: dict contains:
        #             - 'dist': (N) torch tensor, dist to the closest point on the surface
        #             - 'dino_feats': (N, f) torch tensor, the features of the points
        #             - 'mask_*': (N, nq) torch tensor, the instance masks of the points
        #             - 'valid_mask': (N) torch tensor, whether the point is valid
        mask_meshes = []
        colors = []
        for k in res.keys():
            if k.startswith('mask'):
                mask = res[k].detach().cpu().numpy()
                num_instance = mask.shape[1]
                mask = onehot2instance(mask) # (N, nq) -> (N)
                
                # mask_vis = np.zeros((mask.shape[0], 3))
                # mask_vis[mask == 0] = np.array(series_RGB[5])
                # mask_vis[mask == 1] = np.array(series_RGB[1])
                # mask_vis[mask == 2] = np.array(series_RGB[2])
                # mask_vis[mask == 3] = np.array(series_RGB[4])
                # mask_vis[mask == 4] = np.array(series_RGB[6])
                
                # mask_vis = np.concatenate([mask_vis, np.ones((mask_vis.shape[0], 1)) * 255], axis=1).astype(np.uint8)
                
                vertices_color = trimesh.visual.interpolate(mask / num_instance, color_map='jet')
                mask_meshes.append(trimesh.Trimesh(vertices=vertices, faces=triangles[..., ::-1], vertex_colors=vertices_color))
                colors.append(vertices_color)
        return mask_meshes, colors

    def apply_pca_colormap_return_proj(self, features, target_dim=3, proj_V=None, normalize=False, low_rank_min=None, low_rank_max=None, niter=5):
        """Convert a multichannel image to color using PCA.

        Args:
            image: Multichannel image.
            proj_V: Projection matrix to use. If None, use torch low rank PCA.

        Returns:
            Colored PCA image of the multichannel input image.
        """
        features_flat = features.reshape(-1, features.shape[-1])

        # Modified from https://github.com/pfnet-research/distilled-feature-fields/blob/master/train.py
        if proj_V is None:
            mean = features_flat.mean(0)
            with torch.no_grad():
                U, S, V = torch.pca_lowrank(features_flat - mean, niter=niter, q=target_dim) # default q = 6
            proj_V = V[:, :target_dim] # default shape of V: [feature.shape[-1], 6]
        low_rank = features_flat @ proj_V

        # !!! Note that normalization is done in rgb visulization, if using the feat, please open the normalization here !!!
        # to avoid get zero for the min, and cut the max which is out of one
        if normalize:
            if low_rank_min is None:
                low_rank_min = torch.quantile(low_rank, 0.01, dim=0)
            if low_rank_max is None:
                low_rank_max = torch.quantile(low_rank, 0.99, dim=0)
            low_rank = (low_rank - low_rank_min) / (low_rank_max - low_rank_min)
            low_rank = torch.clamp(low_rank, 0, 1)

        features_pca = low_rank #.reshape(features.shape[:-1] + (3,))
        return features_pca, proj_V

    def create_descriptor_pcd(self, pts, res, mask_out_bg, decriptor_type, visualize):
        features = res[decriptor_type].detach().cpu().numpy()

        if mask_out_bg:
            mask = res['mask'].detach().cpu().numpy()
            mask = onehot2instance(mask) # (N, nq) -> (N)
            bg = (mask == 0)
            # use pca, mask out bg is supposed to be true
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            features_pca_fg = pca.fit_transform(features[~bg])
            features_pca = np.zeros((features.shape[0], 3))
            features_pca[~bg] = features_pca_fg

            # features_pca_fg = self.apply_pca_colormap_return_proj(res[decriptor_type][~bg])
            # features_pca = np.zeros((features.shape[0], 3))
            # features_pca[~bg] = features_pca_fg.detach().cpu().numpy()

        else:
            if decriptor_type == 'clip_sims':
                # for sim, without pca
                # for grey color
                features_pca = features[..., :3]
                
                # mark the maximum point of KNN
                from sklearn.neighbors import NearestNeighbors
                M = 500
                k = int(0.05 * M)
                nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(pts)
                distances, indices = nbrs.kneighbors(pts)
                clip_sims = features_pca
                average_affordances = np.array([clip_sims[idx].mean() for idx in indices])
                max_ind = average_affordances.argmax()
                
                # for cmap color
                # features_pca = features[..., 0]
                # cmap = plt.get_cmap('viridis')
                # cmap = plt.get_cmap('YlGnBu')
                # features_pca = cmap(features_pca)
                # features_pca = features_pca[..., :3]
                
            else:
                # for feature, with pca
                features_pca, _ = self.apply_pca_colormap_return_proj(res[decriptor_type])
                features_pca = features_pca.detach().cpu().numpy()

                # use pca, mask out bg is false
                # from sklearn.decomposition import PCA
                # pca = PCA(n_components=3)
                # features_pca = pca.fit_transform(features)
                # use feature of first 3 dimensions
                # features_pca = features[..., :3]

        features_rgb = np.zeros((features_pca.shape[0], 3))
        
        for i in range(features_pca.shape[1]):
            if not mask_out_bg:
                features_rgb[:, i] = (features_pca[:, i] - features_pca[:, i].min()) / (features_pca[:, i].max() - features_pca[:, i].min())
            else:
                features_rgb[~bg, i] = (features_pca[~bg, i] - features_pca[~bg, i].min()) / (features_pca[~bg, i].max() - features_pca[~bg, i].min())
                # features_rgb[~bg, i] = (features_pca_fg[:, i] - features_pca_fg[:, i].min()) / (features_pca_fg[:, i].max() - features_pca_fg[:, i].min())
                # features_rgb[bg, i] = 0.8
        if mask_out_bg:
            features_rgb[bg] = np.ones(3) * 0.8
            
        if decriptor_type == 'clip_sims':
            features_rgb[max_ind, :] = np.array([1, 0, 0])
            
        # TODO why transpose rgb to bgr???
        # features_rgb = features_rgb[..., ::-1]

        # for cmap color
        # cmap = plt.get_cmap('viridis')
        # cmap = plt.get_cmap('YlGnBu')
        # features_rgb = features_rgb[..., 0]
        # features_rgb = cmap(features_rgb)
        # features_rgb = features_rgb[..., :3]
        
        feature_pcd = o3d.geometry.PointCloud()
        feature_pcd.points = o3d.utility.Vector3dVector(pts)
        feature_pcd.colors = o3d.utility.Vector3dVector(features_rgb)
        if visualize:
            o3d.visualization.draw_geometries([feature_pcd])

        return feature_pcd      

    def create_descriptor_mesh(self, vertices, triangles, res, mask_out_bg, decriptor_type):
        features = res[decriptor_type].detach().cpu().numpy()

        if mask_out_bg:
            mask = res['mask'].detach().cpu().numpy()
            mask = onehot2instance(mask) # (N, nq) -> (N)
            bg = (mask == 0)
            # use pca, mask out bg should be true
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            features_pca_fg = pca.fit_transform(features[~bg])
            features_pca = np.zeros((features.shape[0], 3))
            features_pca[~bg] = features_pca_fg

            # for dino and clip feature, with pca
            # !!! dino pca for improve the performance, clip pca for visualize !!!
            # features_pca_fg = self.apply_pca_colormap_return_proj(res[decriptor_type][~bg])
            # features_pca = np.zeros((features.shape[0], 3))
            # features_pca[~bg] = features_pca_fg.detach().cpu().numpy()

        else:
            # for sim, without pca
            if decriptor_type == 'clip_sims':
                features_pca = features[..., :3]
            else:
                features_pca = features[..., :3]
                # use pca, mask out bg is false
                # from sklearn.decomposition import PCA
                # pca = PCA(n_components=3)
                # features_pca = pca.fit_transform(features)

        features_rgb = np.zeros((features_pca.shape[0], 3))
        
        for i in range(features_pca.shape[1]):
            if not mask_out_bg:
                features_rgb[:, i] = (features_pca[:, i] - features_pca[:, i].min()) / (features_pca[:, i].max() - features_pca[:, i].min())
            
            else:
                features_rgb[~bg, i] = (features_pca[~bg, i] - features_pca[~bg, i].min()) / (features_pca[~bg, i].max() - features_pca[~bg, i].min())
                # features_rgb[~bg, i] = (features_pca_fg[:, i] - features_pca_fg[:, i].min()) / (features_pca_fg[:, i].max() - features_pca_fg[:, i].min())
                # features_rgb[bg, i] = 0.8
        if mask_out_bg:
            features_rgb[bg] = np.ones(3) * 0.8
        features_rgb = features_rgb[..., ::-1]
            
        features_rgb = (features_rgb * 255).astype(np.uint8)
        features_rgb = np.concatenate([features_rgb, np.ones((features_rgb.shape[0], 1), dtype=np.uint8) * 255], axis=1)
        features_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles[..., ::-1], vertex_colors=features_rgb)
        return features_mesh, features_rgb

    def create_color_mesh(self, vertices, triangles, res):
        colors = res['color_tensor'].detach().cpu().numpy()
        colors = (colors * 255).astype(np.uint8)
        colors = np.concatenate([colors, np.ones((colors.shape[0], 1), dtype=np.uint8) * 255], axis=1)
        color_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles[..., ::-1], vertex_colors=colors)
        return color_mesh, colors
    
    def select_features(self, color, params):
        H, W = color.shape[:2]
        features = self.extract_features(color[None], params)
        features = F.interpolate(features.permute(0, 3, 1, 2),
                                 size=color.shape[:2],
                                 mode='bilinear',
                                 align_corners=True).permute(0, 2, 3, 1)[0]
        
        param = {'src_pts': []} # (col, row)
        
        color_cands = [(180,119,31),
                       (14,127,255),
                       (44,160,44),
                       (40,39,214),
                       (189,103,148),]
        
        def select_kypts(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                param['src_pts'].append([x, y])
                color_curr = draw_keypoints(color, np.array([[x, y]]), colors=[(0, 0, 255)], radius=5)
                cv2.imshow('src', color_curr[..., ::-1])
            elif event == cv2.EVENT_MOUSEMOVE:
                color_curr = draw_keypoints(color, np.array([[x, y]]), colors=[(255, 0, 0)], radius=5)
                for kp in param['src_pts']:
                    color_curr = draw_keypoints(color_curr, np.array([kp]), colors=[(0, 0, 255)], radius=5)
                cv2.imshow('src', color_curr[..., ::-1])
        
        cv2.imshow('src', color[..., ::-1])
        cv2.setMouseCallback('src', select_kypts, param)
        
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
        
        src_feats = [features[p[1], p[0]] for p in param['src_pts']]
        src_feats = torch.stack(src_feats)
        
        N = src_feats.shape[0]
        try:
            assert N <= len(color_cands)
            colors = color_cands[:N]
        except:
            print('not enough color candidates')
            return
        color_with_kypts = color.copy()
        color_with_kypts = draw_keypoints(color_with_kypts, np.array(param['src_pts']), colors=colors, radius=int(5 * H / 360))
        
        return src_feats, color_with_kypts
    
    def select_features_rand(self, boundaries, N, per_instance=False, res = None, init_idx = -1):
        # randomly select N features for object {query_text} in 3D space 
        res = 0.001 if res is None else res
        dist_threshold = 0.005
        
        grid, grid_shape = create_init_grid(boundaries, res)
        grid = grid.to(self.device, dtype=torch.float32)
        
        # out = self.eval(init_grid)
        with torch.no_grad():
            out = self.batch_eval(grid, return_names=['mask'])
        
        dist_mask = torch.abs(out['dist']) < dist_threshold
        
        label = self.curr_obs_torch['consensus_mask_label']
        
        last_label = label[0]
        
        src_feats_list = []
        img_list = []
        src_pts_list = []
        mask = out['mask'] # (N, num_instances) where 0 is background
        mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-7) # (N, num_instances)
        for i in range(1, len(label)):
            if label[i] == last_label and not per_instance:
                continue
            instance_mask = mask[:, i] > 0.6
            masked_pts = grid[instance_mask & dist_mask & out['valid_mask']]
            
            sample_pts, sample_idx, _ = fps_np(masked_pts.detach().cpu().numpy(), N, init_idx=init_idx)
            # src_feats_list.append(out['dino_feats'][sample_idx])
            src_feats_list.append(self.eval(torch.from_numpy(sample_pts).to(self.device, torch.float32))['dino_feats'])
            src_pts_list.append(sample_pts)
            
            num_pts = sample_pts.shape[0]
            pose = self.curr_obs_torch['pose'][0].detach().cpu().numpy()
            K = self.curr_obs_torch['K'][0].detach().cpu().numpy()
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            img = self.curr_obs_torch['color'][0]
            
            cmap = cm.get_cmap('viridis')
            colors = ((cmap(np.linspace(0, 1, num_pts))[:, :3]) * 255).astype(np.int32)
            
            sample_pts = np.concatenate([sample_pts, np.ones([num_pts, 1])], axis=-1) # [num_pts, 4]
            sample_pts = np.matmul(pose, sample_pts.T)[:3].T # [num_pts, 3]
            
            sample_pts_2d = sample_pts[:, :2] / sample_pts[:, 2:] # [num_pts, 2]
            sample_pts_2d[:, 0] = sample_pts_2d[:, 0] * fx + cx
            sample_pts_2d[:, 1] = sample_pts_2d[:, 1] * fy + cy
            
            sample_pts_2d = sample_pts_2d.astype(np.int32)
            sample_pts_2d = sample_pts_2d.reshape(num_pts, 2)
            img = draw_keypoints(img, sample_pts_2d, colors, radius=5)
            img_list.append(img)
            last_label = label[i]
        
        del out
        return src_feats_list, src_pts_list, img_list
    
    def select_features_rand_v2(self, boundaries, N, per_instance=False):
        N_per_cam = N // self.num_cam
        src_feats_list = []
        img_list = []
        src_pts_list = []
        label = self.curr_obs_torch['mask_label'][0]
        last_label = label[0]
        for i in range(1, len(label)):
            if label[i] == last_label and not per_instance:
                continue
            src_pts_np = []
            for cam_i in range(self.num_cam):
                instance_mask = (self.curr_obs_torch['mask'][cam_i, :, :, i]).detach().cpu().numpy().astype(bool)
                depth_i = self.curr_obs_torch['depth'][cam_i].detach().cpu().numpy()
                K_i = self.curr_obs_torch['K'][cam_i].detach().cpu().numpy()
                pose_i = self.curr_obs_torch['pose'][cam_i].detach().cpu().numpy()
                pose_i = np.concatenate([pose_i, np.array([[0, 0, 0, 1]])], axis=0)
                valid_depth = (depth_i > 0.0) & (depth_i < 1.5)
                instance_mask = instance_mask & valid_depth
                instance_mask = (instance_mask * 255).astype(np.uint8)
                # plt.subplot(1, 2, 1)
                # plt.imshow(instance_mask)
                instance_mask = cv2.erode(instance_mask, np.ones([15, 15], np.uint8), iterations=1)
                # plt.subplot(1, 2, 2)
                # plt.imshow(instance_mask)
                # plt.show()
                instance_mask_idx = np.array(instance_mask.nonzero()).T # (num_pts, 2)
                sel_idx, _, _ = fps_np(instance_mask_idx, N_per_cam)
                
                sel_depth = depth_i[sel_idx[:, 0], sel_idx[:, 1]]
                
                src_pts = np.zeros([N_per_cam, 3])
                src_pts[:, 0] = (sel_idx[:, 1] - K_i[0, 2]) * sel_depth / K_i[0, 0]
                src_pts[:, 1] = (sel_idx[:, 0] - K_i[1, 2]) * sel_depth / K_i[1, 1]
                src_pts[:, 2] = sel_depth
                
                # sample_pts = np.concatenate([sample_pts, np.ones([N, 1])], axis=-1) # [num_pts, 4]
                # sample_pts = np.matmul(pose, sample_pts.T)[:3].T # [num_pts, 3] # world to camera
                
                src_pts = np.matmul(np.linalg.inv(pose_i), np.concatenate([src_pts, np.ones([N_per_cam, 1])], axis=-1).T)[:3].T # [num_pts, 3] # camera to world
                
                src_pts_np.append(src_pts)
            sample_pts = np.concatenate(src_pts_np, axis=0)
            src_pts_list.append(sample_pts)
            src_feats_list.append(self.eval(torch.from_numpy(sample_pts).to(self.device, torch.float32))['dino_feats'])
            
            cmap = cm.get_cmap('jet')
            colors = ((cmap(np.linspace(0, 1, N))[:, :3]) * 255).astype(np.int32)
            
            pose = self.curr_obs_torch['pose'][0].detach().cpu().numpy()
            K = self.curr_obs_torch['K'][0].detach().cpu().numpy()
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            img = self.curr_obs_torch['color'][0]
            
            sample_pts = np.concatenate([sample_pts, np.ones([N, 1])], axis=-1) # [num_pts, 4]
            sample_pts = np.matmul(pose, sample_pts.T)[:3].T # [num_pts, 3]
            
            sample_pts_2d = sample_pts[:, :2] / sample_pts[:, 2:] # [num_pts, 2]
            sample_pts_2d[:, 0] = sample_pts_2d[:, 0] * fx + cx
            sample_pts_2d[:, 1] = sample_pts_2d[:, 1] * fy + cy
            
            sample_pts_2d = sample_pts_2d.astype(np.int32)
            sample_pts_2d = sample_pts_2d.reshape(N, 2)
            img = draw_keypoints(img, sample_pts_2d, colors, radius=5)
            img_list.append(img)
            last_label = label[i]

        return src_feats_list, src_pts_list, img_list
    
    def compute_conf(self, matched_pts, tgt_feats, conf_sigma):
        # :param matched_pts: (K, 3) torch tensor
        # :param tgt_feats: (K, f) torch tensor
        # :return conf: (K, ) torch tensor
        # matched_pts_eval = self.eval(matched_pts, return_names=['dino_feats'], return_inter=True)
        matched_pts_eval = self.eval(matched_pts, return_names=['dino_feats'])
        feat_dist = torch.norm(matched_pts_eval['dino_feats'] - tgt_feats, dim=1) # (K, )
        # inter_feat = matched_pts_eval['dino_feats_inter'] # (num_view, K, f)
        # inter_feat_dist = torch.norm(inter_feat - tgt_feats[None], dim=2) # (num_view, K)
        # inter_feat_conf = torch.exp(-inter_feat_dist / conf_sigma) # (num_view, K)
        # for i in range(inter_feat_conf.shape[0]):
        #     print(f'conf in view {i}: {inter_feat_conf[i,0].item()}')
        conf = torch.exp(-feat_dist / conf_sigma) * torch.exp(-torch.abs(matched_pts_eval['dist']) * 50) * matched_pts_eval['valid_mask'] # (K, )
        return conf
    
    def find_correspondences_with_mask(self,
                                       src_feat_info,
                                       pts,
                                       last_match_pts_list,
                                       res,
                                       debug=False,
                                       debug_info=None):
        mask = res['mask'] # (N, num_instances) where 0 is background
        mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-7) # (N, num_instances)
        # mask_onehot = torch.argmax(mask, dim=1).to(torch.uint8) # (N,)
        # mask = instance2onehot(mask_onehot) # (N, num_instances)
        num_instances = mask.shape[1]
        match_pts_list = []
        conf_list = []
        for i in range(1, num_instances):
            src_feats = src_feat_info[self.curr_obs_torch['mask_label'][0][i]]['src_feats']
            if last_match_pts_list is None:
                last_match_pts = None
            else:
                last_match_pts = torch.from_numpy(last_match_pts_list[i - 1]).to(self.device, dtype=torch.float32)
            
            # instance_mask = mask[:, i]
            instance_mask = mask[:, i] > 0.6
            tgt_feats = res['dino_feats'][res['valid_mask'] & instance_mask] # (N', f)
            pts_i = pts[res['valid_mask'] & instance_mask] # (N', 3)
            # tgt_feats = res['dino_feats'][res['valid_mask']] # (N', f)
            # pts_i = pts[res['valid_mask']] # (N', 3)
            instance_mask = instance_mask[res['valid_mask']] # (N', )
            
            if debug:
                pts_i_np = pts_i.cpu().numpy()
                full_pts = debug_info['full_pts']
                full_mask = debug_info['full_mask']
                full_pts_1 = full_pts[full_mask[:, 1].astype(bool)]
                full_pts_2 = full_pts[full_mask[:, 2].astype(bool)]
                vol = go.Figure(data=[go.Scatter3d(x=pts_i_np[:,0],
                                                        y=pts_i_np[:,1],
                                                        z=pts_i_np[:,2],
                                                        mode='markers',
                                                        marker=dict(
                                                            size=2,
                                                            colorscale='Viridis',
                                                            colorbar=dict(thickness=20, ticklen=4),)),
                                      go.Scatter3d(x=full_pts_1[:,0],
                                                   y=full_pts_1[:,1],
                                                   z=full_pts_1[:,2],
                                                   mode='markers',
                                                   marker=dict(
                                                       size=2,
                                                       colorscale='Viridis',
                                                       colorbar=dict(thickness=20, ticklen=4),)),
                                      go.Scatter3d(x=full_pts_2[:,0],
                                                   y=full_pts_2[:,1],
                                                   z=full_pts_2[:,2],
                                                   mode='markers',
                                                   marker=dict(
                                                       size=2,
                                                       colorscale='Viridis',
                                                       colorbar=dict(thickness=20, ticklen=4),))],
                                        layout=go.Layout(scene=dict(aspectmode='data'),))
                vol.show()
            
            sim_tensor = compute_similarity_tensor_multi(tgt_feats,
                                                        src_feats,
                                                        pts_i,
                                                        last_match_pts,
                                                        scale = 0.5,
                                                        dist_type='l2') # (N', K)
            # sim_tensor = sim_tensor * torch.clamp_min(torch.log(instance_mask[:, None] + 1e-7) + 1, 0) # (N', K)
            match_pts = extract_kypts_gpu(sim_tensor, pts_i, match_metric='sum') # (K, 3)
            # print('stddev of x: ', pts_i[:,0].std().item())
            # print('stddev of y: ', pts_i[:,1].std().item())
            # print('stddev of z: ', pts_i[:,2].std().item())
            # print('stddev:', pts_i.std(dim=0).norm().item())
            topk_sim_idx = torch.topk(sim_tensor, k=1000, dim=0)[1] # (100, K)
            topk_pts = pts_i[topk_sim_idx] # (100, K, 3)
            observability = self.compute_conf(match_pts, src_feats, conf_sigma=1000.) # (K, )
            stability = torch.exp(-topk_pts.std(dim=0).norm(dim=1) * 20.0) # (K, )
            # print('stability:', stability)
            # print('observability:', observability)
            conf = observability * stability
            match_pts_list.append(match_pts.detach().cpu().numpy())
            conf_list.append(conf.detach().cpu().numpy())
        # match_pts = torch.stack(match_pts_list, dim=0) # (num_instances - 1, K, 3)
        return match_pts_list, conf_list
    
    def find_correspondences(self,
                             src_feats,
                             boundaries,
                             instance_id,):
        # :param src_feats torch.Tensor (K, f)
        curr_res = 0.01
        
        curr_que_pts, grid_shape = create_init_grid(boundaries, curr_res)
        curr_que_pts = curr_que_pts.to(self.device, dtype=torch.float32)
        
        out = self.eval(curr_que_pts, return_names=['dinov2_feats', 'mask'])
        # out = self.eval(curr_que_pts, return_names=['dinov2_feats'])
        
        for i in range(3):
            # multi-instance tracking
            mask = out['mask'] # (N, num_instances) where 0 is background
            mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-7) # (N, num_instances)
            # mask_onehot = torch.argmax(mask, dim=1).to(torch.uint8) # (N,)
            # mask = instance2onehot(mask_onehot) # (N, num_instances)
            curr_que_pts_list = []
            mask_instance = mask[:, instance_id] > 0.6 # (N,)
            try:
                assert mask_instance.max() > 0
            except:
                print('no instance found!')
                exit()
            tgt_feats = out['dinov2_feats'][out['valid_mask'] & mask_instance] # (N', f)
            curr_valid_que_pts = curr_que_pts[out['valid_mask'] & mask_instance] # (N', 3)
            assert tgt_feats.shape[0] == curr_valid_que_pts.shape[0]
            sim_vol = compute_similarity_tensor_multi(tgt_feats,
                                                    src_feats,
                                                    None,
                                                    None,
                                                    scale = 0.5,
                                                    dist_type='l2') # (N', K)
            next_que_pts, next_res = octree_subsample(sim_vol,
                                                    curr_valid_que_pts,
                                                    curr_res,
                                                    topK=200)
            curr_que_pts_list.append(next_que_pts)
            curr_res = next_res
            del curr_que_pts
            curr_que_pts = torch.cat(curr_que_pts_list, dim=0)
            out_keys = list(out.keys())
            for k in out_keys:
                del out[k]
            del out
            out = self.batch_eval(curr_que_pts, return_names=['dinov2_feats', 'mask'])
        # src_feat_info = {
        #     self.curr_obs_torch['mask_label'][0][instance_id]: {
        #         'src_feats': src_feats,
        #     }
        # }
        
        mask = out['mask'] # (N, num_instances) where 0 is background
        mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-7) # (N, num_instances)
        # src_feats = src_feat_info[self.curr_obs_torch['mask_label'][0][i]]['src_feats']

        instance_mask = mask[:, instance_id] > 0.6
        tgt_feats = out['dinov2_feats'][out['valid_mask'] & instance_mask] # (N', f)
        pts_i = curr_que_pts[out['valid_mask'] & instance_mask] # (N', 3)
        instance_mask = instance_mask[out['valid_mask']] # (N', )
        
        sim_tensor = compute_similarity_tensor_multi(tgt_feats,
                                                    src_feats,
                                                    None,
                                                    None,
                                                    scale = 0.5,
                                                    dist_type='l2') # (N', K)
        # sim_tensor = sim_tensor * torch.clamp_min(torch.log(instance_mask[:, None] + 1e-7) + 1, 0) # (N', K)
        match_pts = extract_kypts_gpu(sim_tensor, pts_i, match_metric='sum') # (K, 3)
        
        return match_pts

    def tracking(self,
                 src_feat_info,
                 last_match_pts_list,
                 boundaries,
                 rand_ptcl_num):
        # :param src_feat_info dict
        # :param last_match_pts_list list of [rand_ptcl_num, 3] np.array, could be None if no previous match
        
        # debug = (sel_time >= 99999)
        # if debug:
        #     curr_res = 0.002
            
        #     init_grid, grid_shape = create_init_grid(boundaries, curr_res)
        #     init_grid = init_grid.to(self.device, dtype=torch.float32)
            
        #     # out = self.eval(init_grid)
        #     out = self.batch_eval(init_grid, return_names=[])
            
        #     # extract mesh
        #     vertices, triangles = self.extract_mesh(init_grid, out, grid_shape)
        #     # mcubes.export_obj(vertices, triangles, 'sphere.obj')
        
        #     mask = self.curr_obs_torch['mask'].cpu().numpy() # [num_view, H, W, NQ]
        #     mask = onehot2instance(mask) # [num_view, H, W]
        #     mask = mask / mask.max() # [num_view, H, W]
        #     num_view, H, W = mask.shape
        #     cmap = cm.get_cmap('jet')
        #     mask_vis = cmap(mask.reshape(-1)).reshape(num_view, H, W, 4)[..., :3] # [num_view, H, W, 3]
        #     mask_vis = (mask_vis * 255).astype(np.uint8)
        #     merge_vis = np.concatenate([mask_vis[i] for i in range(mask_vis.shape[0])], axis=1)
            
        #     # eval mask and feature of vertices
        #     vertices_tensor = torch.from_numpy(vertices).to(self.device, dtype=torch.float32)
        #     out = self.batch_eval(vertices_tensor, return_names=['dino_feats', 'mask'])
        #     # mask_meshes = self.create_mask_mesh(vertices, triangles, out)
        #     mask_meshes = self.create_instance_mask_mesh(vertices, triangles, out)
        #     full_mask = out['mask'].cpu().numpy() # [N, num_instances]
        #     for mask_mesh in mask_meshes:
        #         mask_mesh.show(smooth=True)
            
        #     plt.imshow(merge_vis)
        #     plt.show()
        #     # feature_mesh = self.create_descriptor_mesh(vertices, triangles, out, {'pca': pca})
        #     # feature_mesh.show(smooth=True)
        
        curr_res = 0.01
        
        curr_que_pts, grid_shape = create_init_grid(boundaries, curr_res)
        curr_que_pts = curr_que_pts.to(self.device, dtype=torch.float32)
        
        out = self.eval(curr_que_pts, return_names=['dino_feats', 'mask'])
        # out = self.eval(curr_que_pts, return_names=['dino_feats'])
        
        for i in range(3):
            
            # multi-instance tracking
            mask = out['mask'] # (N, num_instances) where 0 is background
            mask = mask / (mask.sum(dim=1, keepdim=True) + 1e-7) # (N, num_instances)
            # mask_onehot = torch.argmax(mask, dim=1).to(torch.uint8) # (N,)
            # mask = instance2onehot(mask_onehot) # (N, num_instances)
            curr_que_pts_list = []
            for instance_id in range(1, mask.shape[1]):
                src_feats = src_feat_info[self.curr_obs_torch['mask_label'][0][instance_id]]['src_feats'] # (K, f,)
                # mask_instance = mask[:, instance_id] # (N,)
                mask_instance = mask[:, instance_id] > 0.6 # (N,)
                try:
                    assert mask_instance.max() > 0
                except:
                    print('no instance found!')
                    exit()
                tgt_feats = out['dino_feats'][out['valid_mask'] & mask_instance] # (N', f)
                curr_valid_que_pts = curr_que_pts[out['valid_mask'] & mask_instance] # (N', 3)
                # tgt_feats = out['dino_feats'][out['valid_mask']] # (N', f)
                # curr_valid_que_pts = curr_que_pts[out['valid_mask']] # (N', 3)
                assert tgt_feats.shape[0] == curr_valid_que_pts.shape[0]
                # mask_instance = mask[:, instance_id][out['valid_mask']] # (N',)
                if last_match_pts_list is None:
                    last_match_pts_i = None
                else:
                    last_match_pts_i = torch.from_numpy(last_match_pts_list[instance_id - 1]).to(self.device, dtype=torch.float32) # (K, 3)
                sim_vol = compute_similarity_tensor_multi(tgt_feats,
                                                        src_feats,
                                                        curr_valid_que_pts,
                                                        last_match_pts_i,
                                                        scale = 0.5,
                                                        dist_type='l2') # (N', K)
                # sim_vol = sim_vol * mask_instance[:, None] # (N', K) # weighted using mask
                # sim_vol = sim_vol * torch.clamp_min(torch.log(mask_instance[:, None] + 1e-7) + 1, 0)
                next_que_pts, next_res = octree_subsample(sim_vol,
                                                        curr_valid_que_pts,
                                                        curr_res,
                                                        topK=1000)
                curr_que_pts_list.append(next_que_pts)
            curr_res = next_res
            del curr_que_pts
            curr_que_pts = torch.cat(curr_que_pts_list, dim=0)
            out_keys = list(out.keys())
            for k in out_keys:
                del out[k]
            del out
            out = self.batch_eval(curr_que_pts, return_names=['dino_feats', 'mask'])
            
            # DEPRECATED: single-instance tracking
            # tgt_feats = out['dino_feats'][out['valid_mask']] # (N', f)
            # curr_valid_que_pts = curr_que_pts[out['valid_mask']] # (N', 3)
            # assert tgt_feats.shape[0] == curr_valid_que_pts.shape[0]
    
            # sim_vol = compute_similarity_tensor_multi(tgt_feats[None].permute(0, 2, 1),
            #                                           src_feats,
            #                                           scale = 0.5,
            #                                           dist_type='l2')[0].permute(1, 0) # (N', K)
            # curr_que_pts, curr_res = octree_subsample(sim_vol,
            #                                           curr_valid_que_pts,
            #                                           curr_res,
            #                                           topK=1000)
            # out = self.eval(curr_que_pts, return_names=['dino_feats'])
        
        # match_pts = self.find_correspondences(src_feats, curr_que_pts, out).detach().cpu().numpy()
        # if debug:
        #     debug_info = {'full_pts': vertices,
        #                 'full_mask': full_mask,}
        # else:
        #     debug_info = None
        match_pts_list, conf_list = self.find_correspondences_with_mask(src_feat_info,
                                                                curr_que_pts,
                                                                last_match_pts_list,
                                                                out,
                                                                debug=False,
                                                                debug_info=None)
        semantic_conf_list = []
        for semantic_label in self.curr_obs_torch['semantic_label'][1:]:
            instance_indices = find_indices(self.curr_obs_torch['mask_label'][0], semantic_label)
            semantic_conf = np.zeros(rand_ptcl_num)
            for instance_idx in instance_indices:
                semantic_conf += conf_list[instance_idx - 1]
            semantic_conf /= len(instance_indices)
            semantic_conf_list.append(semantic_conf)
        
        del curr_que_pts
        out_keys = list(out.keys())
        for k in out_keys:
            del out[k]
        del out
        
        # match_pts_list, avg_conf_per_semantic
        return {'match_pts_list': match_pts_list,
                'semantic_conf_list': semantic_conf_list,
                'instance_conf_list': conf_list}

 
if __name__ == '__main__':
    torch.cuda.set_device(0)
    fusion = Fusion(num_cam=4, feat_backbone='dinov2')
    
    