# This is impelementation of iNeRF on NeRF in this repository: https://github.com/yashbhalgat/HashNeRF-pytorch

import torch
import argparse
from nerf.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf.utils import *
import os
from functools import partial
from loss import huber_loss
import torch
from nerf.network import NeRFNetwork
import sys
import json
import numpy as np
import torchvision.transforms as T
import tqdm
import imageio
import cv2
import mediapy as media
import matplotlib.pyplot as plt
from PIL import Image
import cv2
ROOT_DIR = os.getcwd()
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.path = '/home/eph133/torch-ngp-inerf/data/fox'
    parser.output = '/home/eph133/torch-ngp-inerf/output/fox'
    parser.fp16 = True
    parser.preload = False # preload all data into GPU, accelerate training but use more GPU memory
    parser.test = False # test mode
    parser.workspace = 'trial_nerf' # workspace
    parser.seed = 0
    ### training options
    parser.iters = 30000 # training iters
    parser.lr = 1e-2 # initial learning rate
    parser.ckpt = 'latest_model' #'latest'
    parser.num_rays = 4096 # num rays sampled per image for each training step
    parser.cuda_ray = True # use CUDA raymarching instead of pytorch
    parser.max_steps = 1024 # max num steps sampled per ray (only valid when using --cuda_ray)
    parser.num_steps = 512 # num steps sampled per ray (only valid when NOT using --cuda_ray)
    parser.upsample_steps = 0 # num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.update_extra_interval = 16 # iter interval to update extra status (only valid when using --cuda_ray)
    parser.max_ray_batch = 4096 # batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)
    ### network backbone options
    parser.ff = False # use fully-fused MLP
    parser.tcnn = False # use TCNN backend"

    ### dataset options
    parser.color_space = 'srgb' # Color space, supports (linear, srgb)
    # (the default value is for the fox dataset)
    parser.bound = 2 # assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.
    parser.scale = 0.33 # scale camera location into box[-bound, bound]^3")
    parser.offset = [0, 0, 0] # offset of camera location
    parser.dt_gamma = 1/128 # dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)
    parser.min_near = 0.2 # minimum near distance for camera
    parser.density_thresh = 10 #threshold for density grid to be occupied
    parser.bg_radius = -1 #if positive, use a background model at sphere(bg_radius)

    ### GUI options
    parser.gui = False # start a GUI
    parser.W = 1920 # GUI width
    parser.H = 1080 # GUI height")
    parser.radius = 5 #default GUI camera radius from center
    parser.fovy = 50 #default GUI camera fovy
    parser.max_spp = 64 #GUI rendering max sample per pixel

    ### experimental
    parser.error_map = False # use error map to sample rays
    parser.clip_text = '' # text input for CLIP guidance
    parser.rand_pose = -1 #<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    opt = parser
    os.makedirs(opt.output, exist_ok = True)

    print(opt)

    config = {
    'input': './input/1.jpg',
    'target': './input/2.jpg',
    'output': './pose_estimation'
}
input_image = cv2.imread(config['input'], cv2.IMREAD_UNCHANGED)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
input_image = input_image.astype(np.float32) / 255
input_image = torch.from_numpy(np.stack(input_image, axis=0))
input_image = input_image.to(device=device)

target_image = cv2.imread(config['target'], cv2.IMREAD_UNCHANGED)
target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
target_image = target_image.astype(np.float32) / 255
target_image = torch.from_numpy(np.stack(target_image, axis=0))
target_image = target_image.to(device=device)

input_pose = torch.eye(4)
input_pose[2, -1] = 1.3

cam_pose = torch.clone(input_pose.detach()).unsqueeze(0)
cam_pose.requires_grad = True

intrinsics = np.array([1375.52, 1374.49, 554.558, 965.268])

save_path = os.path.join(opt.workspace, 'results')
name = 'fox'
os.makedirs(save_path, exist_ok=True)
seed_everything(opt.seed)
model = NeRFNetwork(encoding = "hashgrid",bound=opt.bound,cuda_ray = opt.cuda_ray,density_scale = 1,min_near = opt.min_near,density_thresh = opt.density_thresh,bg_radius = opt.bg_radius,)
criterion = torch.nn.MSELoss(reduction='none')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))
metrics = [PSNRMeter(),]
#model.eval()
trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, optimizer=optimizer, criterion=criterion, ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, scheduler_update_every_step=True, metrics=metrics, use_checkpoint=opt.ckpt, eval_interval=50)

with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=True):
        perturb = False 
        s = np.sqrt(opt.H * opt.W / opt.num_rays) # only in training, assert num_rays > 0
        rH, rW = int(opt.H / s), int(opt.W / s)
        rays = get_rays(cam_pose, intrinsics / s, rH, rW, -1)
   
        rays_o = rays['rays_o'] # [B, N, 3]
        rays_d = rays['rays_d'] # [B, N, 3]

        outputs = model.render(rays_o, rays_d, staged=True, bg_color=None, perturb=perturb, **vars(opt))

        preds = outputs['image'].reshape(-1, opt.H, opt.W, 3)
        preds_depth = outputs['depth'].reshape(-1, opt.H, opt.W)
    preds = linear_to_srgb(preds)

    pred = preds[0].detach().cpu().numpy()
    pred = (pred * 255).astype(np.uint8)

    pred_depth = preds_depth[0].detach().cpu().numpy()
    pred_depth = (pred_depth * 255).astype(np.uint8)

    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_rgb.png'), cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_path, f'{name}_{i:04d}_depth.png'), pred_depth)

    
