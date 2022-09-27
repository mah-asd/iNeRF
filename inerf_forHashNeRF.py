import os, sys
from datetime import datetime
import numpy as np
import imageio
import json
import pdb
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm, trange
import pickle

import matplotlib.pyplot as plt

from run_nerf_helpers import *
from optimizer import MultiOptimizer
from radam import RAdam
from loss import sigma_sparsity_loss, total_variation_loss

from load_llff import load_llff_data
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
from load_LINEMOD import load_LINEMOD_data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def batchify_rays(rays_flat, chunk=1024*32, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret


def render(H, W, K, chunk=1024*32, rays=None, c2w=None, ndc=True,
                  near=0., far=1.,
                  use_viewdirs=False, c2w_staticcam=None,
                  **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    sh = rays_d.shape # [..., 3]
    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    # Render and reshape
    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]


def render_path(render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, savedir=None, render_factor=0):

    H, W, focal = hwf

    if render_factor!=0:
        # Render downsampled for speed
        H = H//render_factor
        W = W//render_factor
        focal = focal/render_factor

    rgbs = []
    disps = []
    psnrs = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        if gt_imgs is not None and render_factor==0:
            try:
                gt_img = gt_imgs[i].cpu().numpy()
            except:
                gt_img = gt_imgs[i]
            p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_img)))
            print(p)
            psnrs.append(p)


        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)


    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)
    if gt_imgs is not None and render_factor==0:
        avg_psnr = sum(psnrs)/len(psnrs)
        print("Avg PSNR over Test set: ", avg_psnr)
        with open(os.path.join(savedir, "test_psnrs_avg{:0.2f}.pkl".format(avg_psnr)), "wb") as fp:
            pickle.dump(psnrs, fp)

    return rgbs, disps


def create_nerf(args):
    """Instantiate NeRF's MLP model.
    """
    embed_fn, input_ch = get_embedder(args.multires, args, i=args.i_embed)
    if args.i_embed==1:
        # hashed embedding table
        embedding_params = list(embed_fn.parameters())

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        # if using hashed for xyz, use SH for views
        embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args, i=args.i_embed_views)

    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    if args.i_embed==1:
        model = NeRFSmall(num_layers=2,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        hidden_dim_color=64,
                        input_ch=input_ch, input_ch_views=input_ch_views).to(device)
    else:
        model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, output_ch=output_ch, skips=skips,
                 input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
    grad_vars = list(model.parameters())

    model_fine = None

    # if args.i_embed==1:
    #     args.N_importance = 0

    if args.N_importance > 0:
        if args.i_embed==1:
            model_fine = NeRFSmall(num_layers=2,
                        hidden_dim=64,
                        geo_feat_dim=15,
                        num_layers_color=3,
                        hidden_dim_color=64,
                        input_ch=input_ch, input_ch_views=input_ch_views).to(device)
        else:
            model_fine = NeRF(D=args.netdepth_fine, W=args.netwidth_fine,
                          input_ch=input_ch, output_ch=output_ch, skips=skips,
                          input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs).to(device)
        grad_vars += list(model_fine.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                                embed_fn=embed_fn,
                                                                embeddirs_fn=embeddirs_fn,
                                                                netchunk=args.netchunk)

    # Create optimizer
    if args.i_embed==1:
        # sparse_opt = torch.optim.SparseAdam(embedding_params, lr=args.lrate, betas=(0.9, 0.99), eps=1e-15)
        # dense_opt = torch.optim.Adam(grad_vars, lr=args.lrate, betas=(0.9, 0.99), weight_decay=1e-6)
        # optimizer = MultiOptimizer(optimizers={"sparse_opt": sparse_opt, "dense_opt": dense_opt})
        optimizer = RAdam([
                            {'params': grad_vars, 'weight_decay': 1e-6},
                            {'params': embedding_params, 'eps': 1e-15}
                        ], lr=args.lrate, betas=(0.9, 0.99))
    else:
        optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    start = 0
    basedir = args.basedir
    expname = args.expname

    ##########################

    # Load checkpoints
    if args.ft_path is not None and args.ft_path!='None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)

        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

        # Load model
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if model_fine is not None:
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])
        if args.i_embed==1:
            embed_fn.load_state_dict(ckpt['embed_fn_state_dict'])

    ##########################
    # pdb.set_trace()

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_importance' : args.N_importance,
        'network_fine' : model_fine,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'embed_fn': embed_fn,
        'use_viewdirs' : args.use_viewdirs,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # sigma_loss = sigma_sparsity_loss(raw[...,3])
    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])

    # Calculate weights sparsity loss
    try:
        entropy = Categorical(probs = torch.cat([weights, 1.0-weights.sum(-1, keepdim=True)+1e-6], dim=-1)).entropy()
    except:
        pdb.set_trace()
    sparsity_loss = entropy

    return rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss


def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                embed_fn=None,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] # [N_rays, 3] each
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2])
    near, far = bounds[...,0], bounds[...,1] # [-1,1]

    t_vals = torch.linspace(0., 1., steps=N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * (t_vals)
    else:
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]

#     raw = run_network(pts)
    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:

        rgb_map_0, disp_map_0, acc_map_0, sparsity_loss_0 = rgb_map, disp_map, acc_map, sparsity_loss

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

        run_fn = network_fn if network_fine is None else network_fine
#         raw = run_network(pts, fn=run_fn)
        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map, sparsity_loss = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map, 'sparsity_loss': sparsity_loss}
    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['sparsity_loss0'] = sparsity_loss_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")

    return ret


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.config = 'configs/chair.txt' #'config file path'
    parser.expname = 'chair'
    parser.basedir = './logs/' #'where to store ckpts and logs'
    parser.datadir = './data/nerf_synthetic/chair' #'input data directory'
    # training options
    parser.netdepth = 8, #'layers in network'
    parser.netwidth = 256 #'channels per layer'
    parser.netdepth_fine = 8 #'layers in fine network'
    parser.netwidth_fine = 256 #'channels per layer in fine network'
    parser.N_rand = 1024#32*32*4 #'batch size (number of random rays per gradient step)'
    parser.lrate = 0.01 #5e-4 #'learning rate'
    parser.lrate_decay = 500#10 #250 #'exponential learning rate decay (in 1000 steps)'
    parser.chunk = 1024*32 #'number of rays processed in parallel, decrease if running out of memory')
    parser.netchunk = 1024*64 #'number of pts sent through network in parallel, decrease if running out of memory')
    parser.no_batching = False #'only take random rays from 1 image at a time'
    parser.no_reload = False #'do not reload weights from saved ckpt'
    parser.ft_path = None #'specific weights npy file to reload for coarse network'
    # rendering options
    parser.N_samples = 64 # 'number of coarse samples per ray'
    parser.N_importance = 128#0 #number of additional fine samples per ray'
    parser.perturb = 1. #'set to 0. for no jitter, 1. for jitter'
    parser.use_viewdirs = True #'use full 5D input instead of 3D'
    parser.i_embed = 1 #'set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.i_embed_views = 2 #'set 1 for hashed embedding, 0 for default positional encoding, 2 for spherical')
    parser.multires = 10 #'log2 of max freq for positional encoding (3D location)')
    parser.multires_views = 4 #'log2 of max freq for positional encoding (2D direction)')
    parser.raw_noise_std = 0. #'std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.render_only = True #'do not optimize, reload weights and render out render_poses path')
    parser.render_test = False #'render the test set instead of render_poses path')
    parser.render_factor = 0 # 'downsampling factor to speed up rendering, set 4 or 8 for fast preview')
# training options
    parser.precrop_iters = 500#0 #'number of steps to train on central crops')
    parser.precrop_frac = 0.5 # 'fraction of img taken for central crops')
    # dataset options
    parser.dataset_type = 'blender' #'options: llff / blender / deepvoxels')
    parser.testskip = 8 #'will load 1/N images from test/val sets, useful for large datasets like deepvoxels')
    ## deepvoxels flags
    parser.shape = 'greek'#'options : armchair / cube / greek / vase')
    ## blender flags
    parser.white_bkgd = True #'set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.half_res = True#False #'load blender synthetic data at 400x400 instead of 800x800')
    ## llff flags
    parser.factor = 8 #'downsample factor for LLFF images')
    parser.no_ndc = 'store_true' #'do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.lindisp = False #'sampling linearly in disparity rather than depth')
    parser.spherify = False #'set for spherical 360 scenes')
    parser.llffhold = 8 #'will take every 1/N images as LLFF test set, paper uses 8')
    # logging/saving options
    parser.i_print = 100 #'frequency of console printout and metric loggin'
    parser.i_img = 500 #'frequency of tensorboard image logging'
    parser.i_weights = 10000 #'frequency of weight ckpt saving'
    parser.i_testset = 1000 #'frequency of testset saving')
    parser.i_video = 5000 #'frequency of render_poses video saving'
    parser.finest_res = 512 #'finest resolultion for hashed embedding'
    parser.log2_hashmap_size = 19 #'log2 of hashmap size'
    parser.sparse_loss_weight = 1e-10 #'learning rate'
    parser.tv_loss_weight = 1e-6 #'learning rate'
    return parser


torch.set_default_tensor_type('torch.cuda.FloatTensor')
parser = config_parser()
args = parser

                    
# Load data
K = None
# args.dataset_type == 'blender':
images, poses, render_poses, hwf, i_split, bounding_box = load_blender_data(args.datadir, args.half_res, args.testskip)
args.bounding_box = bounding_box
print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
i_train, i_val, i_test = i_split

near = 2.
far = 6.

images = images[...,:3]


# Cast intrinsics to right types
H, W, focal = hwf
H, W = int(H), int(W)
hwf = [H, W, focal]

if K is None:
    K = np.array([[focal, 0, 0.5*W],
                [0, focal, 0.5*H],
                [0, 0, 1]])

# Create log dir and copy the config file

basedir = args.basedir
if args.i_embed==1:
    args.expname += "_hashXYZ"
elif args.i_embed==0:
    args.expname += "_posXYZ"
if args.i_embed_views==2:
    args.expname += "_sphereVIEW"
elif args.i_embed_views==0:
    args.expname += "_posVIEW"
args.expname += "_fine"+str(args.finest_res) + "_log2T"+str(args.log2_hashmap_size)
args.expname += "_lr"+str(args.lrate) + "_decay"+str(args.lrate_decay)
args.expname += "_RAdam"
if args.sparse_loss_weight > 0:
    args.expname += "_sparse" + str(args.sparse_loss_weight)
args.expname += "_TV" + str(args.tv_loss_weight)
#args.expname += datetime.now().strftime('_%H_%M_%d_%m_%Y')
expname = args.expname

os.makedirs(os.path.join(basedir, expname), exist_ok=True)
f = os.path.join(basedir, expname, 'args.txt')
with open(f, 'w') as file:
    for arg in sorted(vars(args)):
        attr = getattr(args, arg)
        file.write('{} = {}\n'.format(arg, attr))
if args.config is not None:
    f = os.path.join(basedir, expname, 'config.txt')
    with open(f, 'w') as file:
        file.write(open(args.config, 'r').read())
# Create nerf model
render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
global_step = start

bds_dict = {'near' : near,'far' : far,}
render_kwargs_train.update(bds_dict)
render_kwargs_test.update(bds_dict)

# Move testing data to GPU
render_poses = torch.Tensor(render_poses).to(device)

# Short circuit if only rendering out from trained model

# Optimization 
testsavedir = os.path.join(args.basedir, args.expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start))
os.makedirs(testsavedir, exist_ok=True)
H, W, focal = hwf
input_image = imageio.v2.imread('./input/11.png')
pose = torch.tensor([[-0.9250140190124512,0.2748899757862091,-0.2622683644294739,-1.0572376251220703],
                     [-0.3799331784248352,-0.6692678928375244,0.6385383605957031,2.5740303993225098],
                     [0.0,0.6903012990951538,0.7235219478607178,2.9166102409362793],
                     [0.0,0.0,0.0,1.0]])

rgbs = []
#with torch.no_grad():
c2w = pose
#c2w.requires_grad = True 
rgb, _= render(H, W, K, chunk=args.chunk, c2w=c2w[:3,:4], **render_kwargs_test)
#rgb, _ = render(H, W, K, chunk=args.chunk, c2w=c2w[:3,:4], **render_kwargs_test)
rgbs.append(rgb.cpu().numpy())
rgb8 = to8b(rgbs[-1])
filename = os.path.join(testsavedir, '{:03d}.png'.format(0))
imageio.imwrite(filename, rgb8)
