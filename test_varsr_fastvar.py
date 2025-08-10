import os
import glob
import argparse
import time
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

import dist
from utils import arg_util
from utils.data_sampler import EvalDistributedSampler
from dataloader.testdataset import TestDataset
from myutils.wavelet_color_fix import adain_color_fix
from models.vqvae import VQVAE
from models.var import VAR_RoPE, build_var_rope_with_fastvar

# --------------------------------------------------
# Helper functions (copied / adapted from test_varsr.py)
# --------------------------------------------------

def numpy_to_pil(images: np.ndarray):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        return [Image.fromarray(image.squeeze(), mode="L") for image in images]
    return [Image.fromarray(image) for image in images]

def pt_to_numpy(images: torch.Tensor) -> np.ndarray:
    return images.cpu().permute(0, 2, 3, 1).float().numpy()

# --------------------------------------------------
# Build model with or without FastVAR
# --------------------------------------------------

def build_models(args, fastvar_cfg):
    """Replicate build_var logic but allow injecting FastVAR pruning flags."""
    from models.vqvae import VQVAE
    # derive width/head as in build_var
    depth = args.depth
    heads = depth
    width = depth * 64
    dpr = 0.1 * depth / 24

    # Build VQVAE
    vae_local = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=True, share_quant_resi=4, v_patch_nums=args.patch_nums).to(dist.get_device())

    # Build VAR
    if fastvar_cfg.enable_fastvar:
        second_last_side, last_side = args.patch_nums[-2], args.patch_nums[-1]
        override_map = {}
        if fastvar_cfg.second_last_ratio > 0:
            override_map[second_last_side] = fastvar_cfg.second_last_ratio
        if fastvar_cfg.last_ratio > 0:
            override_map[last_side] = fastvar_cfg.last_ratio
        var_wo_ddp = VAR_RoPE(
            vae_local=vae_local,
            num_classes=1 + 1, depth=depth, controlnet_depth=depth, embed_dim=width, num_heads=heads,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
            norm_eps=1e-6, shared_aln=args.saln, cond_drop_rate=0.0,
            attn_l2_norm=args.anorm,
            patch_nums=tuple(args.patch_nums),
            flash_if_available=args.fuse, fused_if_available=args.fuse,
            enable_fastvar_prune=True,
            fastvar_later_layer_start=fastvar_cfg.later_layer_start,
            fastvar_min_keep=fastvar_cfg.min_keep,
            fastvar_override_map=override_map,
        ).to(dist.get_device())
        setattr(var_wo_ddp, 'fastvar_quiet', fastvar_cfg.quiet)
        if getattr(var_wo_ddp, 'fastvar_pruner', None) is not None:
            var_wo_ddp.fastvar_pruner.quiet = fastvar_cfg.quiet
    else:
        var_wo_ddp = VAR_RoPE(
            vae_local=vae_local,
            num_classes=1 + 1, depth=depth, controlnet_depth=depth, embed_dim=width, num_heads=heads,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=dpr,
            norm_eps=1e-6, shared_aln=args.saln, cond_drop_rate=0.0,
            attn_l2_norm=args.anorm,
            patch_nums=tuple(args.patch_nums),
            flash_if_available=args.fuse, fused_if_available=args.fuse,
        ).to(dist.get_device())

    # Init weights (mirrors build_var)
    var_wo_ddp.init_weights(init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini)
    return vae_local, var_wo_ddp

# --------------------------------------------------
# Inference Loop
# --------------------------------------------------

def run_inference(args, fastvar_cfg):
    # Load models
    vae, var = build_models(args, fastvar_cfg)
    vae_ckpt = args.vae_model_path
    var_ckpt = args.var_test_path
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu')['trainer']['vae_local'], strict=True)
    model_state = torch.load(var_ckpt, map_location='cpu')
    var.load_state_dict(model_state['trainer']['var_wo_ddp'], strict=True)
    vae.eval(); var.eval()

    dataset_load_reso = args.data_load_reso
    folders = os.listdir("testset/")
    val_sets = []
    for folder in folders:
        dataset_val = TestDataset("testset/" + folder, image_size=dataset_load_reso, tokenizer=None, resize_bak=True)
        from torch.utils.data import DataLoader
        ld_val = DataLoader(
            dataset_val, num_workers=0, pin_memory=True,
            batch_size=round(args.batch_size), sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
            shuffle=False, drop_last=False,
        )
        val_sets.append(ld_val)

    total_images = 0
    total_time = 0.0

    for ld_val in val_sets:
        for batch in ld_val:
            lr_inp = batch['conditioning_pixel_values'].to(args.device, non_blocking=True)
            label_B = batch['label_B'].to(args.device, non_blocking=True)
            B = lr_inp.shape[0]
            torch.cuda.synchronize()
            start = time.time()
            with torch.inference_mode(), torch.autocast('cuda', enabled=True, dtype=torch.float16, cache_enabled=True):
                recon = var.autoregressive_infer_cfg(
                    B=B, cfg=6.0, top_k=1, top_p=0.75,
                    text_hidden=None, lr_inp=lr_inp, negative_text=None, label_B=label_B, lr_inp_scale=None,
                    more_smooth=False,
                    return_intermediate=fastvar_cfg.export_intermediate,
                    intermediate_max=fastvar_cfg.intermediate_max,
                )
            intermediates = None
            if isinstance(recon, tuple):
                recon, intermediates = recon
            torch.cuda.synchronize()
            elapsed = time.time() - start
            total_time += elapsed
            total_images += B

            recon_pil = numpy_to_pil(pt_to_numpy(recon))
            for i in range(B):
                image = recon_pil[i]
                validation_image = Image.open(batch['path'][i].replace('/HR', '/LR')).convert('RGB').resize((512, 512))
                image = adain_color_fix(image, validation_image)
                folder_path, ext_path = os.path.split(batch['path'][i])
                output_name = folder_path.replace('/LR', '/VARPrediction/').replace('/HR', '/VARPrediction/')
                os.makedirs(output_name, exist_ok=True)
                # save final image at root (existing behavior)
                final_out_path = os.path.join(output_name, ext_path)
                image.save(final_out_path)
                # unified per-image scales folder
                base_name_no_ext = os.path.splitext(ext_path)[0]
                scales_dir = os.path.join(output_name, f"{base_name_no_ext}_scales")
                os.makedirs(scales_dir, exist_ok=True)
                # save intermediates (color-fix each) sideX.png
                if intermediates is not None:
                    for side, mid_img in intermediates:
                        mid_np = pt_to_numpy(mid_img[i:i+1])[0]
                        mid_pil = numpy_to_pil(mid_np)[0]
                        mid_pil = adain_color_fix(mid_pil, validation_image)
                        mid_pil.save(os.path.join(scales_dir, f"side{side}.png"))
                # also ensure final scale stored in scales folder (side = last patch size)
                try:
                    final_side = getattr(var, 'patch_nums', [None])[-1]
                    image.save(os.path.join(scales_dir, f"side{final_side}.png"))
                except Exception:
                    pass

    if dist.get_rank() == 0:
        print(f"[FastVAR-Test] enable_fastvar={fastvar_cfg.enable_fastvar} second_last_ratio={fastvar_cfg.second_last_ratio} last_ratio={fastvar_cfg.last_ratio} later_layer_start={fastvar_cfg.later_layer_start} min_keep={fastvar_cfg.min_keep}")
        print(f"[FastVAR-Test] Total images: {total_images}  Total time: {total_time:.3f}s  Avg img time: {total_time/total_images:.4f}s")
        # Print pruning summary
        if fastvar_cfg.enable_fastvar and hasattr(var, 'fastvar_pruner') and var.fastvar_pruner is not None:
            print(var.fastvar_pruner.summary())


# --------------------------------------------------
# Main
# --------------------------------------------------

def parse_fastvar_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--fastvar', action='store_true', help='Enable FastVAR pruning')
    parser.add_argument('--fastvar_second_last_ratio', type=float, default=0.4, help='Drop ratio for second last stage')
    parser.add_argument('--fastvar_last_ratio', type=float, default=0.3, help='Drop ratio for last stage (set 0 to disable)')
    parser.add_argument('--fastvar_later_layer_start', type=int, default=3, help='Start pruning from this layer index inside a stage')
    parser.add_argument('--fastvar_min_keep', type=int, default=64, help='Minimum tokens to keep after pruning')
    parser.add_argument('--fastvar_quiet', action='store_true', help='Suppress FastVAR per-layer prune logs')
    parser.add_argument('--export_intermediate', action='store_true', help='Export intermediate decoded images')
    parser.add_argument('--intermediate_max', type=int, default=10, help='Max number of intermediate stages to export')
    return parser.parse_known_args()[0]

class FastVarCfg:
    def __init__(self, ns):
        self.enable_fastvar = ns.fastvar
        self.second_last_ratio = ns.fastvar_second_last_ratio
        self.last_ratio = ns.fastvar_last_ratio
        self.later_layer_start = ns.fastvar_later_layer_start
        self.min_keep = ns.fastvar_min_keep
        self.quiet = getattr(ns, 'fastvar_quiet', False)
        self.export_intermediate = getattr(ns, 'export_intermediate', False)
        self.intermediate_max = getattr(ns, 'intermediate_max', 10)

if __name__ == '__main__':
    import sys
    # Extract fastvar args first
    fv_ns = parse_fastvar_args()
    # Remove them from sys.argv so arg_util does not warn
    fastvar_flags = {"--fastvar", "--fastvar_second_last_ratio", "--fastvar_last_ratio", "--fastvar_later_layer_start", "--fastvar_min_keep", "--fastvar_quiet", "--export_intermediate", "--intermediate_max"}
    bool_flags = {"--fastvar", "--fastvar_quiet", "--export_intermediate"}
    cleaned = []
    skip_next = False
    for i,a in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if a in fastvar_flags:
            if a not in bool_flags:  # flags expecting a value
                skip_next = True
            continue
        cleaned.append(a)
    sys.argv = [sys.argv[0]] + cleaned
    base_args = arg_util.init_dist_and_get_args()
    fastvar_cfg = FastVarCfg(fv_ns)

    # Ensure depth is consistent (original test script forced 24)
    base_args.depth = getattr(base_args, 'depth', 24) or 24

    # Provide default checkpoint paths if not set
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    default_vae = os.path.join(ckpt_dir, 'VQVAE.pth')
    default_var = os.path.join(ckpt_dir, 'VARSR.pth')
    if not hasattr(base_args, 'vae_model_path') or base_args.vae_model_path is None:
        if os.path.isfile(default_vae):
            base_args.vae_model_path = default_vae
            print(f"[FastVAR-Test] Using default VAE checkpoint: {default_vae}")
        else:
            raise FileNotFoundError('VAE checkpoint path not provided and default not found.')
    if not hasattr(base_args, 'var_test_path') or base_args.var_test_path is None:
        if os.path.isfile(default_var):
            base_args.var_test_path = default_var
            print(f"[FastVAR-Test] Using default VAR checkpoint: {default_var}")
        else:
            raise FileNotFoundError('VAR checkpoint path not provided and default not found.')

    # Patch device reference if missing
    if not hasattr(base_args, 'device'):
        base_args.device = dist.get_device()

    run_inference(base_args, fastvar_cfg)
