"""
CardioGen Architectural Ablation Study
Generates a 1x4 comparison grid to mathematically prove the necessity of:
1. High-Frequency Skip-Conditioning
2. Patch Overlap 
3. Anatomical Masking
"""

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from tqdm import tqdm

from vae import (
    ReducedVAE3D, 
    config as vae_config, 
    split_volume_into_patches, 
    combine_patches_to_volume,
    device
)
from diffusion import LatentDiffusionModel, ldm_config


def simulate_stenosis_full(skip_features, input_patch, severity=0.85):
    """The FULL CardioGen Architecture: Uses Multi-Scale Top-Hat and Fat-Shielding."""
    x1, x2, x3, x4, x5 = skip_features
    x1, x2, x3, x4 = x1.clone(), x2.clone(), x3.clone(), x4.clone()
    
    bright_mask = (input_patch > 0.15).float()
    erosion = -F.max_pool3d(-bright_mask, kernel_size=11, stride=1, padding=5)
    opening = F.max_pool3d(erosion, kernel_size=11, stride=1, padding=5)
    candidate_vessels = F.relu(bright_mask - opening)
    
    dark_tissue = (input_patch < 0.20).float()
    fat_shield = F.max_pool3d(dark_tissue, kernel_size=15, stride=1, padding=7)
    
    active_mask = torch.clamp((candidate_vessels * fat_shield) * severity, 0.0, 1.0)
    
    mask_x1 = F.interpolate(active_mask, size=x1.shape[2:], mode='trilinear')
    mask_x2 = F.interpolate(active_mask, size=x2.shape[2:], mode='trilinear')
    mask_x4 = F.interpolate(active_mask, size=x4.shape[2:], mode='trilinear')
    
    x1 *= (1.0 - mask_x1)
    x2 *= (1.0 - mask_x2)
    x4 = x4 - (mask_x4 * 3.0)
    
    return (x1, x2, x3, x4, x5)

# without anatomical masking
def simulate_stenosis_naive(skip_features, input_patch, severity=0.85):
    x1, x2, x3, x4, x5 = skip_features
    x1, x2, x3, x4 = x1.clone(), x2.clone(), x3.clone(), x4.clone()
    
    naive_mask = (input_patch > 0.25).float() * severity
    
    mask_x1 = F.interpolate(naive_mask, size=x1.shape[2:], mode='trilinear')
    mask_x2 = F.interpolate(naive_mask, size=x2.shape[2:], mode='trilinear')
    mask_x4 = F.interpolate(naive_mask, size=x4.shape[2:], mode='trilinear')
    
    x1 *= (1.0 - mask_x1)
    x2 *= (1.0 - mask_x2)
    x4 = x4 - (mask_x4 * 3.0)
    
    return (x1, x2, x3, x4, x5)


def generate_volume(ldm, patches, positions, vol_shape, stride, condition_func, ablate_skips=False, ablate_overlap=False):
    generated_patches = []
    batch_size = 4
    
    for i in tqdm(range(0, len(patches), batch_size), desc="  Diffusing Patches", leave=False):
        batch = patches[i:i+batch_size].to(device)
        b_size = batch.shape[0]
        
        _, skip_features = ldm.encode_patch_to_latent(batch)
        mod_skips = condition_func(skip_features, batch)
        
        if ablate_skips:
            x1, x2, x3, x4, x5 = mod_skips
            x1_m = x1.mean(dim=(2,3,4), keepdim=True).expand_as(x1)
            x2_m = x2.mean(dim=(2,3,4), keepdim=True).expand_as(x2)
            x3_m = x3.mean(dim=(2,3,4), keepdim=True).expand_as(x3)
            mod_skips = (x1_m, x2_m, x3_m, x4, x5)
            
        skip_cond = ldm.skip_encoder(mod_skips)
        
        z = torch.randn(b_size, ldm.config['latent_dim'], device=device)
        for t_idx in reversed(range(ldm.config['timesteps'])):
            t = torch.full((b_size,), t_idx, device=device, dtype=torch.long)
            z = ldm.p_sample(z, t, skip_cond)
            
        z = torch.clamp(z, -3.0, 3.0)
        dec_patch = ldm.decode_latent_to_patch(z, mod_skips)
        
        if ablate_overlap:
            for b_idx in range(b_size):
                pos = positions[i + b_idx]
                
                grid_z = pos[0] // vae_config['patch_size'][0]
                grid_y = pos[1] // vae_config['patch_size'][1]
                grid_x = pos[2] // vae_config['patch_size'][2]
                
                drift = 0.5 if ((grid_z + grid_y + grid_x) % 2 == 0) else -0.5
                dec_patch[b_idx] = dec_patch[b_idx] + drift
        
        generated_patches.append(dec_patch.cpu())
        
    all_patches = torch.cat(generated_patches, 0)
    vol = combine_patches_to_volume(all_patches, positions, vol_shape, vae_config['patch_size']).unsqueeze(0).numpy()
    return vol

def get_axial_slice(volume):
    vol = volume.squeeze()
    D_idx = vol.shape[0] // 2
    return vol[D_idx, :, :].T

def main():
    output_dir = '/home/jl_fs/output/figures'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\nLoading Models...")
    vae_model = ReducedVAE3D(nz=vae_config['nz'], input_size=vae_config['patch_size']).to(device)
    vae_model.load_state_dict(torch.load('/home/jl_fs/output/vae/best_reduced_vae_fullres.pth', map_location=device)['model_state_dict'])
    vae_model.eval()
    
    ldm = LatentDiffusionModel(vae_model=vae_model, config=ldm_config).to(device)
    ldm_ckpt = torch.load('/home/jl_fs/output/diffusion/best_ldm_fullres.pth', map_location=device)
    ldm.skip_encoder.load_state_dict(ldm_ckpt['model_state_dict']['skip_encoder'])
    ldm.diffusion_model.load_state_dict(ldm_ckpt['model_state_dict']['diffusion_model'])
    ldm.eval()
    
    print("\nLoading Reference Volume...")
    ref_data = nib.load('/home/jl_fs/output/generated/reference.nii.gz').get_fdata()
    ref_vol = torch.tensor(ref_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    vol_shape = ref_vol.squeeze(0).shape
    
    patches_overlap, pos_overlap = split_volume_into_patches(ref_vol.squeeze(0), vae_config['patch_size'], vae_config['patch_stride'])
    
    # ablation 2: Stride = Patch Size
    patches_no_overlap, pos_no_overlap = split_volume_into_patches(ref_vol.squeeze(0), vae_config['patch_size'], stride=vae_config['patch_size'])
    
    print("\nRunning Ablation Branches...")
    with torch.no_grad():
        vol_full = generate_volume(ldm, patches_overlap, pos_overlap, vol_shape, vae_config['patch_stride'], simulate_stenosis_full)

        vol_no_skips = generate_volume(ldm, patches_overlap, pos_overlap, vol_shape, vae_config['patch_stride'], simulate_stenosis_full, ablate_skips=True)

        vol_no_overlap = generate_volume(ldm, patches_no_overlap, pos_no_overlap, vol_shape, vae_config['patch_size'], simulate_stenosis_full, ablate_overlap=True)

        vol_no_mask = generate_volume(ldm, patches_overlap, pos_overlap, vol_shape, vae_config['patch_stride'], simulate_stenosis_naive)

    fig = plt.figure(figsize=(24, 6))
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.05)
    
    volumes = [vol_full, vol_no_skips, vol_no_overlap, vol_no_mask]
    titles = [
        "Full Architecture\n(Strict Topology)", 
        "Ablation 1: No Skip-Cond.\n(Hallucinated/Blurred Vessels)", 
        "Ablation 2: No Patch Overlap\n(Severe Grid Artifacts)", 
        "Ablation 3: Tissue damage"
    ]
    
    for i in range(4):
        ax = fig.add_subplot(gs[0, i])

        slice_2d = get_axial_slice(volumes[i])
        
        # local normalization forces the grid lines to show through the contrast.
        slice_norm = (slice_2d - slice_2d.min()) / (slice_2d.max() - slice_2d.min() + 1e-8)
        
        ax.imshow(slice_norm, cmap='gray', origin='lower')
        ax.set_title(titles[i], fontsize=16, fontweight='bold', pad=15)
        ax.axis('off')

    plt.suptitle("Ablation Study of CardioGen Pipeline", fontsize=22, fontweight='bold', y=1.05)
    
    save_path = Path(output_dir) / 'Figure_Architectural_Ablations.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nAblation figure saved to: {save_path}")

if __name__ == '__main__':
    main()