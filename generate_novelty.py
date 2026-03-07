"""
CardioGen Multi-Seed Structural Novelty Matrix
"""

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import structural_similarity as ssim
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

def create_ssim_difference_map(real_slice, gen_slice, noise_threshold=0.08):
    r_norm = (real_slice - real_slice.min()) / (real_slice.max() - real_slice.min() + 1e-8)
    g_norm = (gen_slice - gen_slice.min()) / (gen_slice.max() - gen_slice.min() + 1e-8)
    
    _, ssim_map = ssim(r_norm, g_norm, data_range=1.0, full=True)
    novelty_map = 1.0 - ssim_map
    
    novelty_map[novelty_map < noise_threshold] = 0.0
    return novelty_map

def get_axial_slice(volume, slice_idx=None):
    vol = volume.squeeze().cpu().numpy() if torch.is_tensor(volume) else volume.squeeze()
    W = vol.shape[2]
    target_idx = W//2 if slice_idx is None else slice_idx
    return vol[:, :, target_idx].T

def main():
    output_dir = '/home/jl_fs/output/figures'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\nLoading models")
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
    patches, positions = split_volume_into_patches(ref_vol.squeeze(0), vae_config['patch_size'], vae_config['patch_stride'])
    vol_shape = ref_vol.squeeze(0).shape
    
    seeds = [42, 1024, 9999]
    generated_volumes = []
    
    for seed_idx, current_seed in enumerate(seeds):
        print(f"\nDiffusing Novel Patient {seed_idx + 1}/3 (Seed: {current_seed})...")
        torch.manual_seed(current_seed)
        patches_novel = []
        
        with torch.no_grad():
            batch_size = 4
            for i in tqdm(range(0, len(patches), batch_size), desc="  Processing Patches", leave=False):
                batch = patches[i:i+batch_size].to(device)
                b_size = batch.shape[0]
                
                _, skip_features = ldm.encode_patch_to_latent(batch)
                x1, x2, x3, x4, x5 = skip_features
                
                def spatial_blob_disrupt(x, strength, target_blob_size=5):
                    raw_noise = torch.randn_like(x)
                    min_spatial_dim = min(x.shape[2], x.shape[3], x.shape[4])
                    actual_blob_size = min(target_blob_size, min_spatial_dim)
                    
                    if actual_blob_size % 2 == 0:
                        actual_blob_size -= 1
                    
                    if actual_blob_size >= 3:
                        smooth_noise = F.avg_pool3d(raw_noise, kernel_size=actual_blob_size, stride=1, padding=actual_blob_size//2)
                    else:
                        smooth_noise = raw_noise
                        
                    smooth_noise = smooth_noise / (smooth_noise.std() + 1e-8)
                    
                    channel_shift = torch.randn(1, x.shape[1], 1, 1, 1, device=device) * (strength * 0.3)
                    local_warp = smooth_noise * strength
                    
                    return x * (1.0 + channel_shift + local_warp)
                
                x1_mod = spatial_blob_disrupt(x1, strength=0.2, target_blob_size=3)
                x2_mod = spatial_blob_disrupt(x2, strength=0.4, target_blob_size=3)
                x3_mod = spatial_blob_disrupt(x3, strength=0.5, target_blob_size=5)
                x4_mod = spatial_blob_disrupt(x4, strength=0.7, target_blob_size=5) 
                x5_mod = spatial_blob_disrupt(x5, strength=0.7, target_blob_size=5)
                
                # shift the high-frequency vessel maps to force rerouting
                roll_y = int(torch.randint(-2, 3, (1,)).item())
                roll_x = int(torch.randint(-2, 3, (1,)).item())
                
                x1_mod = torch.roll(x1_mod, shifts=(roll_y, roll_x), dims=(3, 4))
                x2_mod = torch.roll(x2_mod, shifts=(roll_y, roll_x), dims=(3, 4))
                
                mod_skips = (x1_mod, x2_mod, x3_mod, x4_mod, x5_mod)
                skip_cond = ldm.skip_encoder(mod_skips)
                
                z_novel = torch.randn(b_size, ldm.config['latent_dim'], device=device)
                
                for t_idx in reversed(range(ldm.config['timesteps'])):
                    t = torch.full((b_size,), t_idx, device=device, dtype=torch.long)
                    z_novel = ldm.p_sample(z_novel, t, skip_cond)
                
                z_novel = torch.clamp(z_novel, -3.0, 3.0)
                dec_novel = ldm.decode_latent_to_patch(z_novel, mod_skips)
                patches_novel.append(dec_novel.cpu())
                
        vol_novel = combine_patches_to_volume(torch.cat(patches_novel, 0), positions, vol_shape, vae_config['patch_size']).unsqueeze(0).numpy()
        generated_volumes.append(vol_novel)
    
    raw_slice_real = get_axial_slice(ref_vol)
    
    H_dim, W_dim = raw_slice_real.shape
    crop_h, crop_w = int(H_dim * 0.10), int(W_dim * 0.10)
    
    slice_real = raw_slice_real[crop_h:-crop_h, crop_w:-crop_w]
    
    slices_novel = []
    ssim_maps = []
    for vol in generated_volumes:
        raw_s_novel = get_axial_slice(vol)
        s_novel = raw_s_novel[crop_h:-crop_h, crop_w:-crop_w]
        slices_novel.append(s_novel)
        
        ssim_maps.append(create_ssim_difference_map(slice_real, s_novel, noise_threshold=0.35))
    
    print("\nRendering 2x4 Figure Matrix")
    fig = plt.figure(figsize=(22, 12))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.1, wspace=0.05)
    
    headers = ['Healthy Baseline', f'Novel Patient A', f'Novel Patient B', f'Novel Patient C']
    
    for col, img_slice in enumerate([slice_real] + slices_novel):
        ax = fig.add_subplot(gs[0, col])
        img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
        ax.imshow(img_norm, cmap='gray', origin='lower')
        ax.set_title(headers[col], fontsize=18, fontweight='bold', pad=15)
        ax.axis('off')
        if col == 0:
            ax.text(-0.05, 0.5, 'Axial View', transform=ax.transAxes, fontsize=16, fontweight='bold', rotation=90, va='center', ha='right')

    for col, img_slice in enumerate([np.zeros_like(slice_real)] + ssim_maps):
        ax = fig.add_subplot(gs[1, col])
        
        if col == 0:
            img_norm = (slice_real - slice_real.min()) / (slice_real.max() - slice_real.min() + 1e-8)
            ax.imshow(img_norm, cmap='gray', origin='lower')
        else:
            base_norm = (slice_real - slice_real.min()) / (slice_real.max() - slice_real.min() + 1e-8)
            ax.imshow(base_norm, cmap='gray', origin='lower', alpha=0.5)
            masked_diff = np.ma.masked_where(img_slice == 0, img_slice)
            im = ax.imshow(masked_diff, cmap='autumn', origin='lower', vmin=0, vmax=1)
            
            if col == 3: 
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Structural Shift Intensity', fontsize=14)
                
        ax.axis('off')
        if col == 0:
            ax.text(-0.05, 0.5, 'Structural Diff Map', transform=ax.transAxes, fontsize=16, fontweight='bold', rotation=90, va='center', ha='right')

    plt.suptitle('Multi-Seed Novelty (Vessel Rerouting & Internal Chamber Variance)', fontsize=24, fontweight='bold', y=0.95)
    
    save_path = Path(output_dir) / 'Figure3_Intense_Novelty_Rerouted_Axial.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Intense Multi-Seed Novelty figures successfully saved to: {save_path}")

if __name__ == '__main__':
    main()