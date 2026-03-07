"""
Figure: Targeted Pathology Generatiio
Demonstrates parametric manipulation of latent spatial frequencies to induce 
Myocardial Infarction, Hypertrophic Cardiomyopathy, and Coronary Artery Stenosis.
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
from diffusion import LatentDiffusionModel, ldm_config, match_intensity_clahe


def simulate_myocardial_infarction(skip_features, input_patch, severity=2.5, global_pos=None, global_shape=None, location='anterior'):
    x1, x2, x3, x4, x5 = skip_features
    x1, x2, x3, x4 = x1.clone(), x2.clone(), x3.clone(), x4.clone()
    
    muscle_mask = ((input_patch > 0.05) & (input_patch < 0.35)).float()
    muscle_mask = F.avg_pool3d(muscle_mask, kernel_size=3, stride=1, padding=1)
    
    B, C, D, H, W = input_patch.shape
    spatial_target = torch.ones((B, 1, D, H, W), device=device)
    
    if global_pos is not None and global_shape is not None:
        D_full, H_full, W_full = global_shape
        for b in range(B):
            d_start, h_start, w_start = global_pos[b]
            z_vals = torch.linspace(d_start, d_start + D - 1, D, device=device) / max(1, D_full - 1)
            y_vals = torch.linspace(h_start, h_start + H - 1, H, device=device) / max(1, H_full - 1)
            x_vals = torch.linspace(w_start, w_start + W - 1, W, device=device) / max(1, W_full - 1)
            
            z = z_vals.view(D, 1, 1).expand(D, H, W)
            y = y_vals.view(1, H, 1).expand(D, H, W)
            x = x_vals.view(1, 1, W).expand(D, H, W)
            
            cx, cy, cz = (0.5, 0.35, 0.5) if location == 'anterior' else (0.5, 0.5, 0.5)
            
            raw_target = torch.exp(-((x - cx)**2 / 0.02 + (y - cy)**2 / 0.015 + (z - cz)**2 / 0.02))
            sharp_target = torch.clamp((raw_target - 0.4) * 10.0, 0.0, 1.0)
            spatial_target[b, 0, :, :, :] = sharp_target

    active_mask = torch.clamp(muscle_mask * spatial_target * severity, 0.0, 1.0)
    
    mask_x1 = F.interpolate(active_mask, size=x1.shape[2:], mode='trilinear', align_corners=False)
    mask_x2 = F.interpolate(active_mask, size=x2.shape[2:], mode='trilinear', align_corners=False)
    mask_x3 = F.interpolate(active_mask, size=x3.shape[2:], mode='trilinear', align_corners=False)
    mask_x4 = F.interpolate(active_mask, size=x4.shape[2:], mode='trilinear', align_corners=False)

    blood_mask = (input_patch > 0.35).float()
    def get_blood_sig(feat):
        m = F.interpolate(blood_mask, size=feat.shape[2:], mode='trilinear', align_corners=False)
        valid_pixels_safe = torch.clamp(m.sum(dim=(2,3,4), keepdim=True), min=1.0)
        return (feat * m).sum(dim=(2,3,4), keepdim=True) / valid_pixels_safe

    x1 = torch.lerp(x1, get_blood_sig(x1), mask_x1)
    x2 = torch.lerp(x2, get_blood_sig(x2), mask_x2)
    x3 = torch.lerp(x3, get_blood_sig(x3), mask_x3)
    x4 = torch.lerp(x4, get_blood_sig(x4), mask_x4)
    
    return (x1, x2, x3, x4, x5)

def simulate_cardiomyopathy(skip_features, input_patch, severity=2.5, global_pos=None, global_shape=None):
    x1, x2, x3, x4, x5 = skip_features
    x1, x2, x3, x4 = x1.clone(), x2.clone(), x3.clone(), x4.clone()
    
    raw_muscle_mask = ((input_patch > 0.05) & (input_patch < 0.35)).float()
    smoothed_muscle = F.avg_pool3d(raw_muscle_mask, kernel_size=5, stride=1, padding=2)
    solid_wall_mask = (smoothed_muscle > 0.6).float()
    
    thick_muscle_mask = F.max_pool3d(solid_wall_mask, kernel_size=9, stride=1, padding=4)
    intrusion_mask = F.avg_pool3d(F.relu(thick_muscle_mask - raw_muscle_mask), kernel_size=3, stride=1, padding=1)
    
    B, C, D, H, W = input_patch.shape
    spatial_target = torch.zeros((B, 1, D, H, W), device=device)
    
    if global_pos is not None and global_shape is not None:
        D_full, H_full, W_full = global_shape
        for b in range(B):
            d_start, h_start, w_start = global_pos[b]
            z = (torch.linspace(d_start, d_start + D - 1, D, device=device) / max(1, D_full - 1)).view(D, 1, 1).expand(D, H, W)
            y = (torch.linspace(h_start, h_start + H - 1, H, device=device) / max(1, H_full - 1)).view(1, H, 1).expand(D, H, W)
            x = (torch.linspace(w_start, w_start + W - 1, W, device=device) / max(1, W_full - 1)).view(1, 1, W).expand(D, H, W)
            spatial_target[b, 0, :, :, :] = torch.exp(-((x - 0.5)**2 / 0.06 + (y - 0.45)**2 / 0.06 + (z - 0.5)**2 / 0.06))

    active_mask = torch.clamp(intrusion_mask * spatial_target * (severity / 2.5), 0.0, 1.0)
        
    mask_x1 = F.interpolate(active_mask, size=x1.shape[2:], mode='trilinear', align_corners=False)
    mask_x2 = F.interpolate(active_mask, size=x2.shape[2:], mode='trilinear', align_corners=False)
    mask_x3 = F.interpolate(active_mask, size=x3.shape[2:], mode='trilinear', align_corners=False)
    mask_x4 = F.interpolate(active_mask, size=x4.shape[2:], mode='trilinear', align_corners=False)

    dense_muscle_mask = ((input_patch > 0.10) & (input_patch < 0.25)).float()
    def get_muscle_sig(feat):
        m = F.interpolate(dense_muscle_mask, size=feat.shape[2:], mode='trilinear', align_corners=False)
        return (feat * m).sum(dim=(2,3,4), keepdim=True) / torch.clamp(m.sum(dim=(2,3,4), keepdim=True), min=1.0)

    x1 = torch.lerp(x1, get_muscle_sig(x1), torch.clamp(mask_x1 * 2.0, 0.0, 1.0))
    x2 = torch.lerp(x2, get_muscle_sig(x2), torch.clamp(mask_x2 * 2.0, 0.0, 1.0))
    x3 = torch.lerp(x3, get_muscle_sig(x3), torch.clamp(mask_x3 * 2.0, 0.0, 1.0))
    x4 = torch.lerp(x4, get_muscle_sig(x4) - 0.15, torch.clamp(mask_x4 * 2.0, 0.0, 1.0))
        
    return (x1, x2, x3, x4, x5)

def simulate_stenosis(skip_features, input_patch, severity=2.5, global_pos=None, global_shape=None):
    x1, x2, x3, x4, x5 = skip_features
    x1, x2, x3, x4 = x1.clone(), x2.clone(), x3.clone(), x4.clone() 
    
    bright_mask = (input_patch > 0.15).float()
    
    e15 = -F.max_pool3d(-bright_mask, kernel_size=15, stride=1, padding=7)
    o15 = F.max_pool3d(e15, kernel_size=15, stride=1, padding=7)
    m15 = F.relu(bright_mask - o15)

    e11 = -F.max_pool3d(-bright_mask, kernel_size=11, stride=1, padding=5)
    o11 = F.max_pool3d(e11, kernel_size=11, stride=1, padding=5)
    m11 = F.relu(bright_mask - o11)
    
    e7 = -F.max_pool3d(-bright_mask, kernel_size=7, stride=1, padding=3)
    o7 = F.max_pool3d(e7, kernel_size=7, stride=1, padding=3)
    m7 = F.relu(bright_mask - o7)
    
    candidate_vessels = torch.clamp(m15 + m11 + m7, 0.0, 1.0)
    
    dark_tissue = (input_patch < 0.20).float()
    fat_shield = F.max_pool3d(dark_tissue, kernel_size=15, stride=1, padding=7)
    fat_shield = F.avg_pool3d(fat_shield, kernel_size=5, stride=1, padding=2)
    solid_artery_mask = candidate_vessels * fat_shield
    
    eroded_core = -F.max_pool3d(-solid_artery_mask, kernel_size=3, stride=1, padding=1)
    raw_ring = F.relu(solid_artery_mask - eroded_core)
    
    artery_ring_mask = torch.clamp(raw_ring * 10.0, 0.0, 1.0)
    active_mask = torch.clamp(artery_ring_mask * severity, 0.0, 1.0)
    
    mask_x1 = F.interpolate(active_mask, size=x1.shape[2:], mode='trilinear', align_corners=False)
    mask_x2 = F.interpolate(active_mask, size=x2.shape[2:], mode='trilinear', align_corners=False)
    mask_x3 = F.interpolate(active_mask, size=x3.shape[2:], mode='trilinear', align_corners=False)
    mask_x4 = F.interpolate(active_mask, size=x4.shape[2:], mode='trilinear', align_corners=False)
    
    dark_factor = 0.0001
    x1 = x1 * (1.0 - mask_x1) + (x1 * dark_factor) * mask_x1
    x2 = x2 * (1.0 - mask_x2) + (x2 * dark_factor) * mask_x2
    x3 = x3 * (1.0 - mask_x3) + (x3 * dark_factor) * mask_x3
    x4 = x4 * (1.0 - mask_x4) + (x4 * dark_factor) * mask_x4
    
    return (x1, x2, x3, x4, x5)


def get_axial_slice(volume, slice_idx=None):
    vol = volume.squeeze().cpu().numpy() if torch.is_tensor(volume) else volume.squeeze()
    W = vol.shape[2]
    target_idx = W//2 if slice_idx is None else slice_idx
    return vol[:, :, target_idx].T

def create_filtered_3d_difference(base_volume, path_volume, noise_threshold=0.03):
    base_vol = base_volume.squeeze().cpu().numpy() if torch.is_tensor(base_volume) else base_volume.squeeze()
    path_vol = path_volume.squeeze().cpu().numpy() if torch.is_tensor(path_volume) else path_volume.squeeze()

    v_min, v_max = base_vol.min(), base_vol.max()
    base_norm = (base_vol - v_min) / (v_max - v_min + 1e-8)
    path_norm = (path_vol - v_min) / (v_max - v_min + 1e-8)
    
    diff_map = np.abs(base_norm - path_norm)
    diff_map[diff_map < noise_threshold] = 0.0
    return np.clip(diff_map * 5.0, 0, 1) 

def add_scale_bar(ax, bar_length_mm=40):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x_start, y_pos = xlim[1] - bar_length_mm - 20, ylim[0] - 20
    ax.plot([x_start, x_start + bar_length_mm], [y_pos, y_pos], color='white', linewidth=3)


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
    patches, positions = split_volume_into_patches(ref_vol.squeeze(0), vae_config['patch_size'], vae_config['patch_stride'])
    
    vol_shape = ref_vol.squeeze(0).shape  
    global_shape = vol_shape[1:]          
    
    patches_base, patches_sten, patches_mi, patches_cm = [], [], [], []
    
    print("\nDiffusing Synchronized Pathologies...")
    with torch.no_grad():
        batch_size = 4
        for i in tqdm(range(0, len(patches), batch_size), desc="Processing Patches"):
            batch = patches[i:i+batch_size].to(device)
            b_size = batch.shape[0]
            batch_positions = positions[i:i+batch_size]
            
            _, skip_features = ldm.encode_patch_to_latent(batch)
            
            skip_sten = simulate_stenosis(skip_features, batch, stenosis_severity=2.5, global_pos=batch_positions, global_shape=global_shape)
            skip_mi = simulate_myocardial_infarction(skip_features, batch, severity=2.5, global_pos=batch_positions, global_shape=global_shape, location='anterior')
            skip_cm = simulate_cardiomyopathy(skip_features, batch, severity=2.5, global_pos=batch_positions, global_shape=global_shape)
            
            skip_cond_base = ldm.skip_encoder(skip_features)
            skip_cond_sten = ldm.skip_encoder(skip_sten)
            skip_cond_mi   = ldm.skip_encoder(skip_mi)
            skip_cond_cm   = ldm.skip_encoder(skip_cm)
            
            z_start = torch.randn(b_size, ldm.config['latent_dim'], device=device)
            z_base, z_sten, z_mi, z_cm = z_start.clone(), z_start.clone(), z_start.clone(), z_start.clone()
            
            for t_idx in reversed(range(ldm.config['timesteps'])):
                t = torch.full((b_size,), t_idx, device=device, dtype=torch.long)
                z_base = ldm.p_sample(z_base, t, skip_cond_base)
                z_sten = ldm.p_sample(z_sten, t, skip_cond_sten)
                z_mi   = ldm.p_sample(z_mi, t, skip_cond_mi)
                z_cm   = ldm.p_sample(z_cm, t, skip_cond_cm)
            
            patches_base.append(ldm.decode_latent_to_patch(torch.clamp(z_base, -3.0, 3.0), skip_features).cpu())
            patches_sten.append(ldm.decode_latent_to_patch(torch.clamp(z_sten, -3.0, 3.0), skip_sten).cpu())
            patches_mi.append(ldm.decode_latent_to_patch(torch.clamp(z_mi, -3.0, 3.0), skip_mi).cpu())
            patches_cm.append(ldm.decode_latent_to_patch(torch.clamp(z_cm, -3.0, 3.0), skip_cm).cpu())

    vol_base = combine_patches_to_volume(torch.cat(patches_base, 0), positions, vol_shape, vae_config['patch_size']).unsqueeze(0)
    vol_sten = combine_patches_to_volume(torch.cat(patches_sten, 0), positions, vol_shape, vae_config['patch_size']).unsqueeze(0)
    vol_mi   = combine_patches_to_volume(torch.cat(patches_mi, 0), positions, vol_shape, vae_config['patch_size']).unsqueeze(0)
    vol_cm   = combine_patches_to_volume(torch.cat(patches_cm, 0), positions, vol_shape, vae_config['patch_size']).unsqueeze(0)
    
    diff_sten = create_filtered_3d_difference(vol_base, vol_sten)
    diff_mi   = create_filtered_3d_difference(vol_base, vol_mi)
    diff_cm   = create_filtered_3d_difference(vol_base, vol_cm)
    
    print("\nAssembling 3x3 Axial Matrix...")
    
    ax_base = get_axial_slice(vol_base)

    path_slices = [
        (get_axial_slice(vol_mi),   get_axial_slice(diff_mi)),
        (get_axial_slice(vol_cm),   get_axial_slice(diff_cm)),
        (get_axial_slice(vol_sten), get_axial_slice(diff_sten))
    ]
    
    fig = plt.figure(figsize=(16, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.03, wspace=0.03)
    
    row_titles = ['Myocardial Infarction', 'Hypertrophic Cardiomyopathy', 'Coronary Stenosis']
    col_headers = ['Healthy Baseline (Axial)', 'Generated Pathology', 'Structural Diff Map']
    
    for row, (path_img, diff_img) in enumerate(path_slices):
        images = [ax_base, path_img, diff_img]
        
        for col, img_slice in enumerate(images):
            ax = fig.add_subplot(gs[row, col])
            
            if col == 2:
                base_norm = (ax_base - ax_base.min()) / (ax_base.max() - ax_base.min() + 1e-8)
                ax.imshow(base_norm, cmap='gray', origin='lower', alpha=0.4)
                masked_diff = np.ma.masked_where(img_slice == 0, img_slice)
                im = ax.imshow(masked_diff, cmap='autumn', origin='lower', vmin=0, vmax=1)
            else:
                img_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
                ax.imshow(img_norm, cmap='gray', origin='lower')
                
            ax.axis('off')
            add_scale_bar(ax)
            
            if row == 0:
                ax.set_title(col_headers[col], fontsize=18, fontweight='bold', pad=15)
            if col == 0:
                ax.text(-0.05, 0.5, row_titles[row], transform=ax.transAxes, fontsize=18, fontweight='bold', rotation=90, va='center', ha='right')
                
            if row == 1 and col == 2:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Pathological Shift Intensity', fontsize=14)

    plt.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.05)
    
    fig.suptitle('Parametric Pathology Induction & Structural Localization', fontsize=24, fontweight='bold', y=0.98)
    
    save_path = Path(output_dir) / 'Figure_Merged_Pathologies_Axial.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Merged Axial figure successfully saved to: {save_path}")

if __name__ == '__main__':
    main()