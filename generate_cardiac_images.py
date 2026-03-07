"""
Complete script for generating synthetic cardiac images and simulating cardiac conditions
Uses trained VAE and LDM models
"""

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from diffusion import LatentDiffusionModel, ldm_config, device
from vae import (
    ReducedVAE3D, 
    config as vae_config,
    split_volume_into_patches,
    combine_patches_to_volume,
    load_and_preprocess_nii
)

def visualize_midslice(volume, title, save_path):
    if torch.is_tensor(volume):
        volume_np = volume.cpu().numpy()
    else:
        volume_np = volume
    
    if len(volume_np.shape) == 5:
        mid_slice = volume_np.shape[2] // 2
        slice_img = volume_np[0, 0, mid_slice]
    elif len(volume_np.shape) == 4:
        mid_slice = volume_np.shape[1] // 2
        slice_img = volume_np[0, mid_slice]
    elif len(volume_np.shape) == 3:
        mid_slice = volume_np.shape[0] // 2
        slice_img = volume_np[mid_slice]
    else:
        raise ValueError(f"Unexpected volume shape: {volume_np.shape}")
    
    plt.figure(figsize=(8, 8))
    plt.imshow(slice_img, cmap='gray')
    plt.title(title, fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved: {save_path}")


def visualize_comparison(reference, generated_list, labels, save_path):
    if torch.is_tensor(reference):
        reference_np = reference.cpu().numpy()
    else:
        reference_np = reference
    
    n_samples = len(generated_list) + 1  # +1 for reference
    _, axes = plt.subplots(1, n_samples, figsize=(5 * n_samples, 5))
    
    if n_samples == 1:
        axes = [axes]
    
    if len(reference_np.shape) == 5:
        mid_slice = reference_np.shape[2] // 2
        ref_img = reference_np[0, 0, mid_slice]
    else:
        mid_slice = reference_np.shape[0] // 2
        ref_img = reference_np[mid_slice]
    
    axes[0].imshow(ref_img, cmap='gray')
    axes[0].set_title('Reference', fontsize=12)
    axes[0].axis('off')
    
    for i, (gen_vol, label) in enumerate(zip(generated_list, labels)):
        if torch.is_tensor(gen_vol):
            gen_np = gen_vol.cpu().numpy()
        else:
            gen_np = gen_vol
        
        if len(gen_np.shape) == 5:
            mid_slice = gen_np.shape[2] // 2
            gen_img = gen_np[0, 0, mid_slice]
        else:
            mid_slice = gen_np.shape[0] // 2
            gen_img = gen_np[mid_slice]
        
        axes[i+1].imshow(gen_img, cmap='gray')
        axes[i+1].set_title(label, fontsize=12)
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison saved: {save_path}")


def visualize_progression(volumes, severities, condition_name, save_path):
    n_stages = len(volumes)
    _, axes = plt.subplots(1, n_stages, figsize=(4 * n_stages, 4))
    
    if n_stages == 1:
        axes = [axes]
    
    for i, (vol, severity) in enumerate(zip(volumes, severities)):
        if torch.is_tensor(vol):
            vol_np = vol.cpu().numpy()
        else:
            vol_np = vol
        
        if len(vol_np.shape) == 5:
            mid_slice = vol_np.shape[2] // 2
            slice_img = vol_np[0, 0, mid_slice]
        else:
            mid_slice = vol_np.shape[0] // 2
            slice_img = vol_np[mid_slice]
        
        axes[i].imshow(slice_img, cmap='gray')
        axes[i].set_title(f'Severity: {severity:.2f}', fontsize=12)
        axes[i].axis('off')
    
    plt.suptitle(f'{condition_name} Progression', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Progression visualization saved: {save_path}")


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

def generate_normal_variations(ldm, reference_volume, num_variations=3, output_dir='./output/normal'):
    os.makedirs(output_dir, exist_ok=True)
    
    generated_volumes = []
    
    with torch.no_grad():
        for i in range(num_variations):
            print(f"\nGenerating variation {i+1}/{num_variations}...")
            
            variation = ldm.generate_full_volume(
                reference_volume,
                patch_size=vae_config['patch_size'],
                stride=vae_config['patch_stride'],
                num_variations=1,
                variation_strength=0.35
            )
            
            generated_volumes.append(variation)
            
            vol_np = variation[0, 0].cpu().numpy()
            nii_img = nib.Nifti1Image(vol_np, affine=np.eye(4))
            save_path = f'{output_dir}/normal_variation_{i+1}.nii.gz'
            nib.save(nii_img, save_path)
            print(f"Saved: {save_path}")
            
            viz_path = f'{output_dir}/normal_variation_{i+1}_midslice.png'
            visualize_midslice(variation, f'Normal Variation {i+1}', viz_path)
    
    if len(generated_volumes) > 0:
        comparison_path = f'{output_dir}/normal_variations_comparison.png'
        labels = [f'Variation {i+1}' for i in range(len(generated_volumes))]
        visualize_comparison(reference_volume, generated_volumes, labels, comparison_path)
    
    print(f"\nGenerated {num_variations} normal variations")
    return generated_volumes


def generate_with_condition(ldm, reference_volume, condition_func, condition_params, 
                            num_samples=3, output_dir='./output/condition'):
     
    os.makedirs(output_dir, exist_ok=True)
    
    vol_tensor = reference_volume.squeeze(0)
    patches, positions = split_volume_into_patches(
        vol_tensor, 
        vae_config['patch_size'], 
        vae_config['patch_stride']
    )
    
    generated_volumes = []
    
    with torch.no_grad():
        for sample_idx in range(num_samples):
            print(f"\nGenerating sample {sample_idx+1}/{num_samples}...")
            
            generated_patches_list = []
            batch_size = 4
            
            for i in tqdm(range(0, len(patches), batch_size), desc='Processing patches'):
                batch_patches = patches[i:i+batch_size].to(device)
                
                _, skip_features = ldm.encode_patch_to_latent(batch_patches)
                
                modified_skip_features = condition_func(skip_features, **condition_params)
                
                skip_conditioning = ldm.skip_encoder(modified_skip_features)
                
                z = torch.randn(batch_patches.shape[0], ldm.config['latent_dim'], device=device)
                
                for t_idx in reversed(range(ldm.config['timesteps'])):
                    t = torch.full((batch_patches.shape[0],), t_idx, device=device, dtype=torch.long)
                    z = ldm.p_sample(z, t, skip_conditioning)
                    z = torch.clamp(z, -3.0, 3.0)
                
                gen_batch = ldm.decode_latent_to_patch(z, modified_skip_features)
                generated_patches_list.append(gen_batch.cpu())
            
            all_patches = torch.cat(generated_patches_list, 0)
            vol_shape = vol_tensor.shape
            generated_vol = combine_patches_to_volume(all_patches, positions, vol_shape, vae_config['patch_size'])
            
            if generated_vol.dim() == 4:
                generated_vol = generated_vol.unsqueeze(0)
            
            generated_volumes.append(generated_vol)
            
            vol_np = generated_vol[0, 0].cpu().numpy()
            nii_img = nib.Nifti1Image(vol_np, affine=np.eye(4))
            
            condition_name = condition_func.__name__.replace('simulate_', '')
            save_path = f'{output_dir}/{condition_name}_sample_{sample_idx+1}.nii.gz'
            nib.save(nii_img, save_path)
            print(f"Saved: {save_path}")
            
            viz_path = f'{output_dir}/{condition_name}_sample_{sample_idx+1}_midslice.png'
            visualize_midslice(generated_vol, f'{condition_name.title()} Sample {sample_idx+1}', viz_path)
    
    if len(generated_volumes) > 0:
        condition_name = condition_func.__name__.replace('simulate_', '')
        comparison_path = f'{output_dir}/{condition_name}_comparison.png'
        labels = [f'Sample {i+1}' for i in range(len(generated_volumes))]
        visualize_comparison(reference_volume, generated_volumes, labels, comparison_path)
    
    print(f"\nGenerated {num_samples} samples with {condition_func.__name__}")
    return generated_volumes


def generate_condition_progression(ldm, reference_volume, condition_func, 
                                   severity_range=[0.0, 0.25, 0.5, 0.75, 1.0],
                                   output_dir='./output/progression'):
    
    os.makedirs(output_dir, exist_ok=True)
    
    progression_volumes = []
    
    for severity in severity_range:
        print(f"\nGenerating severity={severity:.2f}...")
        
        volumes = generate_with_condition(
            ldm, 
            reference_volume,
            condition_func,
            {'severity': severity} if 'severity' in condition_func.__code__.co_varnames else {'stenosis_severity': severity},
            num_samples=1,
            output_dir=f'{output_dir}/severity_{int(severity*100):03d}'
        )
        
        progression_volumes.append(volumes[0])
    
    condition_name = condition_func.__name__.replace('simulate_', '')
    progression_viz_path = f'{output_dir}/{condition_name}_progression.png'
    visualize_progression(progression_volumes, severity_range, condition_name.title(), progression_viz_path)
    
    print(f"\nGenerated progression series with {len(severity_range)} stages")
    return progression_volumes


def main():
    vae_checkpoint_path = f'{vae_config.get("outPath", "/home/jl_fs/output")}/vae/best_reduced_vae_fullres.pth'
    reference_data_path = vae_config.get('inPath', '/home/jl_fs/dataset/1-200')
    output_base_dir = '/home/jl_fs/output/generated' 
    
    print("\nLoading models")
    
    vae_model = ReducedVAE3D(
        nz=vae_config['nz'],
        input_size=vae_config['patch_size']
    ).to(device)
    
    if os.path.exists(vae_checkpoint_path):
        checkpoint = torch.load(vae_checkpoint_path, map_location=device)
        vae_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"VAE loaded from {vae_checkpoint_path}")
    else:
        print(f"VAE checkpoint not found: {vae_checkpoint_path}")
        return
    
    vae_model.eval()
    
    ldm = LatentDiffusionModel(vae_model, ldm_config, unconditional=False).to(device)
    ldm.eval()
    print(f"LDM initialized")
    
    nii_files = []
    for root, dirs, files in os.walk(reference_data_path):
        for file in files:
            if file.endswith('img.nii.gz') or file.endswith('diao_0.nii'):
                nii_files.append(os.path.join(root, file))
                break
        if nii_files:
            break
    
    if not nii_files:
        print(f"No reference images found in {reference_data_path}")
        return
    
    reference_data = load_and_preprocess_nii([nii_files[0]], target_shape=vae_config['full_volume_shape'])
    reference_volume = torch.tensor(reference_data[0], dtype=torch.float32).unsqueeze(0).to(device)
    print(f"Reference image loaded: {reference_volume.shape}")
    
    os.makedirs(output_base_dir, exist_ok=True)
    ref_nii = nib.Nifti1Image(reference_volume[0, 0].cpu().numpy(), affine=np.eye(4))
    ref_save_path = f'{output_base_dir}/reference.nii.gz'
    nib.save(ref_nii, ref_save_path)
    print(f"Reference image saved: {ref_save_path}")
    
    normal_volumes = generate_normal_variations(
        ldm, 
        reference_volume, 
        num_variations=3,
        output_dir=f'{output_base_dir}/normal_variations'
    )
    
    
    for severity, label in [(0.5, 'moderate'), (0.7, 'severe')]:
        stenosis_volumes = generate_with_condition(
            ldm,
            reference_volume,
            simulate_stenosis,
            {'stenosis_severity': severity, 'location': 'proximal'},
            num_samples=2,
            output_dir=f'{output_base_dir}/stenosis_{label}'
        )
    
    dilated_volumes = generate_with_condition(
        ldm,
        reference_volume,
        simulate_cardiomyopathy,
        {'condition_type': 'dilated', 'severity': 0.4},
        num_samples=2,
        output_dir=f'{output_base_dir}/cardiomyopathy_dilated'
    )
    
    hypertrophic_volumes = generate_with_condition(
        ldm,
        reference_volume,
        simulate_cardiomyopathy,
        {'condition_type': 'hypertrophic', 'severity': 0.4},
        num_samples=2,
        output_dir=f'{output_base_dir}/cardiomyopathy_hypertrophic'
    )
    
    for location in ['anterior', 'inferior', 'lateral']:
        mi_volumes = generate_with_condition(
            ldm,
            reference_volume,
            simulate_myocardial_infarction,
            {'location': location, 'severity': 0.5},
            num_samples=2,
            output_dir=f'{output_base_dir}/myocardial_infarction_{location}'
        )
    
    
    progression = generate_condition_progression(
        ldm,
        reference_volume,
        simulate_stenosis,
        severity_range=[0.2, 0.4, 0.8],
        output_dir=f'{output_base_dir}/stenosis_progression'
    )

if __name__ == '__main__':
    main()
