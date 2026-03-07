"""
Normal Anatomical Variations Grid
Highlights structural edits made by the generative model using filtered 3D differences.
"""

import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from vae import ReducedVAE3D, config as vae_config, device
from diffusion import LatentDiffusionModel, ldm_config

def get_median_slices(volume):
    vol = volume.squeeze().cpu().numpy() if torch.is_tensor(volume) else volume.squeeze()
    D, H, W = vol.shape
    return vol[:, :, W//2].T, vol[:, H//2, :].T, vol[D//2, :, :].T

def create_filtered_3d_difference(ref_volume, gen_volume, noise_threshold=0.15):
    ref_vol = ref_volume.squeeze().cpu().numpy() if torch.is_tensor(ref_volume) else ref_volume.squeeze()
    gen_vol = gen_volume.squeeze().cpu().numpy() if torch.is_tensor(gen_volume) else gen_volume.squeeze()

    ref_norm = (ref_vol - ref_vol.min()) / (ref_vol.max() - ref_vol.min() + 1e-8)
    gen_norm = (gen_vol - gen_vol.min()) / (gen_vol.max() - gen_vol.min() + 1e-8)
    
    diff_map = np.abs(ref_norm - gen_norm)
    
    diff_map[diff_map < noise_threshold] = 0.0
    diff_map = np.clip(diff_map * 2.5, 0, 1) 
    
    return diff_map

def add_scale_bar(ax, bar_length_mm=40):
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x_start, y_pos = xlim[1] - bar_length_mm - 20, ylim[0] - 20
    ax.plot([x_start, x_start + bar_length_mm], [y_pos, y_pos], color='white', linewidth=3)
    ax.text(x_start + bar_length_mm/2, y_pos - 10, f'{bar_length_mm} mm', 
            color='white', fontsize=8, ha='center', va='top', fontweight='bold')

def main():
    output_dir = '/home/jl_fs/output/figures'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\nLoading models")
    vae_model = ReducedVAE3D(nz=vae_config['nz'], input_size=vae_config['patch_size']).to(device)
    vae_model.load_state_dict(torch.load('/home/jl_fs/output/vae/best_reduced_vae_fullres.pth', map_location=device)['model_state_dict'])
    
    ldm = LatentDiffusionModel(vae_model=vae_model, config=ldm_config).to(device)
    ldm_ckpt = torch.load('/home/jl_fs/output/diffusion/best_ldm_fullres.pth', map_location=device)
    ldm.skip_encoder.load_state_dict(ldm_ckpt['model_state_dict']['skip_encoder'])
    ldm.diffusion_model.load_state_dict(ldm_ckpt['model_state_dict']['diffusion_model'])
    
    ldm.eval()
    vae_model.eval()
    
    ref_data = nib.load('/home/jl_fs/output/generated/reference.nii.gz').get_fdata()
    ref_vol = torch.tensor(ref_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    print("\nGenerating novel anatomy")
    with torch.no_grad():
        gen_vol = ldm.generate_full_volume(
            ref_vol, patch_size=vae_config['patch_size'], stride=vae_config['patch_stride'],
            num_variations=1, variation_strength=0.55
        )
    
    diff_vol_3d = create_filtered_3d_difference(ref_vol, gen_vol)
    
    ref_slices = get_median_slices(ref_vol)
    gen_slices = get_median_slices(gen_vol)
    diff_slices = get_median_slices(diff_vol_3d)
    
    plane_names = ['Axial (Z)', 'Coronal (Y)', 'Sagittal (X)']
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.1, wspace=0.05)
    
    for row, plane in enumerate(plane_names):
        r_img, g_img, d_img = ref_slices[row], gen_slices[row], diff_slices[row]
        r_norm = (r_img - r_img.min()) / (r_img.max() - r_img.min() + 1e-8)
        g_norm = (g_img - g_img.min()) / (g_img.max() - g_img.min() + 1e-8)
        
        ax_ref = fig.add_subplot(gs[row, 0])
        ax_ref.imshow(r_norm, cmap='gray', origin='lower')
        ax_ref.axis('off')
        if row == 0: ax_ref.set_title('Real Reference', fontweight='bold')
        ax_ref.text(-0.1, 0.5, plane, transform=ax_ref.transAxes, fontsize=12, fontweight='bold', rotation=90, va='center', ha='right')
        add_scale_bar(ax_ref)
        
        ax_gen = fig.add_subplot(gs[row, 1])
        ax_gen.imshow(g_norm, cmap='gray', origin='lower')
        ax_gen.axis('off')
        if row == 0: ax_gen.set_title('Generated Sample', fontweight='bold')
        add_scale_bar(ax_gen)
        
        ax_diff = fig.add_subplot(gs[row, 2])
        ax_diff.imshow(r_norm, cmap='gray', origin='lower', alpha=0.4)
        masked_diff = np.ma.masked_where(d_img == 0, d_img)
        im = ax_diff.imshow(masked_diff, cmap='autumn', origin='lower', vmin=0, vmax=1)
        ax_diff.axis('off')
        if row == 0: ax_diff.set_title('Novel Edits (Filtered Diff)', fontweight='bold')
        add_scale_bar(ax_diff)
        
        if row == 1:
            cbar = plt.colorbar(im, ax=ax_diff, fraction=0.046, pad=0.04)
            cbar.set_label('Structural Novelty Intensity', fontsize=10)

    fig.suptitle('Generative Novelty & Structural Editing', fontsize=16, fontweight='bold', y=0.95)
    
    save_path = Path(output_dir) / 'Structural_Novelty.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Figure saved to: {save_path}")

if __name__ == '__main__':
    main()