"""
CardioGen Synthetic Dataset Generator
Safely generates synthetic variations of real patients while preserving 
exact vessel geometry so the original ground truth labels remain perfectly valid.
"""

import os
import glob
import re
from pathlib import Path
import torch
import nibabel as nib
from tqdm import tqdm
import shutil

from vae import (
    ReducedVAE3D, 
    config as vae_config, 
    split_volume_into_patches, 
    combine_patches_to_volume,
    device
)
from diffusion import LatentDiffusionModel, ldm_config

def get_real_dataset(data_dir):
    all_files = glob.glob(os.path.join(data_dir, '**', '*.nii*'), recursive=True)
    img_map, lbl_map = {}, {}
    
    for fpath in all_files:
        if fpath.endswith('.filepart'): continue
            
        match = re.search(r'patient_?(\d+)', fpath.lower())
        if not match:
            parts = Path(fpath).parts
            for part in reversed(parts):
                nums = re.findall(r'\b\d+\b', part)
                for n in nums:
                    val = int(n)
                    if 0 < val <= 250 and part != "1-200": 
                        match = val
                        break
        pid = int(match.group(1)) if isinstance(match, re.Match) else match
        if pid is None: continue
        
        if 'label' in fpath.lower(): lbl_map[pid] = fpath
        elif 'img' in fpath.lower() or 'dia_0' in fpath.lower(): img_map[pid] = fpath

    matched_ids = sorted(list(set(img_map.keys()) & set(lbl_map.keys())))
    return [{"pid": pid, "image": img_map[pid], "label": lbl_map[pid]} for pid in matched_ids]

def main():
    data_dir = '/home/jl_fs/dataset/1-200'
    output_dir = '/home/jl_fs/output/generated/augmented_dataset'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nLoading Models...")
    vae_model = ReducedVAE3D(nz=vae_config['nz'], input_size=vae_config['patch_size']).to(device)
    vae_model.load_state_dict(torch.load('/home/jl_fs/output/vae/best_reduced_vae_fullres.pth', map_location=device)['model_state_dict'])
    vae_model.eval()
    
    ldm = LatentDiffusionModel(vae_model=vae_model, config=ldm_config).to(device)
    ldm_ckpt = torch.load('/home/jl_fs/output/diffusion/best_ldm_fullres.pth', map_location=device)
    ldm.skip_encoder.load_state_dict(ldm_ckpt['model_state_dict']['skip_encoder'])
    ldm.diffusion_model.load_state_dict(ldm_ckpt['model_state_dict']['diffusion_model'])
    ldm.eval()
    
    print("\nIndexing Real Dataset...")
    real_data = get_real_dataset(data_dir)
    print(f"Found {len(real_data)} valid patient volumes to augment.")

    print("\nGenerating Synthetic Dataset")
    for item in tqdm(real_data, desc="Processing Patients"):
        pid = item['pid']
        img_path = item['image']
        lbl_path = item['label']
        
        pat_out_dir = os.path.join(output_dir, f'synth_patient_{pid}')
        os.makedirs(pat_out_dir, exist_ok=True)
        
        nii_img = nib.load(img_path)
        ref_vol = torch.tensor(nii_img.get_fdata(), dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        patches, positions = split_volume_into_patches(ref_vol.squeeze(0), vae_config['patch_size'], vae_config['patch_stride'])
        
        patches_novel = []
        with torch.no_grad():
            batch_size = 2
            for i in range(0, len(patches), batch_size):
                batch = patches[i:i+batch_size].to(device)
                b_size = batch.shape[0]
                
                _, skip_features = ldm.encode_patch_to_latent(batch)
                
                skip_cond = ldm.skip_encoder(skip_features)
                
                z_novel = torch.randn(b_size, ldm.config['latent_dim'], device=device)
                for t_idx in reversed(range(ldm.config['timesteps'])):
                    t = torch.full((b_size,), t_idx, device=device, dtype=torch.long)
                    z_novel = ldm.p_sample(z_novel, t, skip_cond)
                
                z_novel = torch.clamp(z_novel, -3.0, 3.0)
                dec_novel = ldm.decode_latent_to_patch(z_novel, skip_features)
                patches_novel.append(dec_novel.cpu())
                
        vol_novel = combine_patches_to_volume(torch.cat(patches_novel, 0), positions, ref_vol.squeeze(0).shape, vae_config['patch_size']).numpy()
        
        synth_img_path = os.path.join(pat_out_dir, f'patient_{pid}_img.nii')
        synth_nii = nib.Nifti1Image(vol_novel, affine=nii_img.affine)
        nib.save(synth_nii, synth_img_path)
        
        synth_lbl_path = os.path.join(pat_out_dir, f'patient_{pid}_label.nii')
        shutil.copy2(lbl_path, synth_lbl_path)

    print(f"\nSuccessfully generated {len(real_data)} synthetic volumes")
    print(f"Data saved to: {output_dir}")

if __name__ == '__main__':
    main()