"""
Reconstruct Cardiac Images Using Trained VAE
This creates proper input vs. reconstructed pairs for Figure 1
"""

import numpy as np
import torch
import nibabel as nib
import os
from pathlib import Path

from vae import (
    ReducedVAE3D, 
    config as vae_config,
    split_volume_into_patches,
    combine_patches_to_volume,
    load_and_preprocess_nii,
    device
)



def reconstruct_volume(vae_model, input_volume, patch_size=(64, 64, 64), 
                       patch_stride=(48, 48, 48)):
    vae_model.eval()
    
    vol_tensor = input_volume.squeeze(0)  # Remove batch dim
    patches, positions = split_volume_into_patches(
        vol_tensor, 
        patch_size, 
        patch_stride
    )
    
    
    reconstructed_patches = []
    batch_size = 8
    
    with torch.no_grad():
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i+batch_size].to(device)
            
            recon_batch, _, _ = vae_model(batch)
            
            reconstructed_patches.append(recon_batch.cpu())
    
    all_recon_patches = torch.cat(reconstructed_patches, 0)
    vol_shape = vol_tensor.shape
    reconstructed_vol = combine_patches_to_volume(
        all_recon_patches, 
        positions, 
        vol_shape, 
        patch_size
    )
    
    if reconstructed_vol.dim() == 4:
        reconstructed_vol = reconstructed_vol.unsqueeze(0)
    
    print(f"Reconstruction complete: {reconstructed_vol.shape}")
    
    return reconstructed_vol


def main():
    vae_checkpoint_path = f'{vae_config.get("outPath", "/home/jl_fs/output")}/vae/best_reduced_vae_fullres.pth'
    input_data_path = vae_config.get('inPath', '/home/jl_fs/dataset/1-200')
    output_dir = '/home/jl_fs/output/reconstructions'
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("\nLoading trained VAE model")
    
    vae_model = ReducedVAE3D(
        nz=vae_config['nz'],
        input_size=vae_config['patch_size']
    ).to(device)
    
    if os.path.exists(vae_checkpoint_path):
        checkpoint = torch.load(vae_checkpoint_path, map_location=device)
        vae_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"VAE loaded from {vae_checkpoint_path}")
        
        if 'epoch' in checkpoint:
            print(f"Trained for {checkpoint['epoch']} epochs")
        if 'loss' in checkpoint:
            print(f"Final loss: {checkpoint['loss']:.4f}")
    else:
        print(f"VAE checkpoint not found: {vae_checkpoint_path}")
        return
    
    vae_model.eval()
    
    print("\nLoading input cardiac image")
    nii_files = []
    for root, dirs, files in os.walk(input_data_path):
        for file in files:
            if file.endswith('img.nii.gz') or file.endswith('diao_0.nii'):
                nii_files.append(os.path.join(root, file))
                if len(nii_files) >= 3:
                    break
        if len(nii_files) >= 3:
            break
    
    if not nii_files:
        print(f"No input images found in {input_data_path}")
        return
    
    print(f"Found {len(nii_files)} input images")
    
    for idx, nii_file in enumerate(nii_files[:3], 1):
        print("\n" + "="*80)
        print(f"Processing image {idx}/{min(3, len(nii_files))}: {Path(nii_file).name}")
        print("="*80)
        
        input_data = load_and_preprocess_nii([nii_file], 
                                            target_shape=vae_config['full_volume_shape'])
        input_volume = torch.tensor(input_data[0], dtype=torch.float32).unsqueeze(0).to(device)
        
        print(f"Input shape: {input_volume.shape}")
        print(f"Input range: [{input_volume.min():.4f}, {input_volume.max():.4f}]")
        
        reconstructed_volume = reconstruct_volume(
            vae_model, 
            input_volume,
            patch_size=vae_config['patch_size'],
            patch_stride=vae_config['patch_stride']
        )
        
        print(f"Reconstructed shape: {reconstructed_volume.shape}")
        print(f"Reconstructed range: [{reconstructed_volume.min():.4f}, {reconstructed_volume.max():.4f}]")
        
        input_cpu = input_volume.cpu()
        recon_cpu = reconstructed_volume.cpu()
        mse = torch.mean((input_cpu - recon_cpu) ** 2).item()
        print(f"MSE: {mse:.6f}")
        
        orig_nii = nib.Nifti1Image(input_cpu[0, 0].numpy(), affine=np.eye(4))
        orig_path = f'{output_dir}/original_{idx}.nii.gz'
        nib.save(orig_nii, orig_path)
        print(f"Saved original: {orig_path}")
        
        recon_nii = nib.Nifti1Image(recon_cpu[0, 0].numpy(), affine=np.eye(4))
        recon_path = f'{output_dir}/reconstructed_{idx}.nii.gz'
        nib.save(recon_nii, recon_path)
        print(f"Saved reconstructed: {recon_path}")

if __name__ == '__main__':
    main()
