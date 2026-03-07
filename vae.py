"""
VAE for 3D Cardiac Image Reconstruction
"""
import numpy as np
import torch
import torch.nn as nn
import math
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import matplotlib.pyplot as plt
import scipy.io
from skimage.io import imsave
import os
import nibabel as nib
from scipy.ndimage import zoom
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

inPath = '/home/jl_fs/dataset/1-200'
outPath = '/home/jl_fs/output'

config = {
    'nz': 1024,
    'lr': 1e-4,
    'batch_size': 8,  
    'epochs': 201,  
    'reconstruction_weight': 15.0,
    'kl_weight': 5e-6,
    'perceptual_weight': 0.05,
    'beta_schedule': True,
    'full_volume_shape': (512, 512, 275),
    'patch_size': (64, 64, 64),  
    'patch_stride': (48, 48, 48),  
    'patches_per_volume': 16,  
    'use_full_resolution': True  
}

def extract_random_patches(volume, patch_size=(64,64,64), num_patches=16):
    C, D, H, W = volume.shape
    pD, pH, pW = patch_size
    patches = []
    
    max_d = max(1, D - pD)
    max_h = max(1, H - pH)
    max_w = max(1, W - pW)
    
    for _ in range(num_patches):
        d_start = np.random.randint(0, max_d) if max_d > 1 else 0
        h_start = np.random.randint(0, max_h) if max_h > 1 else 0
        w_start = np.random.randint(0, max_w) if max_w > 1 else 0
        
        patch = volume[:, d_start:d_start+pD, h_start:h_start+pH, w_start:w_start+pW]
        
        if patch.shape[1:] != patch_size:
            pd, ph, pw = patch.shape[1:]
            pad = (0, max(0, pW-pw), 0, max(0, pH-ph), 0, max(0, pD-pd))
            patch = F.pad(patch, pad)
        
        patches.append(patch)
    
    return torch.stack(patches, 0)

def split_volume_into_patches(volume, patch_size=(64,64,64), stride=None):
    if stride is None:
        stride = patch_size
    
    C, D, H, W = volume.shape
    pD, pH, pW = patch_size
    sD, sH, sW = stride
    patches = []
    positions = []

    for d in range(0, D - pD + 1, sD):
        for h in range(0, H - pH + 1, sH):
            for w in range(0, W - pW + 1, sW):
                patch = volume[:, d:d+pD, h:h+pH, w:w+pW]
                patches.append(patch)
                positions.append((d, h, w))
    
    if D % sD != 0:
        d = D - pD
        for h in range(0, H - pH + 1, sH):
            for w in range(0, W - pW + 1, sW):
                patch = volume[:, d:d+pD, h:h+pH, w:w+pW]
                patches.append(patch)
                positions.append((d, h, w))
    
    if H % sH != 0:
        h = H - pH
        for d in range(0, D - pD + 1, sD):
            for w in range(0, W - pW + 1, sW):
                patch = volume[:, d:d+pD, h:h+pH, w:w+pW]
                patches.append(patch)
                positions.append((d, h, w))
    
    if W % sW != 0:
        w = W - pW
        for d in range(0, D - pD + 1, sD):
            for h in range(0, H - pH + 1, sH):
                patch = volume[:, d:d+pD, h:h+pH, w:w+pW]
                patches.append(patch)
                positions.append((d, h, w))
    
    return torch.stack(patches, 0), positions

def combine_patches_to_volume(patches, positions, vol_shape, patch_size=(64,64,64)):
    C, D, H, W = vol_shape
    pD, pH, pW = patch_size
    
    volume = torch.zeros(vol_shape, dtype=patches.dtype, device=patches.device)
    weight_map = torch.zeros(vol_shape, dtype=patches.dtype, device=patches.device)
    
    gaussian_weight = torch.ones((C, pD, pH, pW), dtype=patches.dtype, device=patches.device)
    
    for i in range(pD):
        weight_d = min(i + 1, pD - i) / (pD / 2)
        for j in range(pH):
            weight_h = min(j + 1, pH - j) / (pH / 2)
            for k in range(pW):
                weight_w = min(k + 1, pW - k) / (pW / 2)
                gaussian_weight[:, i, j, k] = min(weight_d, weight_h, weight_w)
    
    for idx, (d, h, w) in enumerate(positions):
        patch = patches[idx]
        volume[:, d:d+pD, h:h+pH, w:w+pW] += patch * gaussian_weight
        weight_map[:, d:d+pD, h:h+pH, w:w+pW] += gaussian_weight
    
    volume = volume / (weight_map + 1e-8)
    
    return volume

def pad_or_crop(tensor, target_shape):
    _, _, d, h, w = tensor.shape
    td, th, tw = target_shape

    if d > td:
        d1 = (d - td) // 2
        tensor = tensor[:, :, d1:d1+td, :, :]
    elif d < td:
        pad_d1 = (td - d) // 2
        pad_d2 = td - d - pad_d1
        tensor = F.pad(tensor, (0,0,0,0,pad_d1,pad_d2))

    if h > th:
        h1 = (h - th) // 2
        tensor = tensor[:, :, :, h1:h1+th, :]
    elif h < th:
        pad_h1 = (th - h) // 2
        pad_h2 = th - h - pad_h1
        tensor = F.pad(tensor, (0,0,pad_h1,pad_h2,0,0))

    if w > tw:
        w1 = (w - tw) // 2
        tensor = tensor[:, :, :, :, w1:w1+tw]
    elif w < tw:
        pad_w1 = (tw - w) // 2
        pad_w2 = tw - w - pad_w1
        tensor = F.pad(tensor, (pad_w1,pad_w2,0,0,0,0))

    return tensor

class SimpleResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels)
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
        identity = x
        out = self.conv_block(x)
        out += identity
        return self.activation(out)

class ReducedVAE3D(nn.Module):
    def __init__(self, nz=1024, input_size=(64, 64, 64)):
        super().__init__()
        self.nz = nz
        self.input_size = input_size
        
        self.enc1 = nn.Sequential(
            nn.Conv3d(1, 32, 3, stride=1, padding=1), 
            nn.BatchNorm3d(32), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 64, 4, stride=2, padding=1),
            nn.BatchNorm3d(64), nn.LeakyReLU(0.2, inplace=True),
            SimpleResidualBlock3D(64)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(64, 96, 3, stride=1, padding=1),
            nn.BatchNorm3d(96), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(96, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128), nn.LeakyReLU(0.2, inplace=True),
            SimpleResidualBlock3D(128)
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv3d(128, 192, 3, stride=1, padding=1),
            nn.BatchNorm3d(192), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(192, 256, 4, stride=2, padding=1), 
            nn.BatchNorm3d(256), nn.LeakyReLU(0.2, inplace=True),
            SimpleResidualBlock3D(256)
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv3d(256, 384, 3, stride=1, padding=1),
            nn.BatchNorm3d(384), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(384, 512, 4, stride=2, padding=1), 
            nn.BatchNorm3d(512), nn.LeakyReLU(0.2, inplace=True),
            SimpleResidualBlock3D(512)
        )
        
        self.enc5 = nn.Sequential(
            nn.Conv3d(512, 512, 4, stride=2, padding=1), 
            nn.BatchNorm3d(512), nn.LeakyReLU(0.2, inplace=True),
            SimpleResidualBlock3D(512)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *input_size)
            print(f"Model init: dummy_input shape = {dummy_input.shape}")
            x1 = self.enc1(dummy_input)
            print(f"Model init: x1 shape = {x1.shape}")
            x2 = self.enc2(x1)
            print(f"Model init: x2 shape = {x2.shape}")
            x3 = self.enc3(x2)
            print(f"Model init: x3 shape = {x3.shape}")
            x4 = self.enc4(x3)
            print(f"Model init: x4 shape = {x4.shape}")
            x5 = self.enc5(x4)
            print(f"Model init: x5 shape = {x5.shape}")
            flat_dim = x5.view(1, -1).size(1)
            print(f"Model init: calculated flat_dim = {flat_dim}")
        self.flat_dim = flat_dim
        self.bottleneck_shape = x5.shape[1:]

        self.fc_mu = nn.Sequential(
            nn.Linear(self.flat_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(2048, nz),
        )
        self.fc_logvar = nn.Sequential(
            nn.Linear(self.flat_dim, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(2048, nz),
        )
        self.fc_decode = nn.Sequential(
            nn.Linear(nz, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(2048, self.flat_dim),
        )

        self.dec1 = nn.Sequential(
            SimpleResidualBlock3D(512),
            nn.ConvTranspose3d(512, 384, 4, stride=2, padding=1),
            nn.BatchNorm3d(384), nn.ReLU(inplace=True),
            nn.Conv3d(384, 384, 3, stride=1, padding=1), 
            nn.BatchNorm3d(384), nn.ReLU(inplace=True)
        )
        
        self.dec2 = nn.Sequential(
            SimpleResidualBlock3D(896),
            nn.ConvTranspose3d(896, 256, 4, stride=2, padding=1),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, 3, stride=1, padding=1),
            nn.BatchNorm3d(256), nn.ReLU(inplace=True)
        )
        
        self.dec3 = nn.Sequential(
            SimpleResidualBlock3D(512), 
            nn.ConvTranspose3d(512, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm3d(128), nn.ReLU(inplace=True)
        )
        
        self.dec4 = nn.Sequential(
            SimpleResidualBlock3D(256),  
            nn.ConvTranspose3d(256, 64, 4, stride=2, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(inplace=True)
        )
        
        self.dec5 = nn.Sequential(
            SimpleResidualBlock3D(128),
            nn.ConvTranspose3d(128, 32, 4, stride=2, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, 16, 3, stride=1, padding=1),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.Conv3d(16, 1, 3, stride=1, padding=1)
        )
        
        self.sharpening_layer = nn.Sequential(
            nn.Conv3d(1, 8, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, 3, stride=1, padding=1)
        )
        
        self.intensity_correction = nn.Sequential(
            nn.Conv3d(1, 4, 1, stride=1, padding=0),
            nn.Tanh(),
            nn.Conv3d(4, 1, 1, stride=1, padding=0)
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x1 = self.enc1(x)      
        x2 = self.enc2(x1)     
        x3 = self.enc3(x2)     
        x4 = self.enc4(x3)     
        x5 = self.enc5(x4)     

        flat = x5.view(x5.size(0), -1)
        
        if flat.shape[1] != self.flat_dim:
            print(f"ERROR: Dimension mismatch! Expected {self.flat_dim}, got {flat.shape[1]}")
        
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        z = self.reparameterize(mu, logvar)
        decoded_flat = self.fc_decode(z)
        x5_dec = decoded_flat.view(x5.size())

        d1 = self.dec1(x5_dec)
        x4_cr = pad_or_crop(x4, d1.shape[2:])
        d1_cat = torch.cat([d1, x4_cr], dim=1)

        d2 = self.dec2(d1_cat) 
        x3_cr = pad_or_crop(x3, d2.shape[2:])
        d2_cat = torch.cat([d2, x3_cr], dim=1) 

        d3 = self.dec3(d2_cat) 
        x2_cr = pad_or_crop(x2, d3.shape[2:])
        d3_cat = torch.cat([d3, x2_cr], dim=1) 

        d4 = self.dec4(d3_cat) 
        x1_cr = pad_or_crop(x1, d4.shape[2:])
        d4_cat = torch.cat([d4, x1_cr], dim=1)  

        recon_raw = self.dec5(d4_cat) 
        
        recon_enhanced = self.apply_enhancement_pipeline(recon_raw)
        
        return recon_enhanced, mu, logvar
    
    def apply_enhancement_pipeline(self, recon_raw):
        sharpened = recon_raw + 0.3 * self.sharpening_layer(recon_raw)
        
        intensity_corrected = sharpened + 0.15 * self.intensity_correction(sharpened)
        
        recon = torch.clamp(intensity_corrected, 0.0, 1.0)
        
        return recon
    
    def encode_for_generation(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        flat = x5.view(x5.size(0), -1)
        mu = self.fc_mu(flat)
        logvar = self.fc_logvar(flat)
        
        return mu, logvar, (x1, x2, x3, x4, x5)  
    
    
def simplified_loss_function(recon_x, x, mu, logvar, epoch, config):
    batch_size = x.size(0)

    threshold = 0.01
    foreground_mask = (x > threshold).float()

    def sobel_3d(tensor):
        sobel_x = torch.zeros((1,1,3,3,3), device=tensor.device)
        sobel_y = torch.zeros((1,1,3,3,3), device=tensor.device)
        sobel_z = torch.zeros((1,1,3,3,3), device=tensor.device)
        sobel_x[0,0,:,:,:] = torch.tensor([
            [[1,0,-1],[2,0,-2],[1,0,-1]],
            [[2,0,-2],[4,0,-4],[2,0,-2]],
            [[1,0,-1],[2,0,-2],[1,0,-1]]
        ], dtype=torch.float32, device=tensor.device)
        sobel_y[0,0,:,:,:] = torch.tensor([
            [[1,2,1],[0,0,0],[-1,-2,-1]],
            [[2,4,2],[0,0,0],[-2,-4,-2]],
            [[1,2,1],[0,0,0],[-1,-2,-1]]
        ], dtype=torch.float32, device=tensor.device)
        sobel_z[0,0,:,:,:] = torch.tensor([
            [[1,2,1],[2,4,2],[1,2,1]],
            [[0,0,0],[0,0,0],[0,0,0]],
            [[-1,-2,-1],[-2,-4,-2],[-1,-2,-1]]
        ], dtype=torch.float32, device=tensor.device)

        grad_x = F.conv3d(tensor, sobel_x, padding=1)
        grad_y = F.conv3d(tensor, sobel_y, padding=1)
        grad_z = F.conv3d(tensor, sobel_z, padding=1)
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + grad_z ** 2 + 1e-6)
        return grad

    edge_energy = sobel_3d(x)
    edge_weight = 1.0 + 49.0 * (edge_energy > edge_energy.mean()).float()  
    tube_weight = 1.0 + 19.0 * foreground_mask 
    combined_weight = torch.max(edge_weight, tube_weight)

    mse_loss = torch.sum(combined_weight * (recon_x - x) ** 2)
    l1_loss = torch.sum(combined_weight * torch.abs(recon_x - x))
    
    intensity_loss = torch.abs(torch.mean(recon_x) - torch.mean(x)) + \
                    torch.abs(torch.std(recon_x) - torch.std(x))
    
    recon_contrast = torch.std(recon_x)
    target_contrast = torch.std(x)
    contrast_loss = torch.abs(recon_contrast - target_contrast)
    
    recon_loss = 0.6 * mse_loss + 0.2 * l1_loss + 0.1 * intensity_loss + 0.1 * contrast_loss

    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    if config['beta_schedule']:
        beta = min(config['kl_weight'] * (epoch / 4000), config['kl_weight'])
    else:
        beta = config['kl_weight']

    pred_grad = sobel_3d(recon_x)
    target_grad = sobel_3d(x)
    edge_loss = F.l1_loss(pred_grad, target_grad)
    
    def laplacian_3d(tensor):
        laplacian_kernel = torch.tensor([[[
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, -6, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ]]], dtype=torch.float32, device=tensor.device)
        return F.conv3d(tensor, laplacian_kernel, padding=1)
    
    pred_laplacian = laplacian_3d(recon_x)
    target_laplacian = laplacian_3d(x)
    detail_loss = F.l1_loss(pred_laplacian, target_laplacian)
    
    def sharpness_metric(tensor):
        grad = sobel_3d(tensor)
        return torch.mean(grad)
    
    pred_sharpness = sharpness_metric(recon_x)
    target_sharpness = sharpness_metric(x)
    sharpness_loss = torch.abs(pred_sharpness - target_sharpness)

    perceptual = edge_loss + 0.5 * detail_loss + 0.3 * sharpness_loss
    
    total_loss = (config['reconstruction_weight'] * recon_loss / batch_size +
                  beta * kl_loss / batch_size +
                  config['perceptual_weight'] * perceptual)
    
    return total_loss, recon_loss / batch_size, kl_loss / batch_size, perceptual


def reconstruct_full_volume_patchwise(model, volume, patch_size, stride=None):
    model.eval()
    
    if stride is None:
        stride = tuple(p // 2 for p in patch_size) 
    
    vol_tensor = torch.tensor(volume, dtype=torch.float32)
    
    patches, positions = split_volume_into_patches(vol_tensor, patch_size, stride)
    
    print(f"Reconstructing volume {vol_tensor.shape} using {len(patches)} patches...")
    
    reconstructed_patches = []
    with torch.no_grad():
        batch_size = 4
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i+batch_size].to(device)
            rec_batch, _, _ = model(batch_patches)
            reconstructed_patches.append(rec_batch.cpu())
    
    reconstructed_patches = torch.cat(reconstructed_patches, 0)
    
    full_volume = combine_patches_to_volume(reconstructed_patches, positions, vol_tensor.shape, patch_size)
    
    return full_volume.numpy()

def save_visualization(original, reconstructed, epoch, save_path, generated=None):
    original_np = original.cpu().numpy()
    reconstructed_np = reconstructed.cpu().numpy()
    
    initial_depth_dim = original_np.shape[2]
    
    if original_np.shape != reconstructed_np.shape:
        print(f"Warning: Shape mismatch - Original: {original_np.shape}, Reconstructed: {reconstructed_np.shape}")
        
        if len(original_np.shape) == 5 and len(reconstructed_np.shape) == 4:
            print("Adding missing channel dimension to reconstructed")
            reconstructed_np = np.expand_dims(reconstructed_np, axis=1)
        
        if len(original_np.shape) == 5 and len(reconstructed_np.shape) == 5:
            if original_np.shape[2:] != reconstructed_np.shape[2:]:
                print("Resizing spatial dimensions to match")

                reconstructed_tensor = torch.tensor(reconstructed_np)
                reconstructed_tensor = F.interpolate(reconstructed_tensor, size=original_np.shape[2:], mode='trilinear', align_corners=False)
                reconstructed_np = reconstructed_tensor.numpy()
        
        print(f"Fixed shape - Reconstructed: {reconstructed_np.shape}")
    
    mse = np.mean((original_np - reconstructed_np) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    final_depth_dim = original_np.shape[2]
    calculated_mid = final_depth_dim // 2
    
    if calculated_mid >= final_depth_dim:
        mid_slice = final_depth_dim - 1
    else:
        mid_slice = calculated_mid
    
    mid_slice = max(0, mid_slice)

    n_samples = original_np.shape[0]
    n_rows = 3 if generated is not None else 2
    fig, axes = plt.subplots(n_rows, n_samples, figsize=(5 * n_samples, 4 * n_rows))
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=1)
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_samples):
        
        print(f"Sample {i}")
        print(f"Using mid_slice: {mid_slice}")
        print(f"Original shape: {original_np.shape}, valid range: [0, {original_np.shape[2] - 1}]")
        print(f"Reconstructed shape: {reconstructed_np.shape}, valid range: [0, {reconstructed_np.shape[2] - 1}]")
        
        assert 0 <= mid_slice < original_np.shape[2], f"Invalid slice {mid_slice} for original shape {original_np.shape}"
        assert 0 <= mid_slice < reconstructed_np.shape[2], f"Invalid slice {mid_slice} for reconstructed shape {reconstructed_np.shape}"
        
        axes[0, i].imshow(original_np[i, 0, mid_slice], cmap='gray')
        axes[0, i].set_title(f'Original - Sample {i+1}')
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructed_np[i, 0, mid_slice], cmap='gray')
        axes[1, i].set_title(f'Reconstructed - Sample {i+1}')
        axes[1, i].axis('off')
        
        if generated is not None:
            generated_np = generated.cpu().numpy()
            gen_depth_dim = generated_np.shape[2]
            gen_mid_slice = gen_depth_dim // 2
            gen_safe_slice = max(0, min(gen_mid_slice, gen_depth_dim - 1))
            print(f"Generated sample {i}, using slice {gen_safe_slice} for shape {generated_np.shape}")
            
            axes[2, i].imshow(generated_np[i, 0, gen_safe_slice], cmap='gray')
            axes[2, i].set_title(f'Generated - Sample {i+1}')
            axes[2, i].axis('off')

    title = f'Epoch {epoch} - PSNR: {psnr:.2f} dB, MSE: {mse:.6f}'
    if generated is not None:
        generated_np = generated.cpu().numpy()
        if generated_np.shape == original_np.shape:
            gen_mse = np.mean((generated_np - original_np) ** 2)
            gen_psnr = 20 * np.log10(1.0 / np.sqrt(gen_mse)) if gen_mse > 0 else float('inf')
            title += f'\nGenerated PSNR: {gen_psnr:.2f} dB'
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'{save_path}/reference_with_generation_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.close()

def train_reduced_vae_full_volume(trainData, testData, config):
    print(f"Full volume shape: {config['full_volume_shape']}")
    print(f"Patch size: {config['patch_size']}")
    print(f"Patches per volume per epoch: {config['patches_per_volume']}")
    print(f"Batch size: {config['batch_size']}")
    
    train_volumes_full = []
    for i, vol in enumerate(trainData):
        vol_tensor = torch.tensor(vol, dtype=torch.float32)
        print(f"Training volume {i}: shape {vol_tensor.shape}")
        train_volumes_full.append(vol_tensor)
    
    test_volumes_full = []
    for i, vol in enumerate(testData):
        vol_tensor = torch.tensor(vol, dtype=torch.float32)
        print(f"Test volume {i}: shape {vol_tensor.shape}")
        test_volumes_full.append(vol_tensor)
    
    if len(test_volumes_full) == 0:
        print("Warning: Empty test dataset, using training volumes for validation!")
        test_volumes_full = train_volumes_full

    model = ReducedVAE3D(nz=config['nz'], input_size=config['patch_size']).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30)

    train_losses = []
    test_losses = []
    best_loss = float('inf')

    print(f'Training Patch-Based VAE on device: {device}')
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Patch shape: {config["patch_size"]}')

    for epoch in range(config['epochs']):
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        all_patches = []
        for vol in train_volumes_full:
            patches = extract_random_patches(vol, config['patch_size'], config['patches_per_volume'])
            all_patches.append(patches)
        
        all_patches = torch.cat(all_patches, 0)
        num_patches = len(all_patches)
        
        indices = torch.randperm(num_patches)
        all_patches = all_patches[indices]
        
        for batch_idx in range(0, num_patches, config['batch_size']):
            batch_patches = all_patches[batch_idx:batch_idx+config['batch_size']].to(device)
            
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch_patches)
            loss, recon_loss, kl_loss, perceptual_loss = simplified_loss_function(
                recon_batch, batch_patches, mu, logvar, epoch, config)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1

        avg_epoch_loss = epoch_loss / num_batches
        train_losses.append(avg_epoch_loss)
        
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                test_patches = []
                for vol in test_volumes_full[:2]:
                    patches = extract_random_patches(vol, config['patch_size'], 8)
                    test_patches.append(patches)
                test_patches = torch.cat(test_patches, 0).to(device)
                
                test_recon, test_mu, test_logvar = model(test_patches)
                test_loss, test_recon_loss, test_kl_loss, test_perceptual = simplified_loss_function(
                    test_recon, test_patches, test_mu, test_logvar, epoch, config)
                
                test_losses.append(test_loss.item())
                
                print(f'Epoch: {epoch}/{config["epochs"]}')
                print(f'  Train Loss: {avg_epoch_loss:.6f}')
                print(f'  Test Loss: {test_loss.item():.6f}, Recon: {test_recon_loss.item():.6f}, KL: {test_kl_loss.item():.6f}')
                
                if test_loss < best_loss:
                    best_loss = test_loss
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'loss': test_loss,
                        'config': config
                    }, f'{outPath}/best_reduced_vae_fullres.pth')
                    print(f'  *** New best model saved! ***')
        
        scheduler.step(avg_epoch_loss)
        
        if epoch % 20 == 0 and epoch > 0:
            model.eval()
            print(f"\nReconstructing full resolution test volume at epoch {epoch}...")
            
            test_vol_np = test_volumes_full[0].numpy()
            reconstructed_vol = reconstruct_full_volume_patchwise(
                model, test_vol_np, config['patch_size'], config['patch_stride']
            )
            
            mse = np.mean((test_vol_np - reconstructed_vol) ** 2)
            psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
            print(f"Full volume reconstruction PSNR: {psnr:.2f} dB")
            
            orig_tensor = torch.tensor(test_vol_np).unsqueeze(0)
            recon_tensor = torch.tensor(reconstructed_vol).unsqueeze(0)
            save_visualization(orig_tensor, recon_tensor, epoch, outPath)
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss')
            if len(test_losses) > 0:
                test_epochs = [i*5 for i in range(len(test_losses))]
                plt.plot(test_epochs, test_losses, label='Test Loss', marker='o')
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title(f'Full-Resolution Patch-Based VAE Training (PSNR: {psnr:.2f} dB)')
            plt.savefig(f'{outPath}/fullres_loss_curve_epoch_{epoch}.png')
            plt.close()
            
            model.train()

    return model, train_losses, test_losses

def load_nii_files_recursively(folder_path):
    nii_files = []
    if not os.path.exists(folder_path):
        print(f"Warning: Folder {folder_path} does not exist!")
        return nii_files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('img.nii.gz') or file.endswith('diao_0.nii'):
                nii_files.append(os.path.join(root, file))
    nii_files = nii_files[:10]
    print(f"Found {len(nii_files)} .nii files")
    return nii_files

def resize_volume(volume, target_shape=(512, 512, 275)):
    if volume.shape == target_shape:
        return volume
    zoom_factors = [target_shape[i] / volume.shape[i] for i in range(3)]
    resized_volume = zoom(volume, zoom_factors, order=1)
    return resized_volume

def load_and_preprocess_nii(nii_file_paths, target_shape=(512, 512, 275)):
    data_list = []
    successful_loads = 0
    for i, file_path in enumerate(nii_file_paths):
        try:
            print(f"Loading file {i+1}/{len(nii_file_paths)}: {os.path.basename(file_path)}")
            img = nib.load(file_path)
            data = img.get_fdata()
            data = data.astype(np.float32)
            
            if data.ndim == 4:
                data = data[:, :, :, 0]
            elif data.ndim != 3:
                print(f"Skipping file {file_path}: Unexpected dimensions {data.shape}")
                continue
            
            print(f"Original shape: {data.shape}")
            
            data = resize_volume(data, target_shape)
            print(f"Resized to: {data.shape}")
            
            max_val = np.max(data)
            min_val = np.min(data)
            if max_val > min_val:
                data = (data - min_val) / (max_val - min_val)
            
            data = np.expand_dims(data, axis=0)
            data_list.append(data)
            successful_loads += 1
            
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            continue
    
    if len(data_list) == 0:
        raise ValueError("No valid .nii files could be loaded!")
    
    data_array = np.stack(data_list, axis=0)
    print(f"Successfully loaded {successful_loads} files")
    print(f"Final data shape: {data_array.shape}")
    return data_array

if __name__ == "__main__":
    print(f"Target resolution: {config['full_volume_shape']}")
    print(f"Patch size: {config['patch_size']}")
    print(f"Looking for .nii files in: {inPath}")
    
    nii_files = load_nii_files_recursively(inPath)
    
    if len(nii_files) == 0:
        print("No .nii files found! Creating dummy data at full resolution...")
        
        data = np.random.rand(8, 1, *config['full_volume_shape']).astype(np.float32)
    else:
        print("Loading and preprocessing .nii files at FULL RESOLUTION...")
        data = load_and_preprocess_nii(nii_files, target_shape=config['full_volume_shape'])
    
    split_idx = max(1, int(0.8 * data.shape[0]))
    x_train = data[:split_idx]
    x_test = data[split_idx:]
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Total voxels per volume: {np.prod(config['full_volume_shape']):,}")
    print(f"Memory per volume: ~{np.prod(config['full_volume_shape']) * 4 / 1e9:.2f} GB")
    
    print(f"Patches per volume per epoch: {config['patches_per_volume']}")
    print(f"Total patches per epoch: {len(x_train) * config['patches_per_volume']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Expected training time: ~1-2 hours for {config['epochs']} epochs")
    
    model, train_losses, test_losses = train_reduced_vae_full_volume(x_train, x_test, config)
    
    print("\nCompleted ")
    model.eval()
    
    test_vol_np = x_test[0]
    print(f"Reconstructing test volume with shape: {test_vol_np.shape}")
    
    reconstructed_vol = reconstruct_full_volume_patchwise(
        model, test_vol_np, config['patch_size'], config['patch_stride']
    )
    
    mse = np.mean((test_vol_np - reconstructed_vol) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    print(f"Full resolution: {config['full_volume_shape']}")
    print(f"Reconstruction PSNR: {psnr:.2f} dB")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training epochs: {config['epochs']}")
    
    orig_tensor = torch.tensor(test_vol_np).unsqueeze(0)
    recon_tensor = torch.tensor(reconstructed_vol).unsqueeze(0)
    save_visualization(orig_tensor, recon_tensor, "FINAL", outPath)
