"""
Latent Diffusion Model for Full Resolution Cardiac Image Generation
Works with patch-based VAE trained on 512,512,275 volumes

Architecture:
1. VAE Encoder: Extract latent representations from patches
2. Diffusion Model: Generate in latent space
3. VAE Decoder: Reconstruct patches and stitch to full resolution
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import math
import nibabel as nib
import cv2

from vae import (
    ReducedVAE3D, 
    pad_or_crop, 
    config as vae_config,
    extract_random_patches,
    split_volume_into_patches,
    combine_patches_to_volume,
    load_nii_files_recursively,
    load_and_preprocess_nii
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ldm_config = {
    'latent_dim': 1024,  # must match VAE latent dimension
    'timesteps': 250,
    'beta_start': 1e-4,
    'beta_end': 0.02,
    'lr': 1e-4,
    'batch_size': 8,
    'epochs': 100,
    'ema_decay': 0.9999,
}


def match_intensity_clahe(generated, reference, clip_limit=1.1, tile_grid_size=(8, 8)):
    gen_np = generated.cpu().numpy()
    ref_np = reference.cpu().numpy()
    
    ref_min, ref_max = ref_np.min(), ref_np.max()
    ref_p1, ref_p99 = np.percentile(ref_np, [1, 99])
    ref_mean = ref_np.mean()
    ref_std = ref_np.std()
    
    print(f"[Intensity Matching] Reference stats: min={ref_min:.4f}, max={ref_max:.4f}, "
          f"p1={ref_p1:.4f}, p99={ref_p99:.4f}, mean={ref_mean:.4f}, std={ref_std:.4f}")
    
    matched_volumes = []
    
    for i in range(gen_np.shape[0]):
        gen_vol = gen_np[i, 0] 
        
        gen_min_orig = gen_vol.min()
        gen_max_orig = gen_vol.max()
        gen_mean_orig = gen_vol.mean()
        
        gen_min, gen_max = gen_vol.min(), gen_vol.max()
        if gen_max > gen_min:
            gen_normalized = (gen_vol - gen_min) / (gen_max - gen_min)
        else:
            gen_normalized = gen_vol.copy()
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_slices = []
        
        for slice_idx in range(gen_normalized.shape[0]):
            slice_2d = gen_normalized[slice_idx]
            slice_uint8 = (slice_2d * 255).astype(np.uint8)
            enhanced_slice = clahe.apply(slice_uint8)
            enhanced_slices.append(enhanced_slice.astype(np.float32) / 255.0)
        
        enhanced_vol = np.stack(enhanced_slices, axis=0)
        
        enhanced_vol = enhanced_vol * (ref_p99 - ref_p1) + ref_p1
        enhanced_vol = np.clip(enhanced_vol, ref_min, ref_max)
        
        gen_min_final = enhanced_vol.min()
        gen_max_final = enhanced_vol.max()
        gen_mean_final = enhanced_vol.mean()
        
        if i == 0:
            print(f"[Intensity Matching] Generated #{i+1}:"
                  f"before=[{gen_min_orig:.4f}, {gen_max_orig:.4f}], mean={gen_mean_orig:.4f} → "
                  f"after=[{gen_min_final:.4f}, {gen_max_final:.4f}], mean={gen_mean_final:.4f}")
        
        matched_volumes.append(enhanced_vol)
    
    matched_np = np.stack(matched_volumes, axis=0)[:, np.newaxis, ...]
    matched_tensor = torch.from_numpy(matched_np).float().to(generated.device)
    
    return matched_tensor


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SkipConnectionEncoder(nn.Module):
    def __init__(self, skip_channels_list, output_dim=512):
        super().__init__()
        self.encoders = nn.ModuleList()
        
        for channels in skip_channels_list:
            self.encoders.append(nn.Sequential(
                nn.AdaptiveAvgPool3d((4, 4, 4)),
                nn.Conv3d(channels, channels // 2, 3, padding=1),
                nn.GroupNorm(8, channels // 2),
                nn.SiLU(),
                nn.Conv3d(channels // 2, channels // 4, 3, padding=1),
                nn.GroupNorm(8, channels // 4),
                nn.SiLU(),
                nn.Flatten(),
            ))
        
        total_features = sum([ch // 4 * 64 for ch in skip_channels_list])
        self.combiner = nn.Sequential(
            nn.Linear(total_features, output_dim * 2),
            nn.SiLU(),
            nn.Linear(output_dim * 2, output_dim),
        )
    
    def forward(self, skip_features):
        """tuple of (x1, x2, x3, x4, x5) from VAE encoder"""
        encoded = []
        for i, skip in enumerate(skip_features):
            encoded.append(self.encoders[i](skip))
        
        combined = torch.cat(encoded, dim=1)
        return self.combiner(combined)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, skip_cond_dim):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
        self.skip_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(skip_cond_dim, out_channels)
        )
        
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Linear(in_channels, out_channels)
        )
        
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(out_channels, out_channels)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Linear(in_channels, out_channels)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x, time_emb, skip_cond):
        h = self.conv1(x)
        
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb
        
        skip_emb = self.skip_mlp(skip_cond)
        h = h + skip_emb
        
        h = self.conv2(h)
        return h + self.shortcut(x)


class LatentDiffusionUNet(nn.Module):
    def __init__(self, latent_dim=1024, time_emb_dim=256, skip_cond_dim=512):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        self.enc1 = ResidualBlock(latent_dim, 1536, time_emb_dim, skip_cond_dim)
        self.enc2 = ResidualBlock(1536, 2048, time_emb_dim, skip_cond_dim)
        self.enc3 = ResidualBlock(2048, 2048, time_emb_dim, skip_cond_dim)
        
        self.bottleneck = nn.Sequential(
            ResidualBlock(2048, 2048, time_emb_dim, skip_cond_dim),
            nn.GroupNorm(8, 2048),
            nn.SiLU(),
        )
        
        self.dec1 = ResidualBlock(2048, 2048, time_emb_dim, skip_cond_dim)
        self.dec2 = ResidualBlock(2048, 1536, time_emb_dim, skip_cond_dim)
        self.dec3 = ResidualBlock(1536, latent_dim, time_emb_dim, skip_cond_dim)
        
        self.out = nn.Sequential(
            nn.GroupNorm(8, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(self, z_noisy, timestep, skip_conditioning):
        t_emb = self.time_mlp(timestep)
        
        h1 = self.enc1(z_noisy, t_emb, skip_conditioning)
        h2 = self.enc2(h1, t_emb, skip_conditioning)
        h3 = self.enc3(h2, t_emb, skip_conditioning)
        
        h = self.bottleneck[0](h3, t_emb, skip_conditioning)
        h = self.bottleneck[1:](h)
        
        h = self.dec1(h + h3, t_emb, skip_conditioning)
        h = self.dec2(h + h2, t_emb, skip_conditioning)
        h = self.dec3(h + h1, t_emb, skip_conditioning)
        
        noise_pred = self.out(h)
        return noise_pred


class LatentDiffusionModel(nn.Module):
    def __init__(self, vae_model, config, unconditional=False):
        super().__init__()
        self.vae = vae_model
        self.config = config
        self.unconditional = unconditional
        
        for param in self.vae.parameters():
            param.requires_grad = False
        self.vae.eval()
        
        skip_channels = [64, 128, 256, 512, 512]
        self.skip_encoder = SkipConnectionEncoder(skip_channels, output_dim=512)
        
        self.unconditional_embedding = nn.Parameter(torch.randn(1, 512))
        
        self.diffusion_model = LatentDiffusionUNet(
            latent_dim=config['latent_dim'],
            time_emb_dim=256,
            skip_cond_dim=512
        )
        
        self.timesteps = config['timesteps']
        self.register_buffer('betas', self._cosine_beta_schedule())
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', 
                           torch.sqrt(1.0 - self.alphas_cumprod))
    
    def _cosine_beta_schedule(self):
        steps = self.timesteps
        s = 0.008
        x = torch.linspace(0, steps, steps + 1)
        alphas_cumprod = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def encode_patch_to_latent(self, patch):
        with torch.no_grad():
            x1 = self.vae.enc1(patch)
            x2 = self.vae.enc2(x1)
            x3 = self.vae.enc3(x2)
            x4 = self.vae.enc4(x3)
            x5 = self.vae.enc5(x4)
            
            flat = x5.view(x5.size(0), -1)
            mu = self.vae.fc_mu(flat)
            
            skip_features = (x1, x2, x3, x4, x5)
            
        return mu, skip_features
    
    def decode_latent_to_patch(self, z, skip_features):
        with torch.no_grad():
            x1, x2, x3, x4, x5 = skip_features
            
            decoded_flat = self.vae.fc_decode(z)
            x5_dec = decoded_flat.view(x5.size())
                        
            d1 = self.vae.dec1(x5_dec)
            x4_cr = pad_or_crop(x4, d1.shape[2:])
            d1_cat = torch.cat([d1, x4_cr], dim=1)
            
            d2 = self.vae.dec2(d1_cat)
            x3_cr = pad_or_crop(x3, d2.shape[2:])
            d2_cat = torch.cat([d2, x3_cr], dim=1)
            
            d3 = self.vae.dec3(d2_cat)
            x2_cr = pad_or_crop(x2, d3.shape[2:])
            d3_cat = torch.cat([d3, x2_cr], dim=1)
            
            d4 = self.vae.dec4(d3_cat)
            x1_cr = pad_or_crop(x1, d4.shape[2:])
            d4_cat = torch.cat([d4, x1_cr], dim=1)
            
            recon_raw = self.vae.dec5(d4_cat)
            recon = self.vae.apply_enhancement_pipeline(recon_raw)
            
        return recon


    def q_sample(self, z_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_alphas_cumprod_t * z_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def p_losses(self, patches, t, conditional=True):
        z_start, skip_features = self.encode_patch_to_latent(patches)
        
        if conditional and not self.unconditional:
            skip_conditioning = self.skip_encoder(skip_features)
        else:
            batch_size = patches.shape[0]
            skip_conditioning = self.unconditional_embedding.repeat(batch_size, 1)
        
        noise = torch.randn_like(z_start)
        
        # Add noise to latent
        z_noisy = self.q_sample(z_start, t, noise)
        
        noise_pred = self.diffusion_model(z_noisy, t, skip_conditioning)
        
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, z, t, skip_conditioning):
        betas_t = self.betas[t].view(-1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / self.alphas[t]).view(-1, 1)
        
        noise_pred = self.diffusion_model(z, t, skip_conditioning)
        
        model_mean = sqrt_recip_alphas_t * (
            z - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] == 0:
            return model_mean
        else:
            alphas_cumprod_t = self.alphas_cumprod[t].view(-1, 1)
            alphas_cumprod_t_prev = self.alphas_cumprod[t - 1].view(-1, 1) if t[0] > 0 else torch.ones_like(alphas_cumprod_t)
            
            posterior_variance_t = betas_t * (1.0 - alphas_cumprod_t_prev) / (1.0 - alphas_cumprod_t)
            posterior_variance_t = torch.clamp(posterior_variance_t, min=1e-20) 
            
            noise = torch.randn_like(z)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample_patches(self, reference_patches, num_samples=1, variation_strength=0.15):
        batch_size = reference_patches.shape[0]
        
        _, skip_features = self.encode_patch_to_latent(reference_patches)
        skip_conditioning = self.skip_encoder(skip_features)
        
        if num_samples > 1:
            skip_conditioning = skip_conditioning.repeat_interleave(num_samples, dim=0)
            skip_features = tuple(
                skip.repeat_interleave(num_samples, dim=0) for skip in skip_features
            )
        
        total_samples = batch_size * num_samples
        
        z = torch.randn(total_samples, self.config['latent_dim'], device=reference_patches.device)
        
        for i in tqdm(reversed(range(self.timesteps)), desc='Sampling patches', total=self.timesteps):
            t = torch.full((total_samples,), i, device=reference_patches.device, dtype=torch.long)
            z = self.p_sample(z, t, skip_conditioning)
            
            z = torch.clamp(z, -3.0, 3.0)
        
        if variation_strength > 0:
            print(f"Applying skip feature variation with strength={variation_strength}")
            varied_skip_features = self._add_skip_feature_variation(
                skip_features, 
                variation_strength=variation_strength
            )

            diff = sum([torch.mean(torch.abs(varied - original)).item() 
                       for varied, original in zip(varied_skip_features, skip_features)])
            print(f"Skip feature difference from original: {diff:.6f}")
        else:
            print(f"No variation applied (variation_strength=0)")
            varied_skip_features = skip_features
        
        generated_patches = self.decode_latent_to_patch(z, varied_skip_features)
        
        return generated_patches
    
    def _add_skip_feature_variation(self, skip_features, variation_strength=0.15):
        
        varied_features = []
        
        for i, skip_feat in enumerate(skip_features):
            feat_std = skip_feat.std(dim=(2, 3, 4), keepdim=True)
            
            kernel_size = 3 if i < 2 else 5
            noise = torch.randn_like(skip_feat)
            
            if kernel_size > 1:
                padding = kernel_size // 2
                noise = F.avg_pool3d(
                    F.pad(noise, (padding,) * 6, mode='replicate'),
                    kernel_size=kernel_size,
                    stride=1
                )
            
            scaled_noise = noise * feat_std * variation_strength
            
            level_scale = 0.5 if i < 2 else 1.0
            varied_feat = skip_feat + scaled_noise * level_scale
            
            varied_features.append(varied_feat)
        
        return tuple(varied_features)
    
    @torch.no_grad()
    def generate_full_volume(self, reference_volume=None, patch_size=(64, 64, 64), 
                            stride=(48, 48, 48), num_variations=1, 
                            target_shape=(512, 512, 275), variation_strength=0.15):
        
        print(f"Generating {num_variations} full resolution volume")
        
        print(f"Conditional generation with variation_strength={variation_strength}")
        vol_tensor = reference_volume.squeeze(0)  # Remove batch dim
        
        patches, positions = split_volume_into_patches(vol_tensor, patch_size, stride)
        
        print(f"  Processing {len(patches)} patches")
        
        generated_patches_list = []
        batch_size = 4
        
        for i in range(0, len(patches), batch_size):
            batch_patches = patches[i:i+batch_size].to(device)
            
            gen_batch = self.sample_patches(
                batch_patches, 
                num_samples=num_variations,
                variation_strength=variation_strength
            )
            
            generated_patches_list.append(gen_batch.cpu())
        
        all_generated_patches = torch.cat(generated_patches_list, 0)
                
        generated_volumes = []
        for var_idx in range(num_variations):
            var_patches = all_generated_patches[var_idx::num_variations]
            
            vol_shape = vol_tensor.shape
            generated_vol = combine_patches_to_volume(var_patches, positions, vol_shape, patch_size)
            
            # generated_vol already has shape (C, D, H, W), we need (1, C, D, H, W) for batch
            if generated_vol.dim() == 4:
                generated_vol = generated_vol.unsqueeze(0)
            
            generated_volumes.append(generated_vol)
        
        generated_volumes = torch.stack(generated_volumes, 0)
        
        if generated_volumes.dim() == 6:
            generated_volumes = generated_volumes.squeeze(1)
        
        print(f"Generated volumes shape: {generated_volumes.shape}")
        
        if reference_volume is not None:
            print(f"Applying intensity matching to reference")
            generated_volumes = match_intensity_clahe(
                generated_volumes, 
                reference_volume,
                clip_limit=1.1,
                tile_grid_size=(8, 8)
            )
            print(f"Intensity matching complete.")
        
        return generated_volumes
    
    
def train_ldm(ldm_model, train_volumes, config, save_path, unconditional_prob=0.5):
    optimizer = optim.AdamW(
        list(ldm_model.skip_encoder.parameters()) + 
        list(ldm_model.diffusion_model.parameters()) +
        [ldm_model.unconditional_embedding],
        lr=config['lr'],
        weight_decay=1e-5
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-6
    )
    
    losses = []
    best_loss = float('inf')
    
    print(f"Device: {device}")
    print(f"Training volumes: {len(train_volumes)}")
    print(f"Batch size: {config['batch_size']} patches")
    print(f"Timesteps: {config['timesteps']}")
    print(f"Epochs: {config['epochs']}")
    
    for epoch in range(config['epochs']):
        ldm_model.train()
        ldm_model.vae.eval()
        
        epoch_loss = 0
        num_batches = 0
        
        all_patches = []
        for vol in train_volumes:
            patches = extract_random_patches(
                vol, 
                vae_config['patch_size'], 
                vae_config['patches_per_volume']
            )
            all_patches.append(patches)
        
        all_patches = torch.cat(all_patches, 0)
        indices = torch.randperm(len(all_patches))
        all_patches = all_patches[indices]
        
        for batch_idx in range(0, len(all_patches), config['batch_size']):
            batch_patches = all_patches[batch_idx:batch_idx+config['batch_size']].to(device)
            
            t = torch.randint(0, config['timesteps'], (batch_patches.shape[0],), device=device).long()
            
            use_conditional = torch.rand(1).item() > unconditional_prob
            
            loss = ldm_model.p_losses(batch_patches, t, conditional=use_conditional)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(ldm_model.skip_encoder.parameters()) + 
                list(ldm_model.diffusion_model.parameters()) +
                [ldm_model.unconditional_embedding],
                max_norm=1.0
            )
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                
                print(f"Epoch {epoch}/{config['epochs']}, Batch {batch_idx}/{len(all_patches)//config['batch_size']}, "
                      f"Loss: {loss.item():.6f}")
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        scheduler.step()
        
        print(f"Epoch {epoch} - Avg Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': {
                    'skip_encoder': ldm_model.skip_encoder.state_dict(),
                    'diffusion_model': ldm_model.diffusion_model.state_dict(),
                },
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, f'{save_path}/best_ldm_fullres.pth')
            print(f"Best model saved (loss: {avg_loss:.6f})")
        
        if epoch % 50 == 0 and epoch > 0:
            print(f"\nGenerating sample at epoch {epoch}")
            ldm_model.eval()
            with torch.no_grad():
                reference = train_volumes[0].unsqueeze(0).to(device)
                generated = ldm_model.generate_full_volume(
                    reference, 
                    patch_size=vae_config['patch_size'],
                    stride=vae_config['patch_stride'],
                    num_variations=2,
                    variation_strength=0.15
                )
                visualize_generation(reference, generated, epoch, save_path)
            ldm_model.train()
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LDM Training Loss')
    plt.grid(True)
    plt.savefig(f'{save_path}/ldm_training_loss.png', dpi=150)
    plt.close()
    
    return ldm_model, losses


def visualize_generation(reference, generated, epoch, save_path):
    ref_np = reference.cpu().numpy()
    gen_np = generated.cpu().numpy()
    
    ref_mid_d = ref_np.shape[2] // 2
    ref_mid_h = ref_np.shape[3] // 2
    ref_mid_w = ref_np.shape[4] // 2
    
    gen_mid_d = gen_np.shape[2] // 2
    gen_mid_h = gen_np.shape[3] // 2
    gen_mid_w = gen_np.shape[4] // 2
    
    ref_mid_d = min(ref_mid_d, ref_np.shape[2] - 1)
    ref_mid_h = min(ref_mid_h, ref_np.shape[3] - 1)
    ref_mid_w = min(ref_mid_w, ref_np.shape[4] - 1)
    
    gen_mid_d = min(gen_mid_d, gen_np.shape[2] - 1)
    gen_mid_h = min(gen_mid_h, gen_np.shape[3] - 1)
    gen_mid_w = min(gen_mid_w, gen_np.shape[4] - 1)
    
    n_samples = min(4, generated.shape[0])
    _, axes = plt.subplots(3, n_samples + 1, figsize=(4 * (n_samples + 1), 12))
    
    axes[0, 0].imshow(ref_np[0, 0, ref_mid_d, :, :], cmap='gray')
    axes[0, 0].set_title('Reference (Axial)')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(ref_np[0, 0, :, ref_mid_h, :], cmap='gray')
    axes[1, 0].set_title('Reference (Coronal)')
    axes[1, 0].axis('off')
    
    axes[2, 0].imshow(ref_np[0, 0, :, :, ref_mid_w], cmap='gray')
    axes[2, 0].set_title('Reference (Sagittal)')
    axes[2, 0].axis('off')
    
    for i in range(n_samples):
        axes[0, i + 1].imshow(gen_np[i, 0, gen_mid_d, :, :], cmap='gray')
        axes[0, i + 1].set_title(f'Generated {i+1} (Axial)')
        axes[0, i + 1].axis('off')
        
        axes[1, i + 1].imshow(gen_np[i, 0, :, gen_mid_h, :], cmap='gray')
        axes[1, i + 1].set_title(f'Generated {i+1} (Coronal)')
        axes[1, i + 1].axis('off')
        
        axes[2, i + 1].imshow(gen_np[i, 0, :, :, gen_mid_w], cmap='gray')
        axes[2, i + 1].set_title(f'Generated {i+1} (Sagittal)')
        axes[2, i + 1].axis('off')
    
    plt.suptitle(f'LDM Full Resolution Generation - Epoch {epoch}')
    plt.tight_layout()
    plt.savefig(f'{save_path}/ldm_generation_epoch_{epoch}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualization saved: ldm_generation_epoch_{epoch}.png")


def compute_metrics(original, generated):
    original = np.clip(original, 0, 1)
    generated = np.clip(generated, 0, 1)
    
    mse = np.mean((original - generated) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    
    orig_mean = np.mean(original)
    orig_std = np.std(original)
    gen_mean = np.mean(generated)
    gen_std = np.std(generated)
    
    return {
        'psnr': psnr, 
        'mse': mse,
        'orig_mean': orig_mean,
        'orig_std': orig_std,
        'gen_mean': gen_mean,
        'gen_std': gen_std
    }


def main():
    vae_checkpoint_path = f'{vae_config.get("outPath", "/home/jl_fs/output/vae")}/best_reduced_vae_fullres.pth'
    output_path = vae_config.get('outPath', '/home/jl_fs/output/diffusion')
    data_path = vae_config.get('inPath', '/home/jl_fs/dataset/1-200')
    
    print("\nLoading trained VAE model")
    vae_model = ReducedVAE3D(
        nz=vae_config['nz'],
        input_size=vae_config['patch_size']
    ).to(device)
    
    if os.path.exists(vae_checkpoint_path):
        checkpoint = torch.load(vae_checkpoint_path, map_location=device)
        vae_model.load_state_dict(checkpoint['model_state_dict'])
        print(f"VAE loaded from {vae_checkpoint_path}")
        print(f"Trained for {checkpoint.get('epoch', 'unknown')} epochs")
        print(f"Best loss: {checkpoint.get('loss', 'unknown'):.6f}")
    else:
        print(f"VAE checkpoint not found at {vae_checkpoint_path}")
        print("Please train VAE first using: python vae.py")
        return
    
    vae_model.eval()
    
    print("\nLoading training data")
    print(f"Looking for .nii files in: {data_path}")
    
    nii_files = load_nii_files_recursively(data_path)
    
    if len(nii_files) == 0:
        print("No .nii files found! Creating dummy data...")
        data = np.random.rand(5, 1, *vae_config['full_volume_shape']).astype(np.float32)
    else:
        print("Loading and preprocessing .nii files at FULL RESOLUTION...")
        data = load_and_preprocess_nii(nii_files, target_shape=vae_config['full_volume_shape'])
    
    train_data = [torch.tensor(vol, dtype=torch.float32) for vol in data]
    print(f"Training data: {len(train_data)} volumes")
    print(f"Volume shape: {train_data[0].shape}")
    
    print("\nCreating Latent Diffusion Model")
    
    print("\nGeneration mode options:")
    print("1. Conditional only (requires reference image)")
    print("2. Unconditional only (generates from scratch)")
    print("3. Both (supports both modes)")
    
    ldm = LatentDiffusionModel(vae_model, ldm_config, unconditional=False).to(device)
   
    total_params = sum(p.numel() for p in ldm.parameters() if p.requires_grad)
    print(f"✓ LDM created with {total_params:,} trainable parameters")
    print(f"  Skip encoder: {sum(p.numel() for p in ldm.skip_encoder.parameters()):,}")
    print(f"  Diffusion model: {sum(p.numel() for p in ldm.diffusion_model.parameters()):,}")
    print(f"  Unconditional embedding: {ldm.unconditional_embedding.numel():,}")
    
    print("\n[Step 4] Training Latent Diffusion Model...")
    print(f"Configuration:")
    print(f"  Epochs: {ldm_config['epochs']}")
    print(f"  Batch size: {ldm_config['batch_size']} patches")
    print(f"  Learning rate: {ldm_config['lr']}")
    print(f"  Timesteps: {ldm_config['timesteps']}")
    print(f"  Patches per volume: {vae_config['patches_per_volume']}")
    
    ldm, _ = train_ldm(ldm, train_data, ldm_config, output_path, unconditional_prob=0.0)
    
    print("\nTraining completed!")
    
    print("\nGenerating full resolution synthetic cardiac images...")
    ldm.eval()
    
    reference_volume = train_data[0].unsqueeze(0).to(device)
    
    print("\nConditional Generation")
    print("Generating 4 variations based on reference volume")
    print("Using variation_strength=0.20 for diversity")
    with torch.no_grad():
        generated_conditional = ldm.generate_full_volume(
            reference_volume,
            patch_size=vae_config['patch_size'],
            stride=vae_config['patch_stride'],
            num_variations=4,
            variation_strength=0.20  
        )
        
    print("\nVisualizing final results")
    visualize_generation(reference_volume, generated_conditional, 'FINAL_CONDITIONAL', output_path)
    
    print("\nComputing quality metrics")
    ref_np = reference_volume.cpu().numpy()
    
    print("\nConditional Generation Metrics")
    for i in range(generated_conditional.shape[0]):
        gen_np = generated_conditional[i:i+1].cpu().numpy()
        metrics = compute_metrics(ref_np, gen_np)
        print(f"Generated Volume {i+1}:")
        print(f"  PSNR: {metrics['psnr']:.2f} dB")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  Original - Mean: {metrics['orig_mean']:.4f}, Std: {metrics['orig_std']:.4f}")
        print(f"  Generated - Mean: {metrics['gen_mean']:.4f}, Std: {metrics['gen_std']:.4f}")
    
    print("\nSaving generated volumes")
    
    for i in range(generated_conditional.shape[0]):
        vol_np = generated_conditional[i, 0].cpu().numpy()
        nii_img = nib.Nifti1Image(vol_np, affine=np.eye(4))
        nii_path = f'{output_path}/generated_conditional_{i+1}.nii.gz'
        nib.save(nii_img, nii_path)
        print(f"  Saved: {nii_path}")

if __name__ == '__main__':
    main()
