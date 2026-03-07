"""
Comprehensive Metrics Calculator for CardioGen Paper
Calculates following metrics:
1. VAE Reconstruction Metrics (PSNR, SSIM)
2. Generative Realism (FID)
3. Anti-Memorization Metrics (Nearest Neighbor L2 Distance)
"""

import os
import glob
import numpy as np
import torch
from scipy.linalg import sqrtm
from scipy.ndimage import zoom
from scipy.spatial.distance import cdist
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision.models.video import r3d_18, R3D_18_Weights
from tqdm import tqdm
import json
import sys
from io import StringIO

from vae import ReducedVAE3D, config as vae_config, load_and_preprocess_nii, reconstruct_full_volume_patchwise

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PaperMetricsCalculator:    
    def __init__(self, vae_path, segmentation_path, data_dir, generated_dir, output_dir):
        self.vae_path = vae_path
        self.segmentation_path = segmentation_path
        self.data_dir = data_dir
        self.generated_dir = generated_dir
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.results = {}
        
    def _get_real_files(self, limit=None):
        nii_files = []
        if not os.path.exists(self.data_dir):
            print(f"Warning: Folder {self.data_dir} does not exist!")
            return nii_files
            
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('img.nii.gz') or file.endswith('diao_0.nii'):
                    nii_files.append(os.path.join(root, file))
                    
        if limit and len(nii_files) > limit:
            return nii_files[:limit]
        return nii_files
    
    def calculate_all_metrics(self):    
        # 1. VAE Reconstruction Metrics
        print("\nVAE Reconstruction Metrics (PSNR, SSIM)...")
        vae_metrics = self.calculate_vae_metrics()
        self.results['vae_reconstruction'] = vae_metrics
        
        # 2. Generative Realism (FID)
        print("\nGenerative Realism (FID)...")
        fid_metrics = self.calculate_fid()
        self.results['generative_realism'] = fid_metrics
        
        # 3. Anti-Memorization Metrics
        print("\nAnti-Memorization Metrics (Nearest Neighbor Distance)...")
        novelty_metrics = self.calculate_novelty_metrics()
        self.results['anti_memorization'] = novelty_metrics
        
        
        self.save_results()
        
        return self.results
    
    def calculate_vae_metrics(self):
        print("Loading VAE model...")
        vae = ReducedVAE3D(
            nz=vae_config['nz'],
            input_size=vae_config['patch_size']
        ).to(device)
        
        if os.path.exists(self.vae_path):
            checkpoint = torch.load(self.vae_path, map_location=device)
            vae.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded VAE from epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            print(f"ERROR: VAE model not found at {self.vae_path}")
            return {'psnr': 0, 'ssim': 0, 'error': 'Model not found'}
        
        vae.eval()

        print("Loading test volumes...")
        test_files = self._get_real_files(limit=10)
        
        if len(test_files) == 0:
            print(f"  ERROR: No test files found in {self.data_dir}")
            return {'psnr': 0, 'ssim': 0, 'error': 'No test files'}
        
        print(f"Found {len(test_files)} test files")
        
        psnr_scores = []
        ssim_scores = []
        
        with torch.no_grad():
            for file_path in tqdm(test_files, desc="Processing volumes"):
                try:
                    
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                    try:
                        volume_batch = load_and_preprocess_nii([file_path], target_shape=(512, 512, 275))
                    finally:
                        sys.stdout = old_stdout
                    
                    orig_vol = volume_batch[0]  

                    recon_vol = reconstruct_full_volume_patchwise(
                        vae, orig_vol, vae_config['patch_size'], vae_config['patch_stride']
                    )
                    
                    orig_3d = orig_vol[0]
                    recon_3d = recon_vol[0]
                    
                    psnr_val = psnr(orig_3d, recon_3d, data_range=1.0)
                    psnr_scores.append(psnr_val)
                    
                    ssim_val = ssim(orig_3d, recon_3d, data_range=1.0)
                    ssim_scores.append(ssim_val)
                
                except Exception as e:
                    print(f"Warning: Failed to process {file_path}: {e}")
                    continue
        
        if len(psnr_scores) == 0:
            return {'psnr': 0, 'ssim': 0, 'error': 'No valid patches processed'}
        
        results = {
            'psnr_mean': float(np.mean(psnr_scores)),
            'psnr_std': float(np.std(psnr_scores)),
            'ssim_mean': float(np.mean(ssim_scores)),
            'ssim_std': float(np.std(ssim_scores)),
            'num_patches': len(psnr_scores)
        }
        
        print(f"\nResults:")
        print(f"PSNR: {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
        print(f"SSIM: {results['ssim_mean']:.4f} ± {results['ssim_std']:.4f}")
        
        return results
    
    def _get_real_file_path(self, path):
        if os.path.isdir(path):
            if path.endswith('.img.nii'):
                dia_path = os.path.join(path, 'dia_0.nii')
                if not os.path.exists(dia_path):
                    dia_path = os.path.join(path, '_dia_0.nii')
                if os.path.exists(dia_path):
                    return dia_path
        elif os.path.isfile(path):
            return path
        return None

    def calculate_fid(self):
        """
        Calculate Fréchet Inception Distance (FID) using a pre-trained 3D ResNet feature extractor 
        """
        print("Loading True 3D FID Feature Extractor (R3D-18)...")
        
        model = r3d_18(weights=R3D_18_Weights.DEFAULT).to(device)
        model.fc = torch.nn.Identity() 
        model.eval()

        real_files = self._get_real_files(limit=50)
        generated_files = glob.glob(os.path.join(self.generated_dir, '**/*.nii.gz'), recursive=True)[:50]
        
        if len(real_files) == 0 or len(generated_files) == 0:
            print(f"ERROR: Missing files for FID (Real: {len(real_files)}, Gen: {len(generated_files)})")
            return {'fid': 0, 'error': 'Missing files'}

        def get_deep_embeddings(file_paths, desc):
            embeddings = []
            
            mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1).to(device)
            std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1).to(device)

            for path in tqdm(file_paths, desc=desc):
                try:
                    old_stdout = sys.stdout
                    sys.stdout = StringIO()
                    try:
                        vol_batch = load_and_preprocess_nii([path], target_shape=(256, 256, 128))
                    finally:
                        sys.stdout = old_stdout
                        
                    vol = vol_batch[0, 0] 

                    D, H, W = vol.shape
                    d_start, d_end = D//4, 3*D//4
                    h_start, h_end = H//4, 3*H//4
                    w_start, w_end = W//4, 3*W//4
                    
                    local_patch = vol[d_start:d_end, h_start:h_end, w_start:w_end]
                    
                    zoom_factors = (128/local_patch.shape[0], 128/local_patch.shape[1], 64/local_patch.shape[2])
                    patch_resized = zoom(local_patch, zoom_factors, order=1)
                    
                    x = torch.from_numpy(patch_resized).to(device).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1, 1)
                    
                    x = (x - mean) / std
                    
                    with torch.no_grad():
                        feat = model(x)
                    embeddings.append(feat.cpu().numpy())
                except Exception as e:
                    continue
            return np.concatenate(embeddings, axis=0) if embeddings else np.array([])

        print("Extracting deep features from real volumes...")
        real_features = get_deep_embeddings(real_files, "Real Embeddings")
        
        print("Extracting deep features from generated volumes...")
        gen_features = get_deep_embeddings(generated_files, "Gen Embeddings")

        if real_features.size == 0 or gen_features.size == 0:
            return {'fid': 0, 'error': 'Feature extraction failed'}

        fid_score = self._calculate_fid_from_features(real_features, gen_features)
        
        results = {
            'fid': float(fid_score),
            'num_real': len(real_features),
            'num_generated': len(gen_features)
        }
        
        print(f"\n  Results:")
        print(f"    True 3D FID: {results['fid']:.2f}")
        return results
    
    def _extract_features_batch(self, file_paths, target_shape=(64, 64, 34)):
        features_list = []
        
        for file_path in tqdm(file_paths, desc="  Extracting features"):
            try:
                actual_path = self._get_real_file_path(file_path)
                if actual_path is None:
                    continue

                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                try:
                    volume_batch = load_and_preprocess_nii([actual_path], target_shape=target_shape)
                finally:
                    sys.stdout = old_stdout
                
                img_norm = volume_batch[0, 0]
                
                features = self._extract_statistical_features(img_norm)
                features_list.append(features)
            
            except Exception as e:
                continue
        
        return np.array(features_list)

    def _extract_statistical_features(self, volume):
        features = []

        features.append(np.mean(volume))
        features.append(np.std(volume))
        features.append(np.median(volume))
        features.append(np.percentile(volume, 25))
        features.append(np.percentile(volume, 75))
        
        grad_x = np.gradient(volume, axis=0)
        grad_y = np.gradient(volume, axis=1)
        grad_z = np.gradient(volume, axis=2)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        
        features.append(np.mean(grad_mag))
        features.append(np.std(grad_mag))
        
        features.append(np.mean(volume > 0.3))  
        features.append(np.mean(volume > 0.5))  
        
        return np.array(features)
    
    def _calculate_fid_from_features(self, real_features, gen_features):        
        mu_real = np.mean(real_features, axis=0)
        mu_gen = np.mean(gen_features, axis=0)
        
        sigma_real = np.cov(real_features, rowvar=False)
        sigma_gen = np.cov(gen_features, rowvar=False)
        
        diff = mu_real - mu_gen
        mean_diff = diff.dot(diff)
        
        covmean, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = mean_diff + np.trace(sigma_real) + np.trace(sigma_gen) - 2.0 * np.trace(covmean)
        
        return float(fid)
    
    def _sqrtm_psd(self, matrix):
        eigvals, eigvecs = np.linalg.eigh(matrix)
        
        eigvals = np.maximum(eigvals, 0)
        
        sqrt_eigvals = np.sqrt(eigvals)
        
        return eigvecs.dot(np.diag(sqrt_eigvals)).dot(eigvecs.T)
    
    def calculate_novelty_metrics(self):
        print("Loading real and generated volumes...")
        
        real_files = self._get_real_files(limit=30)
        
        generated_files = glob.glob(os.path.join(self.generated_dir, '**/*.nii.gz'), recursive=True)[:30]
        
        if len(real_files) == 0 or len(generated_files) == 0:
            print(f"ERROR: Need both real and generated files")
            print(f"Real files found: {len(real_files)}")
            print(f"Generated files found: {len(generated_files)}")
            return {'mean_distance': 0, 'error': 'Missing files'}
        
        print(f"Real: {len(real_files)}, Generated: {len(generated_files)}")
        
        print("Extracting features...")
        real_features = self._extract_features_for_distance(real_files)
        gen_features = self._extract_features_for_distance(generated_files)
        
        if len(real_features) == 0 or len(gen_features) == 0:
            print(f"ERROR: No valid features extracted")
            return {'mean_distance': 0, 'error': 'No valid features'}
        
        print("Calculating nearest neighbor distances...")
        distances = []
        
        for gen_feat in tqdm(gen_features, desc="Computing distances"):
            dists = cdist([gen_feat], real_features, metric='euclidean')[0]
            min_dist = np.min(dists)
            distances.append(min_dist)
        
        distances = np.array(distances)
        
        results = {
            'mean_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances)),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances)),
            'median_distance': float(np.median(distances)),
            'num_samples': len(distances)
        }
        
        print(f"\nResults:")
        print(f"Mean L2 Distance: {results['mean_distance']:.4f}")
        print(f"Std L2 Distance: {results['std_distance']:.4f}")
        print(f"Variance: {results['std_distance']**2:.4f}")
        
        return results

    def _extract_features_for_distance(self, file_paths, target_shape=(32, 32, 16)):
        features_list = []
        
        for file_path in tqdm(file_paths, desc="  Loading volumes"):
            try:
                actual_path = self._get_real_file_path(file_path)
                if actual_path is None:
                    continue

                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                try:
                    volume_batch = load_and_preprocess_nii([actual_path], target_shape=target_shape)
                finally:
                    sys.stdout = old_stdout
                
                img_norm = volume_batch[0, 0]
                
                features = img_norm.flatten()
                features_list.append(features)
            
            except Exception as e:
                continue
        
        return np.array(features_list)
    
    def save_results(self):
        output_path = os.path.join(self.output_dir, 'paper_metrics.json')
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def generate_summary_report(self):
        report_path = os.path.join(self.output_dir, 'metrics_summary.txt')
        
        lines = []
        
        vae = self.results.get('vae_reconstruction', {})
        lines.append("1. VAE RECONSTRUCTION QUALITY")
        lines.append("-" * 80)
        lines.append(f"   PSNR: {vae.get('psnr_mean', 0):.2f} ± {vae.get('psnr_std', 0):.2f} dB")
        lines.append(f"   SSIM: {vae.get('ssim_mean', 0):.4f} ± {vae.get('ssim_std', 0):.4f}")
        lines.append("")
        
        fid = self.results.get('generative_realism', {})
        lines.append("2. GENERATIVE REALISM")
        lines.append("-" * 80)
        lines.append(f"   FID Score: {fid.get('fid', 0):.2f}")
        lines.append("")
        
        novelty = self.results.get('anti_memorization', {})
        lines.append("3. ANTI-MEMORIZATION (NOVELTY)")
        lines.append("-" * 80)
        lines.append(f"   Mean NN Distance : {novelty.get('mean_distance', 0):.4f}")
        lines.append(f"   Std NN Distance : {novelty.get('std_distance', 0):.4f}")
        lines.append(f"   Variance : {novelty.get('std_distance', 0)**2:.4f}")
        lines.append("")
        
        seg = self.results.get('segmentation_performance', {})
        lines.append("4. DOWNSTREAM UTILITY (SEGMENTATION)")
        lines.append("-" * 80)
        lines.append(f"   Baseline DSC (Real Only): {seg.get('baseline_dice', 0):.4f}")
        lines.append(f"   Augmented DSC (Real + Synthetic): {seg.get('augmented_dice', 0):.4f}")
        lines.append(f"   Improvement: +{seg.get('improvement', 0):.4f} ({seg.get('improvement', 0)*100:.1f}%)")
        lines.append("")
        
        
        report_text = '\n'.join(lines)
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"\nSummary report saved to: {report_path}")
        print("\n" + report_text)


def main():
    vae_path = '/home/jl_fs/output/vae/best_reduced_vae_fullres.pth'
    segmentation_path = '/home/jl_fs/output/segmentation/best_vnet_coronary.pth'
    data_dir = '/home/jl_fs/dataset/1-200'
    generated_dir = '/home/jl_fs/output/generated' 
    output_dir = '/home/jl_fs/output/paper_metrics'
    
    calculator = PaperMetricsCalculator(
        vae_path=vae_path,
        segmentation_path=segmentation_path,
        data_dir=data_dir,
        generated_dir=generated_dir,
        output_dir=output_dir
    )
    
    results = calculator.calculate_all_metrics()
    
    calculator.generate_summary_report()
    
    print("Complete")

if __name__ == "__main__":
    main()