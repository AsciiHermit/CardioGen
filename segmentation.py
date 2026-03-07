"""
Optimized Pipeline: Robust loading, HU windowing, Focal Loss, and Edge Shaving.
"""

import os
import glob
import numpy as np
import torch
import re

from tqdm import tqdm

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityRanged, 
    CropForegroundd, RandCropByPosNegLabeld, RandFlipd, RandAffined,
    RandGaussianNoised, EnsureTyped, AsDiscrete, MapTransform, KeepLargestConnectedComponent
)
from monai.networks.nets import VNet
from monai.losses import DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = (96, 96, 96) 
BATCH_SIZE = 2 
EPOCHS = 100
LEARNING_RATE = 1e-4
VALIDATION_INTERVAL = 10

class ShaveEdgesd(MapTransform):
    def __init__(self, keys, crop_pct=0.10, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.crop_pct = crop_pct

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if key in d:
                img = d[key] 
                
                d_dim, h_dim, w_dim = img.shape[1], img.shape[2], img.shape[3]
                
                d_crop = int(d_dim * self.crop_pct)
                h_crop = int(h_dim * self.crop_pct)
                w_crop = int(w_dim * self.crop_pct)
                
                d[key] = img[:, 
                             d_crop : d_dim - d_crop, 
                             h_crop : h_dim - h_crop, 
                             w_crop : w_dim - w_crop]
        return d

class IsolateArteryLabeld(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            lbl_t = d[key]
            unique_vals, counts = torch.unique(lbl_t, return_counts=True)
            
            if len(unique_vals) <= 1:
                d[key] = torch.zeros_like(lbl_t)
            else:
                bg_val = unique_vals[torch.argmax(counts)]
                d[key] = (lbl_t != bg_val).float()
        return d

def get_datasets(data_dir, synth_dir):
    real_dicts = []
    real_img_dirs = glob.glob(os.path.join(data_dir, '*.img.nii'))
    for img_dir in real_img_dirs:
        img_path = os.path.join(img_dir, 'dia_0.nii')
        if not os.path.exists(img_path): img_path = os.path.join(img_dir, 'diao_0.nii')
        base_name = os.path.basename(img_dir).replace('.img.nii', '')
        lbl_path = os.path.join(data_dir, f"{base_name}.label.nii", 'label.nii')
        
        pid_match = re.search(r'(?:patient_)?(\d+)', base_name.lower())
        pid = pid_match.group(1) if pid_match else base_name
            
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            real_dicts.append({"image": img_path, "label": lbl_path, "type": "real", "pid": pid})
    
    real_dicts = sorted(real_dicts, key=lambda x: x["image"])

    synth_dicts = []
    synth_images = glob.glob(os.path.join(synth_dir, '**', '*_img.nii*'), recursive=True)
    for s_img in synth_images:
        s_lbl = s_img.replace('_img.', '_label.')
        
        pid_match = re.search(r'patient_(\d+)', s_img.lower())
        pid = pid_match.group(1) if pid_match else None
        
        if os.path.exists(s_lbl) and pid:
            synth_dicts.append({"image": s_img, "label": s_lbl, "type": "synth", "pid": pid})
            
    if len(real_dicts) == 0:
        raise ValueError(f"No real images found in {data_dir}. Check paths!")

    return real_dicts, synth_dicts

def train_model(model_name, train_files, val_files, output_dir):

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),

        ShaveEdgesd(keys=["image", "label"], crop_pct=0.30),
        
        ScaleIntensityRanged(keys=["image"], a_min=50.0, a_max=600.0, b_min=0.0, b_max=1.0, clip=True),
        IsolateArteryLabeld(keys=["label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label", spatial_size=PATCH_SIZE, 
            pos=5, neg=1, num_samples=4, image_key="image", image_threshold=0
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=[0, 1, 2]),
        RandAffined(keys=["image", "label"], prob=0.3, rotate_range=0.2, translate_range=5, mode=('bilinear', 'nearest')),
        RandGaussianNoised(keys=["image"], prob=0.5, mean=0.0, std=0.05),
        EnsureTyped(keys=["image", "label"])
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        
        ShaveEdgesd(keys=["image", "label"], crop_pct=0.30),
        
        ScaleIntensityRanged(keys=["image"], a_min=50.0, a_max=600.0, b_min=0.0, b_max=1.0, clip=True),
        IsolateArteryLabeld(keys=["label"]),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        EnsureTyped(keys=["image", "label"])
    ])

    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1)

    model = VNet(spatial_dims=3, in_channels=1, out_channels=2).to(DEVICE)
    
    loss_function = DiceFocalLoss(to_onehot_y=True, softmax=True, include_background=False, squared_pred=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    
    post_clean = KeepLargestConnectedComponent(applied_labels=[1], num_components=3)
    post_pred, post_label = AsDiscrete(to_onehot=2), AsDiscrete(to_onehot=2)

    best_metric = -1
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False):
            inputs, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        scheduler.step(epoch_loss)

        if (epoch + 1) % VALIDATION_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                for v_batch in val_loader:
                    v_in, v_lbl = v_batch["image"].to(DEVICE), v_batch["label"].to(DEVICE)
                    v_out = sliding_window_inference(v_in, PATCH_SIZE, 4, model, overlap=0.5)
                    v_pre = torch.argmax(v_out, dim=1, keepdim=True)

                    v_pre = post_clean(v_pre) 

                    dice_metric(y_pred=post_pred(v_pre), y=post_label(v_lbl.long()))
                    hd95_metric(y_pred=post_pred(v_pre), y=post_label(v_lbl.long()))
                
                metric_dice = dice_metric.aggregate().item()
                metric_hd95 = hd95_metric.aggregate().item()
                
                dice_metric.reset()
                hd95_metric.reset()
                
                print(f"Epoch {epoch+1} REAL Val -> Dice: {metric_dice:.4f} | HD95: {metric_hd95:.2f}mm")
                
                if metric_dice > best_metric:
                    best_metric = metric_dice
                    torch.save({'model_state_dict': model.state_dict()}, os.path.join(output_dir, f"{model_name}.pth"))
                    print(f"  => Saved New Best {model_name}")

    print(f"FINISH: {model_name} | Best Dice: {best_metric:.4f}")

def main():
    data_dir = '/home/jl_fs/dataset/1-200'
    synth_dir = '/home/jl_fs/output/generated/augmented_dataset'
    out_dir = '/home/jl_fs/output/segmentation'
    os.makedirs(out_dir, exist_ok=True)
    
    real_files, synth_files = get_datasets(data_dir, synth_dir)
    
    total_files = len(real_files)
    
    if total_files >= 160:
        train_real = real_files[:140]
        val_real = real_files[140:160]
    else:
        print(f"\nWARNING: Only found {total_files} real patients. Falling back to 80/20 split.")
        split_idx = max(1, int(total_files * 0.8)) 
        train_real = real_files[:split_idx]
        val_real = real_files[split_idx:]
    
    safe_train_pids = {item["pid"] for item in train_real}
    
    safe_synth_files = [item for item in synth_files if item["pid"] in safe_train_pids]
    
    print(f"Total Real Train: {len(train_real)} | Valid Synth Augmentations: {len(safe_synth_files)}")
    print(f"Total Held-Out Val: {len(val_real)}")
    
    train_model("best_vnet_baseline", train_real, val_real, out_dir)
    
    train_joint = train_real + safe_synth_files
    train_model("best_vnet_cardiogen_joint", train_joint, val_real, out_dir)

if __name__ == "__main__":
    main()