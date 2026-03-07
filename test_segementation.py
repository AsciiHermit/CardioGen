"""
Calculates DSC, HD95, and Sensitivity (Distal Recall) for the trained models.
"""

import os
import glob
import numpy as np
import torch
import nibabel as nib
from tqdm import tqdm

from monai.transforms import Compose, MapTransform, EnsureTyped, AsDiscrete
from monai.networks.nets import VNet
from monai.metrics import DiceMetric, HausdorffDistanceMetric, ConfusionMatrixMetric
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH_SIZE = (64, 64, 64)

class LoadAndFormatData(MapTransform):
    def __call__(self, data):
        d = dict(data)
        img_nii = nib.load(d["image"])
        img_data = np.ascontiguousarray(np.squeeze(img_nii.get_fdata().astype(np.float32)))
        v_min, v_max = -200.0, 800.0
        img_data = np.clip(img_data, v_min, v_max)
        img_t = torch.from_numpy(img_data)
        d["image"] = ((img_t - v_min) / (v_max - v_min)).unsqueeze(0) 

        lbl_data = np.ascontiguousarray(np.squeeze(nib.load(d["label"]).get_fdata().astype(np.float32)))
        lbl_t = torch.from_numpy(lbl_data)
        unique_vals, counts = torch.unique(lbl_t, return_counts=True)
        if len(unique_vals) <= 1:
            d["label"] = torch.zeros_like(lbl_t).unsqueeze(0)
        else:
            bg_val = unique_vals[torch.argmax(counts)]
            d["label"] = (lbl_t != bg_val).float().unsqueeze(0) 
        
        return d

def get_test_files(data_dir):
    real_dicts = []
    real_img_dirs = glob.glob(os.path.join(data_dir, '*.img.nii'))
    for img_dir in real_img_dirs:
        img_path = os.path.join(img_dir, 'dia_0.nii')
        if not os.path.exists(img_path): img_path = os.path.join(img_dir, 'diao_0.nii')
        base_name = os.path.basename(img_dir).replace('.img.nii', '')
        lbl_path = os.path.join(data_dir, f"{base_name}.label.nii", 'label.nii')
        if os.path.exists(img_path) and os.path.exists(lbl_path):
            real_dicts.append({"image": img_path, "label": lbl_path, "pid": base_name})
    
    real_dicts = sorted(real_dicts, key=lambda x: x["image"])
    
    return real_dicts[160:]


def evaluate_model(model_path, test_loader):
    print(f"\nEvaluating: {os.path.basename(model_path)}")
    
    if not os.path.exists(model_path):
        print(f"  -> WARNING: Model not found at {model_path}")
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    model = VNet(spatial_dims=3, in_channels=1, out_channels=2).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE)['model_state_dict'])
    model.eval()

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")

    recall_metric = ConfusionMatrixMetric(include_background=False, metric_name="sensitivity", reduction="mean")
    
    post_pred = AsDiscrete(to_onehot=2)
    post_label = AsDiscrete(to_onehot=2)

    dice_scores, hd95_scores, recall_scores = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Patients"):
            inputs, labels = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            
            outputs = sliding_window_inference(inputs, PATCH_SIZE, 4, model, overlap=0.5)
            preds = torch.argmax(outputs, dim=1, keepdim=True)
            
            preds_onehot = post_pred(preds)
            labels_onehot = post_label(labels.long())
            
            dice_metric(y_pred=preds_onehot, y=labels_onehot)
            dice_scores.append(dice_metric.aggregate().item())
            dice_metric.reset()
            
            hd95_metric(y_pred=preds_onehot, y=labels_onehot)
            hd_val = hd95_metric.aggregate().item()
           
            if not np.isnan(hd_val) and not np.isinf(hd_val):
                hd95_scores.append(hd_val)
            hd95_metric.reset()
            
            recall_metric(y_pred=preds_onehot, y=labels_onehot)
            recall_scores.append(recall_metric.aggregate()[0].item()) 
            recall_metric.reset()

    m_dice = np.mean(dice_scores)
    std_dice = np.std(dice_scores)
    m_hd95 = np.mean(hd95_scores) if hd95_scores else 0.0
    std_hd95 = np.std(hd95_scores) if hd95_scores else 0.0
    m_recall = np.mean(recall_scores) * 100 

    print(f"DSC: {m_dice:.4f} | HD95: {m_hd95:.2f}mm | Recall: {m_recall:.2f}%")
    return m_dice, std_dice, m_hd95, std_hd95, m_recall

def main():
    data_dir = '/home/jl_fs/dataset/1-200'
    model_dir = '/home/jl_fs/output/segmentation'
    
    test_files = get_test_files(data_dir)
    print(f"Found {len(test_files)}  patients.")
    
    test_transforms = Compose([
        LoadAndFormatData(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"])
    ])
    
    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)

    base_path = os.path.join(model_dir, 'best_vnet_baseline.pth')
    b_dice, b_dice_std, b_hd, b_hd_std, b_recall = evaluate_model(base_path, test_loader)
    
    coro_path = os.path.join(model_dir, 'best_vnet_cardiogen_joint.pth')
    c_dice, c_dice_std, c_hd, c_hd_std, c_recall = evaluate_model(coro_path, test_loader)


if __name__ == "__main__":
    main()