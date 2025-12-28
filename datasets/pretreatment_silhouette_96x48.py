import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count

def resize_silhouette_to_96x48(sil_img):
    if sil_img.sum() <= 10000:
        return None
    
    y_sum = sil_img.sum(axis=1)
    y_indices = np.where(y_sum != 0)[0]
    if len(y_indices) == 0: return None
    y_top, y_btm = y_indices[0], y_indices[-1]
    img = sil_img[y_top: y_btm + 1, :]
    
    ratio = img.shape[1] / img.shape[0]
    target_h = 96
    target_w = int(target_h * ratio)
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    
    x_csum = img.sum(axis=0).cumsum()
    total_sum = img.sum()
    x_center = np.searchsorted(x_csum, total_sum / 2)
    
    half_width = 24
    left = x_center - half_width
    right = x_center + half_width
    
    if left < 0:
        pad_l = np.zeros((96, abs(left)), dtype=img.dtype)
        img = np.concatenate([pad_l, img], axis=1)
        left, right = 0, 48
    if right > img.shape[1]:
        pad_r = np.zeros((96, right - img.shape[1]), dtype=img.dtype)
        img = np.concatenate([img, pad_r], axis=1)
    
    result = img[:, left:right].astype('uint8')
    
    if result.shape != (96, 48):
        result = cv2.resize(result, (48, 96), interpolation=cv2.INTER_NEAREST)
    
    result[result > 0] = 255
    return result

def worker(args):
    sil_dir, output_dir = args
    sil_files = sorted([f for f in os.listdir(sil_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if not sil_files: return False
    
    sils_96x48 = []
    for sil_file in sil_files:
        sil_path = os.path.join(sil_dir, sil_file)
        sil_img = cv2.imread(sil_path, cv2.IMREAD_GRAYSCALE)
        if sil_img is None: continue
        
        resized_sil = resize_silhouette_to_96x48(sil_img)
        if resized_sil is not None:
            sils_96x48.append(resized_sil)
    
    if not sils_96x48: return False
    
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "01-Sils_96x48.pkl"), 'wb') as f:
        pickle.dump(np.array(sils_96x48), f)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='/data0/aix23907/gait/CCPG_G_SIL')
    parser.add_argument('--output_path', type=str, default='/data0/aix23907/gait/CCPG_DenoisingGait')
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    args = parser.parse_args()

    tasks = []
    print("Listing sequences...")
    for id_dir in sorted(os.listdir(args.input_path)):
        id_path = os.path.join(args.input_path, id_dir)
        if not os.path.isdir(id_path): continue
        for ty_dir in sorted(os.listdir(id_path)):
            ty_path = os.path.join(id_path, ty_dir)
            for vi_dir in sorted(os.listdir(ty_path)):
                vi_path = os.path.join(ty_path, vi_dir)
                out_path = os.path.join(args.output_path, id_dir, ty_dir, vi_dir)
                tasks.append((vi_path, out_path))

    print(f"Starting with {args.num_workers} workers...")
    with Pool(args.num_workers) as p:
        results = list(tqdm(p.imap_unordered(worker, tasks), total=len(tasks), desc="Processing"))

    print(f"\nCompleted! Success: {sum(results)}, Failed: {len(results) - sum(results)}")

if __name__ == '__main__':
    main()