import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm
import argparse


def resize_silhouette_to_96x48(sil_img):
    if sil_img.sum() <= 10000:
        return None
    
    y_sum = sil_img.sum(axis=1)
    y_top = (y_sum != 0).argmax(axis=0)
    y_btm = (y_sum != 0).cumsum(axis=0).argmax(axis=0)
    img = sil_img[y_top: y_btm + 1, :]
    
    ratio = img.shape[1] / img.shape[0]
    target_h = 96
    target_w = int(target_h * ratio)
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    x_csum = img.sum(axis=0).cumsum()
    x_center = None
    for idx, csum in enumerate(x_csum):
        if csum > img.sum() / 2:
            x_center = idx
            break
    
    if x_center is None:
        return None
    
    half_width = 48 // 2
    left = x_center - half_width
    right = x_center + half_width
    
    if left <= 0 or right >= img.shape[1]:
        left += half_width
        right += half_width
        padding = np.zeros((img.shape[0], half_width), dtype=img.dtype)
        img = np.concatenate([padding, img, padding], axis=1)
    
    result = img[:, left:right].astype('uint8')
    
    if result.shape != (96, 48):
        result = cv2.resize(result, (48, 96), interpolation=cv2.INTER_CUBIC)
    
    result[result > 0] = 255
    
    return result


def process_silhouette_sequence(sil_dir, output_dir):
    sil_files = sorted([f for f in os.listdir(sil_dir) 
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    if len(sil_files) == 0:
        return False
    
    sils_96x48 = []
    for sil_file in sil_files:
        sil_path = os.path.join(sil_dir, sil_file)
        sil_img = cv2.imread(sil_path, cv2.IMREAD_GRAYSCALE)
        
        if sil_img is None:
            continue
        
        resized_sil = resize_silhouette_to_96x48(sil_img)
        if resized_sil is not None:
            sils_96x48.append(resized_sil)
    
    if len(sils_96x48) == 0:
        return False
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "01-Sils_96x48.pkl")
    sils_array = np.array(sils_96x48)
    
    with open(output_path, 'wb') as f:
        pickle.dump(sils_array, f)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Resize silhouettes to 96x48 for DenoisingGait')
    parser.add_argument('--input_path', type=str, 
                       default='/data0/aix23907/gait/CCPG_D_MASK_FACE_SHOE',
                       help='Root path of raw silhouette dataset')
    parser.add_argument('--output_path', type=str,
                       default='/data0/aix23907/gait/CCPG_DenoisingGait',
                       help='Output path (should match DenoisingGait output path)')
    args = parser.parse_args()
    
    sequences_processed = 0
    sequences_failed = 0
    
    for id_dir in tqdm(sorted(os.listdir(args.input_path)), desc="Processing IDs"):
        id_path = os.path.join(args.input_path, id_dir)
        if not os.path.isdir(id_path):
            continue
        
        for type_dir in sorted(os.listdir(id_path)):
            type_path = os.path.join(id_path, type_dir)
            if not os.path.isdir(type_path):
                continue
            
            for view_dir in sorted(os.listdir(type_path)):
                view_path = os.path.join(type_path, view_dir)
                if not os.path.isdir(view_path):
                    continue
                
                output_dir = os.path.join(args.output_path, id_dir, type_dir, view_dir)
                
                success = process_silhouette_sequence(view_path, output_dir)
                if success:
                    sequences_processed += 1
                else:
                    sequences_failed += 1
                    tqdm.write(f"Failed: {id_dir}/{type_dir}/{view_dir}")
    
    print(f"\nCompleted!")
    print(f"Successfully processed: {sequences_processed} sequences")
    print(f"Failed: {sequences_failed} sequences")


if __name__ == '__main__':
    main()

