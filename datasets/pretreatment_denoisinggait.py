import os
import pickle
import numpy as np
import cv2
import torch
from tqdm import tqdm
import argparse
from multiprocessing import Pool
import torch.multiprocessing as mp

try:
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
except ImportError:
    print("Error: diffusers library required.")
    print("Install: pip install diffusers transformers accelerate")
    raise


def upscale_image(img_array, target_size=(768, 384)):
    img = np.transpose(img_array, (1, 2, 0))
    
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
    
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    upscaled = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    upscaled = cv2.cvtColor(upscaled, cv2.COLOR_BGR2RGB)
    upscaled = np.transpose(upscaled, (2, 0, 1))
    
    return upscaled


def extract_diffusion_features(rgb_images, vae, unet, scheduler, device, timestep=700, batch_size=8, desc="", gpu_position=1):
    all_predicted_noise = []
    num_frames = rgb_images.shape[0]
    current_batch_size = batch_size
    
    i = 0
    while i < num_frames:
        end_idx = min(i + current_batch_size, num_frames)
        batch_images = rgb_images[i:end_idx]
        
        try:
            images_tensor = torch.from_numpy(batch_images).float().to(device)
            images_tensor = images_tensor / 255.0 * 2.0 - 1.0
            
            with torch.no_grad():
                latents = vae.encode(images_tensor).latent_dist.mean
                latents = latents * vae.config.scaling_factor
                
                batch_size_actual = latents.shape[0]
                noise = torch.randn(latents.shape, device=device, dtype=latents.dtype)
                
                timesteps = torch.full(
                    (batch_size_actual,), 
                    timestep, 
                    device=device, 
                    dtype=torch.long
                )
                noisy_latents = scheduler.add_noise(latents, noise, timesteps)
                
                encoder_hidden_states = torch.zeros(
                    (batch_size_actual, 77, 768), device=device, dtype=latents.dtype
                )
                
                predicted_noise = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                ).sample
            
            all_predicted_noise.append(predicted_noise.cpu().numpy())
            
            del images_tensor, latents, noise, noisy_latents, predicted_noise
            torch.cuda.empty_cache()
            
            i = end_idx
            current_batch_size = batch_size
            
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current_batch_size == 1:
                raise RuntimeError("OOM even with batch_size=1")
            current_batch_size = max(1, current_batch_size // 2)
            tqdm.write(f"OOM detected, reducing batch_size to {current_batch_size}", file=None)
    
    return np.concatenate(all_predicted_noise, axis=0)


def process_sequence(input_pkl_path, output_dir, vae, unet, scheduler, device, timestep=700, batch_size=8, seq_name="", gpu_id=0):
    gpu_position = 1 + gpu_id
    
    output_path = os.path.join(output_dir, "00-DenoisingFea_96x48.pkl")
    if os.path.exists(output_path):
        return True
    
    with open(input_pkl_path, 'rb') as f:
        rgb_256x128 = pickle.load(f)
    
    if len(rgb_256x128) == 0:
        return False
    
    num_frames = len(rgb_256x128)
    
    rgb_768x384_list = []
    for frame in tqdm(rgb_256x128, desc=f"GPU{gpu_id} Upscale", leave=False, position=gpu_position, ncols=80):
        upscaled = upscale_image(frame, target_size=(768, 384))
        rgb_768x384_list.append(upscaled)
    
    rgb_768x384 = np.array(rgb_768x384_list)
    
    predicted_noise = extract_diffusion_features(
        rgb_768x384, vae, unet, scheduler, device, timestep, batch_size, 
        desc=f"GPU{gpu_id} Diffusion", gpu_position=gpu_position
    )
    
    # UNet outputs (frames, 4, 48, 96) = (frames, channels, height, width)
    # DenoisingGait expects (frames, 4, 96, 48) = (frames, channels, height, width)
    # Transpose: (frames, 4, 48, 96) -> (frames, 4, 96, 48)
    predicted_noise = predicted_noise.transpose(0, 1, 3, 2)
    
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(predicted_noise, f)
    
    return True


def process_worker(args_tuple):
    id_dir, type_dir, view_dir, input_pkl_path, output_path, sd_model_path, timestep, batch_size, gpu_id = args_tuple
    
    device = torch.device(f'cuda:{gpu_id}')
    gpu_position = 1 + gpu_id
    
    try:
        vae = AutoencoderKL.from_pretrained(sd_model_path, subfolder="vae").to(device)
        unet = UNet2DConditionModel.from_pretrained(sd_model_path, subfolder="unet").to(device)
        scheduler = DDPMScheduler.from_pretrained(sd_model_path, subfolder="scheduler")
        
        vae.eval()
        unet.eval()
        
        output_dir = os.path.join(output_path, id_dir, type_dir, view_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        seq_name = f"{id_dir}/{type_dir}/{view_dir}"
        
        success = process_sequence(
            input_pkl_path, output_dir, vae, unet, scheduler, 
            device, timestep, batch_size, seq_name, gpu_id
        )
        
        del vae, unet, scheduler
        torch.cuda.empty_cache()
        
        return (id_dir, type_dir, view_dir, success)
    except Exception as e:
        tqdm.write(f"[GPU {gpu_id}] ERROR: {id_dir}/{type_dir}/{view_dir} - {e}", file=None)
        return (id_dir, type_dir, view_dir, False)


def main():
    parser = argparse.ArgumentParser(
        description='Extract Diffusion features for DenoisingGait (2-stage preprocessing)'
    )
    parser.add_argument('--input_path', type=str, 
                       default='/data0/aix23907/gait/CCPG_RGB_256x128',
                       help='Path to 256x128 aligned RGB pkl files')
    parser.add_argument('--output_path', type=str, 
                       default='/data0/aix23907/gait/CCPG_DenoisingGait',
                       help='Output path for DenoisingGait data')
    parser.add_argument('--sd_model_path', type=str, 
                       default='runwayml/stable-diffusion-v1-5',
                       help='Stable Diffusion model path')
    parser.add_argument('--timestep', type=int, default=700,
                       help='Timestep for noise prediction (default: 700)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for processing frames (default: 16)')
    parser.add_argument('--num_gpus', type=int, default=8,
                       help='Number of GPUs to use (default: 8)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("DenoisingGait 2-stage Preprocessing: Diffusion Feature Extraction")
    print("=" * 60)
    print(f"Input path: {args.input_path}")
    print(f"Output path: {args.output_path}")
    print(f"SD Model: {args.sd_model_path}")
    print(f"Timestep: {args.timestep}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of GPUs: {args.num_gpus}")
    print("=" * 60)
    
    all_sequences = []
    for id_dir in sorted(os.listdir(args.input_path)):
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
                
                pkl_files = [f for f in os.listdir(view_path) 
                            if f.endswith('-aligned-rgbs.pkl')]
                
                if pkl_files:
                    all_sequences.append((
                        id_dir, type_dir, view_dir, 
                        os.path.join(view_path, pkl_files[0])
                    ))
    
    print(f"\nTotal sequences: {len(all_sequences)}")
    
    worker_args = []
    for idx, (id_dir, type_dir, view_dir, input_pkl_path) in enumerate(all_sequences):
        gpu_id = idx % args.num_gpus
        worker_args.append((
            id_dir, type_dir, view_dir, input_pkl_path, args.output_path,
            args.sd_model_path, args.timestep, args.batch_size, gpu_id
        ))
    
    mp.set_start_method('spawn', force=True)
    processed_count = 0
    error_count = 0
    
    print("\n" + "=" * 60)
    print("GPU Status:")
    for gpu_id in range(args.num_gpus):
        print(f"  GPU {gpu_id}: Ready")
    print("=" * 60 + "\n")
    
    with Pool(processes=args.num_gpus) as pool:
        results = list(tqdm(
            pool.imap(process_worker, worker_args),
            total=len(worker_args),
            desc="Processing all sequences",
            position=0,
            ncols=100
        ))
    
    for _, _, _, success in results:
        if success:
            processed_count += 1
        else:
            error_count += 1
    
    print("\n" + "=" * 60)
    print("Processing completed!")
    print(f"Successfully processed: {processed_count} sequences")
    print(f"Errors: {error_count} sequences")
    print("=" * 60)


if __name__ == '__main__':
    main()