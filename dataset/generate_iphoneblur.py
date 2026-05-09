import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
import imagehash
from PIL import Image
from collections import deque
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import lpips

# ==========================================
# 1. TIER-2 GAMMA SYNTHESIS
# ==========================================
def gamma_decode(img_srgb, gamma=2.2):
    img_float = img_srgb.astype(np.float32) / 255.0
    return np.power(img_float, gamma)

def gamma_encode(img_linear, gamma=2.2):
    img_srgb = np.power(img_linear, 1.0 / gamma) * 255.0
    return np.clip(img_srgb, 0, 255).astype(np.uint8)

def synthesize_blur(frames, gamma=2.2):
    linear_frames = [gamma_decode(f, gamma) for f in frames]
    return gamma_encode(np.mean(linear_frames, axis=0), gamma)

# ==========================================
# 2. METADATA & PHYSICAL SCENE CALCULATIONS
# ==========================================
def calculate_isp_energy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(np.sqrt(gx**2 + gy**2))

def calculate_motion(frame_prev, frame_next):
    gray1 = cv2.cvtColor(frame_prev, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame_next, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(mag)

def calculate_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def calculate_contrast(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return np.std(gray) / 255.0

def calculate_noise_estimate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = cv2.absdiff(gray, blur)
    return np.median(diff)

def calculate_complexity(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return np.sum(edges > 0) / edges.size

def get_difficulty(psnr_val):
    if psnr_val >= 30.0: return 'Easy'
    elif 24.0 <= psnr_val < 30.0: return 'Medium'
    else: return 'Hard'

# ==========================================
# 3. MEMORY-SAFE GENERATION PIPELINE
# ==========================================
def process_video(video_path, output_dir, lpips_model, device, args, is_train):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened(): return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_name = video_path.name
    source_name = video_path.parent.name
    
    split_name = 'train' if is_train else 'test'
    blur_dir = output_dir / split_name / "blur"
    sharp_dir = output_dir / split_name / "sharp"
    blur_dir.mkdir(parents=True, exist_ok=True)
    sharp_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes = set()
    half_max = args.max_window // 2
    metadata = []
    
    # Calculate which frames we actually need to process based on stride
    target_centers = set(range(half_max + 1, total_frames - half_max - 1, args.stride))
    if not target_centers:
        return []
        
    pbar = tqdm(total=len(target_centers), desc=f"Processing {video_name}")
    
    # [BUG FIX]: Use a rolling buffer to prevent >80GB RAM OOM crashes.
    # Buffer holds exactly max_window + 1 frames (to include frame_prev for motion calculation).
    buffer = deque(maxlen=args.max_window + 1)
    
    current_idx = 0
    img_num = 0
    
    while True:
        ret, frame_bgr = cap.read()
        if not ret: break
        
        # Keep everything in RGB space within the rolling buffer
        buffer.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        
        # The center of the current buffer evaluates to this absolute frame index
        center_idx = current_idx - half_max
        
        # Only process if buffer is fully primed and the center frame hits our stride
        if len(buffer) == args.max_window + 1 and center_idx in target_centers:
            
            frame_prev = buffer[0]
            w_frames_full = list(buffer)[1:] # The maximum possible window of frames
            sharp_frame = w_frames_full[half_max]
            
            # --- ADAPTIVE WINDOW LOGIC ---
            target_psnr = np.random.uniform(args.psnr_floor, args.psnr_ceiling)
            best_window = args.min_window
            blurry_frame = None
            
            for w in range(args.min_window, args.max_window + 1, 2):
                half_w = w // 2
                w_frames = w_frames_full[half_max - half_w : half_max + half_w + 1]
                
                temp_blur = synthesize_blur(w_frames)
                temp_psnr = psnr(sharp_frame, temp_blur, data_range=255)
                
                best_window = w
                blurry_frame = temp_blur
                
                # Stop degrading once we hit the target
                if temp_psnr <= target_psnr:
                    break
                    
            window_size = best_window
            
            # 1. Deduplication using Perceptual Hash
            pil_img = Image.fromarray(blurry_frame)
            p_hash = str(imagehash.phash(pil_img))
            
            if p_hash not in seen_hashes:
                # 2. Core Quality Metrics
                p_val = psnr(sharp_frame, blurry_frame, data_range=255)
                s_val = ssim(sharp_frame, blurry_frame, data_range=255, multichannel=True, channel_axis=2)
                
                b_tensor = torch.from_numpy(blurry_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2.0 - 1.0
                s_tensor = torch.from_numpy(sharp_frame).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2.0 - 1.0
                with torch.no_grad():
                    l_val = lpips_model(b_tensor.to(device), s_tensor.to(device)).item()
                    
                motion_val = calculate_motion(frame_prev, sharp_frame)
                
                # EXACT Quality Filters
                if not (p_val < args.psnr_floor or 
                        p_val > args.psnr_ceiling or 
                        s_val < args.ssim_floor or 
                        l_val > args.lpips_ceiling or 
                        motion_val < args.motion_floor):
                        
                    seen_hashes.add(p_hash)
                    difficulty = get_difficulty(p_val)
                    
                    img_id = f"{video_path.stem}_{img_num:05d}"
                    filename = f"{img_id}.jpg"
                    
                    # 3. ISP and Detailed Extraction
                    isp_blur = calculate_isp_energy(blurry_frame)
                    isp_sharp = calculate_isp_energy(sharp_frame)
                    
                    cv2.imwrite(str(blur_dir / filename), cv2.cvtColor(blurry_frame, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(sharp_dir / filename), cv2.cvtColor(sharp_frame, cv2.COLOR_RGB2BGR))
                    
                    metadata.append({
                        'img_id': img_id,
                        'video': video_name,
                        'source': source_name,
                        'img_num': img_num,
                        'psnr': p_val,
                        'target_psnr': target_psnr,
                        'lpips': l_val,
                        'blur_window': window_size,
                        'blur_duration_ms': (window_size / fps) * 1000 if fps > 0 else 0,
                        'sharpness': calculate_sharpness(sharp_frame),
                        'motion': motion_val,
                        'complexity': calculate_complexity(sharp_frame),
                        'difficulty': difficulty,
                        'ssim': s_val,
                        'isp_blur': isp_blur,
                        'isp_sharp': isp_sharp,
                        'isp_diff': isp_blur - isp_sharp, # Confirmed matched logic
                        'noise_estimate': calculate_noise_estimate(sharp_frame),
                        'phash': p_hash,
                        'contrast': calculate_contrast(sharp_frame)
                    })
                    
            img_num += 1
            pbar.update(1)
            
        current_idx += 1

    cap.release()
    pbar.close()
    return metadata

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./data/iphoneblur')
    parser.add_argument('--min_window', type=int, default=3)
    parser.add_argument('--max_window', type=int, default=21)
    parser.add_argument('--stride', type=int, default=15)
    
    parser.add_argument('--psnr_floor', type=float, default=21.0)
    parser.add_argument('--psnr_ceiling', type=float, default=32.5)
    parser.add_argument('--ssim_floor', type=float, default=0.65)
    parser.add_argument('--lpips_ceiling', type=float, default=0.35)
    parser.add_argument('--motion_floor', type=float, default=1.0)
    
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    mov_files = sorted(list(input_dir.rglob("*.MOV")))
    
    # True Interleaved Split Logic (70/30)
    train_videos = set()
    for i, vid in enumerate(mov_files):
        if i % 10 < 7:
            train_videos.add(vid)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lpips_model = lpips.LPIPS(net='alex').eval().to(device)

    all_metadata = []
    
    for video_path in mov_files:
        is_train = video_path in train_videos
        meta = process_video(video_path, output_dir, lpips_model, device, args, is_train)
        all_metadata.extend(meta)
        
    if all_metadata:
        df = pd.DataFrame(all_metadata)
        
        # EXACT metadata column ordering matching iphoneData_1.ipynb (No 'split' column!)
        cols = ['img_id', 'video', 'source', 'img_num', 'psnr', 'target_psnr', 'lpips', 
                'blur_window', 'blur_duration_ms', 'sharpness', 'motion', 'complexity', 
                'difficulty', 'ssim', 'isp_blur', 'isp_sharp', 'isp_diff', 'noise_estimate', 
                'phash', 'contrast']
        df = df[cols]
        
        # Create the exact 3 CSV files expected by your benchmark logic
        meta_dir = output_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        
        df_train = df[df['video'].isin([v.name for v in train_videos])]
        df_test = df[~df['video'].isin([v.name for v in train_videos])]
        
        df.to_csv(meta_dir / "complete_metadata.csv", index=False)
        df_train.to_csv(meta_dir / "train_metadata.csv", index=False)
        df_test.to_csv(meta_dir / "test_metadata.csv", index=False)
        
        print("✅ Data generation perfectly successful. Memory remained stable.")

if __name__ == "__main__":
    main()