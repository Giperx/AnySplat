import os
import sys
import torch
import re
import math
from pathlib import Path
from torchvision.utils import save_image
from tqdm import tqdm
# CUDA_VISIBLE_DEVICES=0 python batchInferNuScenesFusion260128.py
# ================= 项目路径设置 =================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.model.anysplat import AnySplat
from src.utils.image import process_image_nuscenes

# ================= 配置区域 =================
# 批处理大小 (根据显存大小调整，建议 2, 4, 8)
BATCH_SIZE = 8

RENDER_H = 252
TARGET_W = 448     # 原始/裁切目标宽度 (GT宽度)
WIDE_W = 896       # 渲染宽度

# --- 融合功能开关 ---
ENABLE_FUSION = False  # <--- [核心修改] True: 开启融合并保存fusion图; False: 仅保存原始render图

# 融合设置
BLEND_EDGE_WIDTH = 100 # 边缘渐变宽度 (像素)
FUSION_METHOD = 'two_band' # 'simple' or 'two_band'

# PRETRAINED_PATH = "/home/test/LIVA/XZP/FeedForward/fine_tune3/AnySplat_1218infer/finetune_weights/anysplat_og/anysplat_hfog_1108"
PRETRAINED_PATH = "/home/test/LIVA/XZP/FeedForward/fine_tune3/AnySplat_1218infer/finetune_weights/260129_singleFramesReTrainGSHeadEpoch3Iter30000/weights"
DATASET_ROOT = "/home/test/LIVA/XZP/FeedForward/fine_tune3/AnySplat_1218infer/datasets/nuscenes/processed_10Hz/trainval2"
VAL_LIST_PATH = "/home/test/LIVA/XZP/FeedForward/fine_tune3/AnySplat_1218infer/nuScenes_Val.txt"

# 保存路径处理
SAVE_ROOT = "/home/test/LIVA/XZP/FeedForward/fine_tune3/AnySplat_1218infer/renders_val_split_fusionBatch2/xxxx" 
# [核心修改] 根据开关调整文件夹命名，方便区分
if ENABLE_FUSION:
    folder_suffix = f"fusion_{FUSION_METHOD}_{BLEND_EDGE_WIDTH}px"
else:
    folder_suffix = "render_only" # 如果不融合，标记文件夹为纯渲染

SAVE_ROOT = SAVE_ROOT.replace('xxxx', PRETRAINED_PATH.split('/')[-2]).replace('fusion', folder_suffix)

# ================= 工具函数 =================

def load_scene_list(txt_path):
    if not os.path.exists(txt_path):
        print(f"Error: Val list not found at {txt_path}")
        return []
    with open(txt_path, 'r') as f:
        scenes = [line.strip() for line in f.readlines() if line.strip()]
    return scenes

def get_scene_frames(scene_path):
    img_dir = os.path.join(scene_path, "images")
    if not os.path.exists(img_dir):
        return []
    frame_ids = []
    pattern = re.compile(r"^(.*)_0\.(jpg|png|jpeg)$", re.IGNORECASE)
    # 扫描一次文件夹
    for f in sorted(os.listdir(img_dir)):
        match = pattern.match(f)
        if match:
            frame_ids.append(match.group(1))
    return sorted(frame_ids)

def get_image_paths_for_frame(scene_path, frame_id, group_type):
    img_dir = os.path.join(scene_path, "images")
    if group_type == 'front':
        indices = [0, 1, 2]
    else: 
        indices = [5, 4, 3]
    
    paths = []
    for idx in indices:
        found = False
        for ext in ['.jpg', '.png', '.jpeg']:
            p = os.path.join(img_dir, f"{frame_id}_{idx}{ext}")
            if os.path.exists(p):
                paths.append(p)
                found = True
                break
        if not found:
            return None, None
    return paths, indices

# ================= 融合策略函数 =================

def create_edge_blend_mask(h, w, edge_width, device, strategy='smoothstep'):
    mask = torch.ones((1, h, w), device=device)
    if edge_width <= 0: return mask
    
    t = torch.linspace(0, 1, edge_width, device=device)
    if strategy == 'smoothstep':
        weight = 3 * t**2 - 2 * t**3
    else:
        weight = t # linear

    mask[:, :, :edge_width] = weight.view(1, 1, -1)
    mask[:, :, -edge_width:] = torch.flip(weight, dims=[0]).view(1, 1, -1)
    return mask

def two_band_blending(render_img, gt_img, mask, blur_sigma=5):
    import torchvision
    blurrer = torchvision.transforms.GaussianBlur(kernel_size=21, sigma=blur_sigma)
    render_low = blurrer(render_img)
    gt_low = blurrer(gt_img)
    render_high = render_img - render_low
    gt_high = gt_img - gt_low
    
    result_low = gt_low * mask + render_low * (1 - mask)
    result_high = gt_high * mask + render_high * (1 - mask)
    return result_low + result_high

def blend_images(render_img, gt_img, mask):
    if render_img.shape != gt_img.shape:
        gt_resized = torch.nn.functional.interpolate(
            gt_img.unsqueeze(0), 
            size=render_img.shape[-2:], 
            mode='bilinear'
        ).squeeze(0)
    else:
        gt_resized = gt_img
    return gt_resized * mask + render_img * (1 - mask)

# ================= 主流程 =================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading model from {PRETRAINED_PATH}...")
    model = AnySplat.from_pretrained(PRETRAINED_PATH).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    # [核心修改] 仅当开启融合时才创建 Mask
    fusion_mask = None
    if ENABLE_FUSION:
        fusion_mask = create_edge_blend_mask(RENDER_H, TARGET_W, BLEND_EDGE_WIDTH, device)

    # 1. 收集所有任务
    scenes = load_scene_list(VAL_LIST_PATH)
    all_tasks = []
    
    print("Scanning scenes to collect tasks...")
    for scene_id in tqdm(scenes, desc="Scanning"):
        scene_dir = os.path.join(DATASET_ROOT, scene_id)
        frames = get_scene_frames(scene_dir)
        
        # 确保该场景的保存目录存在
        scene_save_dir = os.path.join(SAVE_ROOT, scene_id)
        os.makedirs(scene_save_dir, exist_ok=True)
        
        for frame_id in frames:
            for group_type in ['front', 'back']:
                all_tasks.append({
                    'scene_id': scene_id,
                    'frame_id': frame_id,
                    'group_type': group_type,
                    'scene_dir': scene_dir,
                    'save_dir': scene_save_dir
                })

    total_tasks = len(all_tasks)
    num_batches = math.ceil(total_tasks / BATCH_SIZE)
    print(f"Total tasks: {total_tasks}. Batch size: {BATCH_SIZE}. Total batches: {num_batches}")
    print(f"Fusion Enabled: {ENABLE_FUSION}")
    print(f"Saving to {SAVE_ROOT}")

    # 2. 按 Batch 处理
    for i in tqdm(range(num_batches), desc="Processing Batches"):
        # 获取当前 Batch 的任务
        batch_tasks_meta = all_tasks[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        
        batch_images = []
        valid_tasks = [] # 记录有效任务（防止某些图片缺失）
        
        # 加载数据 (IO密集)
        for task in batch_tasks_meta:
            paths, indices = get_image_paths_for_frame(task['scene_dir'], task['frame_id'], task['group_type'])
            if paths:
                # 加载3张图 -> [3, C, H, W]
                imgs = [process_image_nuscenes(p) for p in paths]
                # 用于推理的 Tensor 栈
                batch_images.append(torch.stack(imgs))
                
                # 记录元数据供后续保存使用
                task['cam_indices'] = indices
                
                # [核心修改] 仅当开启融合时，才在 CPU 缓存原始图片用于融合计算
                # 这样如果只跑 render，可以节省大量内存
                if ENABLE_FUSION:
                    task['input_images_cpu'] = torch.stack(imgs)
                
                valid_tasks.append(task)
        
        if not valid_tasks:
            continue
            
        # 堆叠 Batch -> [B, 3, 3, H, W]
        input_tensor = torch.stack(batch_images).to(device)
        current_bs = input_tensor.shape[0]
        views_per_group = 3 # 组内固定3张图
        
        with torch.no_grad():
            # [Batch Inference]
            gaussians, pred_context_pose = model.inference((input_tensor + 1) * 0.5)
            
            # Modify Intrinsics
            new_intrinsics = pred_context_pose['intrinsic'].clone()
            width_scale = TARGET_W / WIDE_W
            new_intrinsics[..., 0, 0] *= width_scale
            
            # Prepare depth range
            t_near = torch.ones(current_bs, views_per_group, device=device) * 0.01
            t_far = torch.ones(current_bs, views_per_group, device=device) * 100.0
            
            # [Batch Render]
            outputs = model.decoder.forward(
                gaussians,
                pred_context_pose['extrinsic'],
                new_intrinsics.float(),
                t_near,
                t_far,
                (RENDER_H, WIDE_W),
            )
            # Output: [B, 3, C, H, W]
            rendered_batch = outputs.color
            
            # 3. 保存与融合
            for b in range(current_bs):
                task = valid_tasks[b]
                group_render = rendered_batch[b] # [3, C, H, W]
                
                # 如果开启融合，需要取出缓存的GT
                if ENABLE_FUSION:
                    group_input_gt = task['input_images_cpu'].to(device) # [3, C, H, W] in [-1, 1]
                
                # 遍历组内的 3 个相机
                for v in range(views_per_group):
                    cam_idx = task['cam_indices'][v]
                    pred_img = group_render[v]
                    
                    # A. [Always] 保存原始宽图
                    wide_name = f"{task['frame_id']}_{cam_idx}_wide.jpg"
                    save_image(pred_img, os.path.join(task['save_dir'], wide_name))
                    
                    # B. [Optional] 处理融合 (仅当开启开关，且为每组第一张)
                    if ENABLE_FUSION and v == 0:
                        # 准备裁剪后的 GT (转为 [0, 1])
                        gt_img = (group_input_gt[v] + 1) * 0.5
                        
                        # 准备裁剪后的 Render
                        start_x = (WIDE_W - TARGET_W) // 2
                        pred_crop = pred_img[:, :, start_x : start_x + TARGET_W]
                        
                        # 执行融合
                        if FUSION_METHOD == 'simple':
                            fusion_crop = blend_images(pred_crop, gt_img, fusion_mask)
                        elif FUSION_METHOD == 'two_band':
                            fusion_crop = two_band_blending(pred_crop, gt_img, fusion_mask)
                        else:
                            fusion_crop = pred_crop

                        # 贴回宽图背景
                        pred_img_fusion = pred_img.clone()
                        pred_img_fusion[:, :, start_x : start_x + TARGET_W] = fusion_crop
                        
                        wide_fusion_name = f"{task['frame_id']}_{cam_idx}_wide_fusion.jpg"
                        save_image(pred_img_fusion, os.path.join(task['save_dir'], wide_fusion_name))

    print(f"Rendering Finished. Saved to {SAVE_ROOT}")

if __name__ == "__main__":
    main()