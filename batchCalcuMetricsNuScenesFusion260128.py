import os
import sys
import torch
import numpy as np
import re
import math
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm
# CUDA_VISIBLE_DEVICES=1 python batchCalcuMetricsNuScenesFusion260128.py
# ================= 项目路径设置 =================
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.image import process_image_nuscenes

# ================= 配置区域 =================
# [核心修改] 批处理大小
BATCH_SIZE = 32  # 建议设大一点 (16-64)，LPIPS 批处理极快

RENDER_H = 252
TARGET_W = 448     # 裁剪目标宽度
WIDE_W = 896       # 渲染宽度

# 功能开关
EVAL_FUSION = False        # True: 计算Fusion指标; False: 仅计算Original指标
SELECTED_CAMERAS = [0, 5] # 仅计算这些相机的指标 (空列表 [] 代表计算所有相机 0-5)

# 路径配置
PRETRAINED_PATH = "/home/test/LIVA/XZP/FeedForward/fine_tune3/AnySplat_1218infer/finetune_weights/anysplat_og/anysplat_hfog_1108"
DATASET_ROOT = "/home/test/LIVA/XZP/FeedForward/fine_tune3/AnySplat_1218infer/datasets/nuscenes/processed_10Hz/trainval2"
VAL_LIST_PATH = "/home/test/LIVA/XZP/FeedForward/fine_tune3/AnySplat_1218infer/nuScenes_Val.txt"

# SAVE_ROOT = "/home/test/LIVA/XZP/FeedForward/fine_tune3/AnySplat_1218infer/renders_val_split_fusionBatch/xxxx" 
# BLEND_EDGE_WIDTH = 100 
# FUSION_METHOD = 'two_band' 

# folder_suffix = f"fusion_{FUSION_METHOD}_{BLEND_EDGE_WIDTH}px" 
# SAVE_ROOT = SAVE_ROOT.replace('xxxx', PRETRAINED_PATH.split('/')[-2]).replace('fusion', folder_suffix)
SAVE_ROOT = "/home/test/LIVA/XZP/FeedForward/fine_tune3/AnySplat_1218/renders_val_split_render_onlyBatch/260129_singleFramesReTrainGSHeadEpoch2Iter25000"
RESULT_TXT_PATH = os.path.join(SAVE_ROOT, f"metrics_report_cams_{'_'.join(map(str, SELECTED_CAMERAS)) if SELECTED_CAMERAS else 'all'}_batched.txt")

# ================= 统计类 =================

class MetricTracker:
    def __init__(self):
        self.reset()
    def reset(self):
        self.psnr = []
        self.ssim = []
        self.lpips = []
    def update(self, psnr, ssim, lpips):
        self.psnr.append(psnr)
        self.ssim.append(ssim)
        self.lpips.append(lpips)
    def get_avg(self):
        return {
            'psnr': np.mean(self.psnr) if self.psnr else 0.0,
            'ssim': np.mean(self.ssim) if self.ssim else 0.0,
            'lpips': np.mean(self.lpips) if self.lpips else 0.0
        }

def format_metrics(m):
    avg = m.get_avg()
    return f"P={avg['psnr']:.2f} | S={avg['ssim']:.4f} | L={avg['lpips']:.4f}"

# ================= 工具函数 =================

def load_scene_list(txt_path):
    if not os.path.exists(txt_path): return []
    with open(txt_path, 'r') as f:
        scenes = [line.strip() for line in f.readlines() if line.strip()]
    return scenes

def get_scene_frames(scene_path):
    img_dir = os.path.join(scene_path, "images")
    if not os.path.exists(img_dir): return []
    frame_ids = []
    pattern = re.compile(r"^(.*)_0\.(jpg|png|jpeg)$", re.IGNORECASE)
    for f in sorted(os.listdir(img_dir)):
        match = pattern.match(f)
        if match: frame_ids.append(match.group(1))
    return sorted(frame_ids)

def get_image_path(scene_dir, frame_id, cam_idx):
    # 简单的单个图片路径获取
    img_dir = os.path.join(scene_dir, "images")
    for ext in ['.jpg', '.png', '.jpeg']:
        p = os.path.join(img_dir, f"{frame_id}_{cam_idx}{ext}")
        if os.path.exists(p):
            return p
    return None

def load_tensor_from_path(path):
    # 返回 CPU tensor，稍后统一转 GPU
    if not os.path.exists(path): return None
    img = Image.open(path).convert('RGB')
    return transforms.ToTensor()(img)

def should_process_cam(cam_idx):
    if not SELECTED_CAMERAS: return True
    return cam_idx in SELECTED_CAMERAS

# ================= 主流程 =================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Metrics (LPIPS) on {device}...")
    # [核心修复] 增加 reduction='none'，确保返回 [Batch_Size] 大小的张量，而不是一个平均数
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='alex', reduction='none').to(device)
    # LPIPS eval mode
    lpips_fn.eval()
    
    scenes = load_scene_list(VAL_LIST_PATH)
    if not os.path.exists(SAVE_ROOT):
        print(f"Error: Render dir {SAVE_ROOT} does not exist.")
        return

    # === 全局统计器 ===
    def create_stats_dict():
        d = {
            'total': MetricTracker(),
            'front_grp': MetricTracker(),
            'back_grp': MetricTracker(),
            'cam0_5_avg': MetricTracker()
        }
        for i in range(6): d[i] = MetricTracker()
        return d

    global_stats_orig = create_stats_dict()
    global_stats_fusion = create_stats_dict() if EVAL_FUSION else None
    
    # === 场景级统计器存储 ===
    # 为了最后能按场景打印，我们需要保存每个场景的统计对象
    # 结构: scene_stats_map[scene_id] = {'orig': stats_dict, 'fusion': stats_dict}
    scene_stats_map = {}
    
    # 1. 扫描并收集所有任务
    all_tasks = []
    print("Scanning scenes to collect evaluation tasks...")
    
    for scene_id in tqdm(scenes, desc="Scanning"):
        scene_dir = os.path.join(DATASET_ROOT, scene_id)
        render_scene_dir = os.path.join(SAVE_ROOT, scene_id)
        
        # 初始化该场景的统计器
        scene_orig = create_stats_dict()
        scene_fusion = create_stats_dict() if EVAL_FUSION else None
        scene_stats_map[scene_id] = {'orig': scene_orig, 'fusion': scene_fusion}
        
        frames = get_scene_frames(scene_dir)
        
        for frame_id in frames:
            # 遍历所有相机 0-5
            for cam_idx in range(6):
                if not should_process_cam(cam_idx):
                    continue
                
                # 确定组别
                grp_key = 'front_grp' if cam_idx in [0, 1, 2] else 'back_grp'
                
                # 确定是否是 Fusion 相机 (0 或 5)
                # 逻辑：只有每组的第一张（0 或 5）才有融合图
                is_fusion_cam = (cam_idx == 0 or cam_idx == 5)
                
                gt_path = get_image_path(scene_dir, frame_id, cam_idx)
                if gt_path is None: continue

                task = {
                    'scene_id': scene_id,
                    'frame_id': frame_id,
                    'cam_idx': cam_idx,
                    'grp_key': grp_key,
                    'gt_path': gt_path,
                    'render_dir': render_scene_dir,
                    'is_fusion_cam': is_fusion_cam
                }
                all_tasks.append(task)

    # 2. Batch 处理
    total_tasks = len(all_tasks)
    num_batches = math.ceil(total_tasks / BATCH_SIZE)
    print(f"Total tasks: {total_tasks}. Batch size: {BATCH_SIZE}. Batches: {num_batches}")
    
    # 裁剪参数
    start_x = (WIDE_W - TARGET_W) // 2

    for i in tqdm(range(num_batches), desc="Evaluating Batches"):
        batch_tasks = all_tasks[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        
        # 准备数据容器
        gt_list = []
        pred_orig_list = []
        pred_fusion_list = [] # 仅当 EVAL_FUSION 时使用
        
        # 用于记录有效数据的索引映射，因为有些图可能读不到
        valid_indices = [] 
        
        # 数据加载 (CPU -> List)
        for idx, task in enumerate(batch_tasks):
            # A. Load GT
            # process_image_nuscenes 加载并 resize, 返回 [-1, 1]
            # 我们在这里把它转为 [0, 1]
            gt_tensor_raw = process_image_nuscenes(task['gt_path'])
            gt_img = (gt_tensor_raw + 1) * 0.5
            
            # B. Load Render Original
            wide_path = os.path.join(task['render_dir'], f"{task['frame_id']}_{task['cam_idx']}_wide.jpg")
            render_wide = load_tensor_from_path(wide_path) # CPU tensor [C, H, W]
            
            if render_wide is None:
                continue
                
            # Crop
            # pred_crop = render_wide[:, start_x : start_x + TARGET_W]
            pred_crop = render_wide[:, :, start_x : start_x + TARGET_W]
            
            gt_list.append(gt_img)
            pred_orig_list.append(pred_crop)
            
            # C. Load Render Fusion (Optional)
            if EVAL_FUSION:
                fusion_path = os.path.join(task['render_dir'], f"{task['frame_id']}_{task['cam_idx']}_wide_fusion.jpg")
                
                # 如果是融合相机且文件存在，加载融合图；否则沿用原始裁剪图
                if task['is_fusion_cam'] and os.path.exists(fusion_path):
                    render_wide_f = load_tensor_from_path(fusion_path)
                    # pred_crop_f = render_wide_f[:, start_x : start_x + TARGET_W]
                    pred_crop_f = render_wide_f[:, :, start_x : start_x + TARGET_W]
                else:
                    pred_crop_f = pred_crop # Clone if needed, but for metric calc it's fine
                
                pred_fusion_list.append(pred_crop_f)
            
            valid_indices.append(idx)

        if not valid_indices:
            continue

        # 转 GPU Batch [B, C, H, W]
        gt_batch = torch.stack(gt_list).to(device)
        pred_orig_batch = torch.stack(pred_orig_list).to(device)
        if EVAL_FUSION:
            pred_fusion_batch = torch.stack(pred_fusion_list).to(device)
        
        # Batch Calculation
        with torch.no_grad():
            # Original Metrics
            # LPIPS supports batch directly
            lpips_orig = lpips_fn(pred_orig_batch, gt_batch) # [B]
            
            # PSNR/SSIM usually done per image, but we can loop over the GPU tensor
            # which is faster than transferring back and forth
            current_bs = gt_batch.shape[0]
            
            # 分发结果
            for b in range(current_bs):
                # 获取对应的 Task
                task_idx = valid_indices[b]
                task = batch_tasks[task_idx]
                scene_id = task['scene_id']
                cam_idx = task['cam_idx']
                grp_key = task['grp_key']
                
                # --- Original Metrics ---
                p = peak_signal_noise_ratio(pred_orig_batch[b:b+1], gt_batch[b:b+1], data_range=1.0).item()
                s = structural_similarity_index_measure(pred_orig_batch[b:b+1], gt_batch[b:b+1], data_range=1.0).item()
                l = lpips_orig[b].item()
                
                # 更新统计器 (Global & Scene)
                scene_orig_stats = scene_stats_map[scene_id]['orig']
                
                def update_all(d_global, d_scene, cam, p, s, l):
                    # Global Update
                    d_global['total'].update(p, s, l)
                    d_global[grp_key].update(p, s, l)
                    d_global[cam].update(p, s, l)
                    if cam in [0, 5]: d_global['cam0_5_avg'].update(p, s, l)
                    
                    # Scene Update
                    d_scene['total'].update(p, s, l)
                    d_scene[grp_key].update(p, s, l)
                    d_scene[cam].update(p, s, l)
                    if cam in [0, 5]: d_scene['cam0_5_avg'].update(p, s, l)

                update_all(global_stats_orig, scene_orig_stats, cam_idx, p, s, l)
                
                # --- Fusion Metrics ---
                if EVAL_FUSION:
                    # 如果这帧图就是原始图（没做融合），其实 PSNR 应该完全一样
                    # 但为了逻辑严谨，我们还是对 batch 中的对应 tensor 算一遍
                    # (或者你可以加个 if 判断，如果 tensor 相同直接复用 p,s,l，节省计算)
                    
                    # LPIPS Batch calc for Fusion
                    # 注意：为了代码简单，这里没有像 Original 那样做一次性 Batch LPIPS，
                    # 而是为了对齐索引，我们假设 pred_fusion_batch 也准备好了
                    pass 

            # [优化] Fusion 的 LPIPS 也可以一次性算
            if EVAL_FUSION:
                lpips_fusion = lpips_fn(pred_fusion_batch, gt_batch)
                
                for b in range(current_bs):
                    task_idx = valid_indices[b]
                    task = batch_tasks[task_idx]
                    scene_id = task['scene_id']
                    cam_idx = task['cam_idx']
                    grp_key = task['grp_key']
                    
                    pf = peak_signal_noise_ratio(pred_fusion_batch[b:b+1], gt_batch[b:b+1], data_range=1.0).item()
                    sf = structural_similarity_index_measure(pred_fusion_batch[b:b+1], gt_batch[b:b+1], data_range=1.0).item()
                    lf = lpips_fusion[b].item()
                    
                    scene_fusion_stats = scene_stats_map[scene_id]['fusion']
                    update_all(global_stats_fusion, scene_fusion_stats, cam_idx, pf, sf, lf)

    # ================= 写入报告 =================
    print("Generating report...")
    
    with open(RESULT_TXT_PATH, 'w') as f:
        f.write("Evaluation Report (Batched)\n")
        f.write(f"Source: {SAVE_ROOT}\n")
        f.write(f"Fusion Evaluated: {EVAL_FUSION}\n")
        f.write(f"Selected Cameras: {SELECTED_CAMERAS if SELECTED_CAMERAS else 'All'}\n")
        f.write("================================================================================\n")
        
        def fmt(tracker): return format_metrics(tracker)

        def write_block(title, d):
            f.write(f">>> [{title}] <<<\n")
            f.write(f"Total Average                : {fmt(d['total'])}\n")
            f.write(f"Cam 0 & 5 Average            : {fmt(d['cam0_5_avg'])}\n")
            f.write(f"Group Front Avg (0,1,2)      : {fmt(d['front_grp'])}\n")
            f.write(f"Group Back Avg  (5,4,3)      : {fmt(d['back_grp'])}\n")
            f.write("-" * 40 + "\n")
            f.write("Individual Cameras:\n")
            for c in range(6):
                if should_process_cam(c):
                    f.write(f"  Cam {c}: {fmt(d[c])}\n")
            f.write("\n")

        write_block("Original Renders", global_stats_orig)
        if EVAL_FUSION:
            write_block("Fusion Renders", global_stats_fusion)

        f.write("================================================================================\n")
        f.write("Per Scene Breakdown:\n")
        
        # 遍历所有场景写入日志
        for scene_id in scenes:
            if scene_id not in scene_stats_map: continue
            stats = scene_stats_map[scene_id]
            
            def get_line(title, d):
                return (f"  [{title}] Total: {fmt(d['total'])} | Cam0+5: {fmt(d['cam0_5_avg'])}\n"
                        f"     Front Det -> 0: {fmt(d[0])} | 1: {fmt(d[1])} | 2: {fmt(d[2])}\n"
                        f"     Back  Det -> 5: {fmt(d[5])} | 4: {fmt(d[4])} | 3: {fmt(d[3])}")

            f.write(f"Scene {scene_id}:\n")
            f.write(get_line("Original", stats['orig']) + "\n")
            if EVAL_FUSION:
                f.write(get_line("Fusion  ", stats['fusion']) + "\n")
            f.write("\n")
            
    print(f"\nEvaluation finished. Results saved to {RESULT_TXT_PATH}")

if __name__ == "__main__":
    main()