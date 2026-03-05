260120 comment

当前分支没有dynamic head以及动静分离处理。只处理输入的多帧数据，损失函数和AnySplat OG保持一致。

260128 comment

* src/model/encoder/anysplat.py

修改Gaussian Head，输入特征增加DINO，使用0阶球谐系数。

* src/model/encoder/common/gaussian_adapter.py

处理DGGT Gaussian Head 逻辑，使用0阶球谐系数 config/model/encoder/anysplat.yaml:sh_degree.
scales = 0.1 * F.softplus(scales)

OG gs_dpt_head:
class UnifiedGaussianAdapter(GaussianAdapter):
        scales = 0.003 * F.softplus(scales)
        scales = scales.clamp_max(0.5)

---

nuScenes_Train.txt和nuScenes_Val.txt有000～849对应编号，划分训练和验证。

1. model处理相关

* /src/main.py

增加load weight相关处理部分：prepare_checkpoint_path，处理pretrain-model的选择。load_hf_model_weights中只加载["aggregator", "camera_head", "depth_head"]，作为后续frozen model只用来提取pose和depth map。当前过渡，根据flag_gaussian_head判断是否加载gaussian_head权重。

* src/model/encoder/anysplat.py

删除distill相关代码，增加frozen componets逻辑。distill_infos["conf_mask"]来自于frozen模型而不是distill VGGT。

同理，需要注释src/model/model_wrapper.py中distill损失计算部分，不使用distill_loss。

2. datasets处理相关

* config/dataset/view_sampler/all.yaml

补充参数字段，否则src/dataset/data_sampler.py中初始化DynamicBatchSampler时会报错。

* src/dataset/dataset_nuscenes.py

nuScenes数据读入代码，读入10Hz版本数据集。强制resize到448x448。train时samples为700 scenes的所有组合；val时为150 scenes到第一个组合。

根据numTimes采样连续数量的帧。front视角和back视角是随机概率，不影响samples数量。

target现在只取了cur所有视角，但是不影响train和val，因为target只在test中使用。

* TODO：增加intervalTimes间隔时间采样。

* nuscenes.yaml

config/dataset和config/experiment下新建nuscenes.yaml

* src/dataset/__init__.py

补充DATASETS和DatasetCfgWrapper字段

* src/dataset/data_module.py    1

补充多数据训练时选择数据集概率，暂定为100%，因为目前只考虑nuScenes单数据集训练。

* src/dataset/data_sampler.py

fixed图像的高度相关random_ps_h；实际上后续src/dataset/dataset_nuscenes.py没有使用

初始化DynamicBatchSampler时fixed image num for each dataset

* src/dataset/data_module.py    2

3. model head相关

* src/model/encoder/vggt/models/aggregator.py

增加return output_list_with_tokens, dino_token_list供后续的gs_head和dynamic_head使用。

* src/model/encoder/heads/GaussianHead.py   head_act.py utils.py

增加对应py文件。GaussianHead和原来一致，返回长度特征7+3rgb+1: [color,opacity,scale,rotation] conf. 原来的head返回特征后续对应为scales, rotations, sh, conf。

* TODO: GaussianHead的返回应修改为feat，衔接原来的vggt_dpt_gs_head的返回结果，供后续的vol使用。
* TODO: gs_activate_head的逻辑处理进UnifiedGaussianAdapter中。 还需check。

* src/model/encoder/anysplat.py   dynamic_head

self.dynamic_head = DPTHeadDGGT(dim_in= head_params.enc_embed_dim, output_dim = 1 + 1, activation="linear")

重构vol部分，处理动静分离。合并当前帧与历史帧静态部分。dynamic_conf加到depth_dict中

20260109 对于重构vol部分，额外增加返回static_gaussians逻辑，表示当前与历史帧所有静态高斯部分。

4. 训练过程相关

* /home/test/LIVA/XZP/FeedForward/fine_tune3/AnySplat_1218/src/loss/loss_dynamic_mask.py

增加dynamic_mask loss, 使用BCE loss，参考论文UniSplat。对所有帧计算loss。

* /home/test/LIVA/XZP/FeedForward/fine_tune3/AnySplat_1218/src/model/model/anysplat.py 1

增加剔除不需要的视图逻辑，比如当前帧只需要[:, :3, ...]。根据gaussians和static_gaussians，只渲染当前帧或所有帧。返回output和output_static.注意render output_static还是需要完整intrinsics和extrinsics，因为参考帧是第一帧。后续计算loss通过索引获得static部分。

* /home/test/LIVA/XZP/FeedForward/fine_tune3/AnySplat_1218/src/model/model_wrapper.py

增加dynamic_mask loss。

loss计算中，因为动态物体在变化，直接对所有帧相同监督会出现问题：可能出现当前帧动态物体遮挡历史帧静态背景。因此需要考虑额外得到static_gaussians来render历史帧的实际静态图像进行光度损失计算。

默认的mse、lpips、depth consis修改逻辑：cur当前帧正常处理；his静态历史帧增加dynamic mask处理，只监督静态部分。

psnr_probabilistic修改为只处理当前帧; model_wrapper中相关代码逻辑符合修改后的loss

dynamic_head的参数放入new_params中。

5. 训练结果相关

* src/misc/LocalLogger.py

重构Init和image保存名称。

* /home/test/LIVA/XZP/FeedForward/fine_tune3/AnySplat_1218/src/model/model/anysplat.py 2

增加宽视野render逻辑。

* /home/test/LIVA/XZP/FeedForward/fine_tune3/AnySplat_1218/src/model/model_wrapper.py

validation_step中，val/psnr、ssim、lpips、consis_absrel、consis_delta1、consis_mse只计算cur当前帧。

comparison原输出组图序列只保留cur当前帧；comparison_static只输出his历史帧的static部分；comparison_wide只输出cur所有当前帧和历史帧front-view的宽视野图像。

render_video_interpolation处理render视频为最后一帧的-2～第一帧的2视频。左视跳右视。

dynamic_mask的render结果放入comparison和comparison_static

* TODO：处理validation_step中相关逻辑，符合修改后代码。


tar -czvf anysplat0202omni.tar.gz --exclude=./AnySplat_1218/anysplat_hfog_1108 --exclude=./AnySplat_1218/datasets --exclude=./AnySplat_1218/outputs --exclude=./AnySplat_1218/.git --exclude=./AnySplat_1218/.vscode ./AnySplat_1218

---

