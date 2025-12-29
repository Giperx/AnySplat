nuScenes_Train.txt和nuScenes_Val.txt有000～849对应编号，划分训练和验证。

1. model处理相关

* /src/main.py

增加load weight相关处理部分：prepare_checkpoint_path，处理pretrain-model的选择。load_hf_model_weights中只加载["aggregator", "camera_head", "depth_head"]，作为后续frozen model只用来提取pose和depth map。

* src/model/encoder/anysplat.py

删除distill相关代码，增加frozen componets逻辑。distill_infos["conf_mask"]来自于frozen模型而不是distill VGGT。

同理，需要注释src/model/model_wrapper.py中distill损失计算部分，不使用distill_loss。

2. datasets处理相关

* src/dataset/dataset_nuscenes.py

nuScenes数据读入代码，读入10Hz版本数据集。强制resize到448x448。train时samples为700 scenes的所有组合；val时为150 scenes到第一个组合。

根据numTimes采样连续数量的帧。front视角和back视角是随机概率，不影响samples数量。

* TODO：增加intervalTimes间隔时间采样。

* nuscenes.yaml

config/dataset和config/experiment下新建nuscenes.yaml

* src/dataset/__init__.py

补充DATASETS和DatasetCfgWrapper字段

* src/dataset/data_module.py    1

补充多数据训练时选择数据集概率，暂定为100%，因为目前只考虑nuScenes单数据集训练。

* src/dataset/data_sampler.py

fixed图像的高度相关random_ps_h；

初始化DynamicBatchSampler时fixed image num for each dataset

* src/dataset/data_module.py    2

3. model head相关

* TODO：修改Gaussian Head，增加Dynamic Head

* src/model/encoder/heads/GaussianHead.py   head_act.py utils.py

增加对应py文件。GaussianHead和原来一致，返回长度特征7+3rgb+1: [color,opacity,scale,rotation] conf. 原来的head返回特征后续对应为scales, rotations, sh, conf。

* TODO: GaussianHead的返回应修改为feat，衔接原来的vggt_dpt_gs_head的返回结果，供后续的vol使用。
* TODO: gs_activate_head的逻辑处理进UnifiedGaussianAdapter中。 还需check。
* TODO: 增加判断逻辑，选用哪种gs head。

* dynamic_head

self.dynamic_head = DPTHeadDGGT(dim_in= head_params.enc_embed_dim, output_dim = 1 + 1, activation="linear")

* TODO: 合并相关mask逻辑，实现动静分离。

* src/model/encoder/vggt/models/aggregator.py

增加return output_list_with_tokens, dino_token_list供后续的gs_head和dynamic_head使用。

4. 训练过程相关

* TODO：宽视野图像生成
