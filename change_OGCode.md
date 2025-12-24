nuScenes_Train.txt和nuScenes_Val.txt有000～849对应编号，划分训练和验证。

1. model处理相关

* /src/main.py

增加load weight相关处理部分：prepare_checkpoint_path，处理pretrain-model的选择。load_hf_model_weights中只加载["aggregator", "camera_head", "depth_head"]，作为后续frozen model只用来提取pose和depth map。

* src/model/encoder/anysplat.py

删除distill相关代码，增加frozen componets逻辑。distill_infos["conf_mask"]来自于frozen模型而不是distill VGGT。

同理，需要注释src/model/model_wrapper.py中distill损失计算部分，不使用distill_loss。

