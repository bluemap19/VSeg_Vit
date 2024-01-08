# Self-Supervised Vision Transformers with Simsiam

## Pretrained models

model pretrained effect:

![alt](resource\pre_train_effect.png)

full borehole image segmentation parament:

| 参数 |      模型      |  单缝  | 杂交缝 |  整体  |
| :--: | :------------: | :----: | :----: | :----: |
| IoU  |      UNet      | 0.6788 | 0.6954 | 0.6871 |
| IoU  | Attention UNet | 0.6484 | 0.6972 | 0.6728 |
| IoU  |    SE-UNet     | 0.6504 | 0.6942 | 0.6723 |
| IoU  |    VSeg-Vit    | 0.7466 | 0.7032 | 0.7249 |
| Dice |      UNet      | 0.6588 | 0.7286 | 0.6937 |
| Dice | Attention UNet | 0.6292 | 0.7198 | 0.6745 |
| Dice |    SE-UNet     | 0.6411 | 0.7231 | 0.6821 |
| Dice |    VSeg-Vit    | 0.8524 | 0.7934 | 0.8229 |
|  P   |      UNet      | 0.6664 | 0.7498 | 0.7081 |
|  P   | Attention UNet | 0.6374 | 0.7384 | 0.6879 |
|  P   |    SE-UNet     | 0.6499 | 0.7365 | 0.6932 |
|  P   |    VSeg-Vit    | 0.9565 | 0.836  | 0.8962 |



V Block structure:

![alt](resource\VModel.png)

model weight:

connect email:puremaple19@outlook.com



# Train

## Documentation

Please install [PyTorch](https://pytorch.org/) and download the [ImageNet](https://imagenet.stanford.edu/) dataset. This codebase has been developed with python version 3.6, PyTorch version 1.7.1, CUDA 11.0 and torchvision 0.8.2. The exact arguments to reproduce the models presented in our paper can be found in the `args` column of the [pretrained models section](https://github.com/facebookresearch/dino#pretrained-models). For a glimpse at the full documentation of DINO training please run

simsiam/ele_seg_2/train_bottle_model.py

simsiam/ele_seg_2/train_up1_model.py

simsiam/ele_seg_2/train_up2_model.py

simsiam/ele_seg_2/train_up3_model.py



# Test

run simsiam/ele_seg_2/show_pic_segmentation_effect.py to show full borehole segmentation effect



# Effect

model stage effect

![alt](resource\Seg_task_Visualization.png)

model compare effect:

![alt](resource\model_effect_compare.png)
