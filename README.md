# Multi_model_galaxy_classfication

本项目旨在利用多种深度学习模型对星系进行分类,减少误检率。我们选择了Vision Transformer (ViT) 系列中的ViT-tiny、ViT-small和ViT-base，Swin Transformer系列中的Swin-tiny、Swin-small和Swin-base，以及ResNet系列的ResNet-18、ResNet-50和ResNet-101作为基模型。所有模型均加载了基于ImageNet-1K数据集预训练的权重以初始化网络参数。随后，从目标数据集中随机选取80%的数据用于每个模型的微调训练，而保留剩余的20%数据作为测试集来评估模型性能。最终结果是通过集成学习中的投票机制得出，即每个基模型对测试样本进行预测后，根据投票结果决定最终预测结果。

# 以低表面亮度星系为例

**数据集制作**：

   - 图像数据：从[DESI DR10](https://www.legacysurvey.org/dr10/)下载所需的fits图像数据。
   - 星表数据：[Du.2015](https://arxiv.org/abs/1504.07711) 利用α.40星表发现的非侧向星系的1129个LSBG
