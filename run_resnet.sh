
MODELNAME=("resnet18" "resnet50" "resnet101")
LR=("1e-3")
WD=("5e-3")
# LR=("1e-4" "5e-4" "1e-3" "5e-2" "1e-1" )
# WD=("5e-4" "1e-3" "1e-2" "10.0" "3.0" "1.0")                       
for lr in "${LR[@]}"; do
    for model in "${MODELNAME[@]}"; do
        for wd in "${WD[@]}"; do
            echo "Training ${model} with lr=${lr} and wd=${wd}"
            CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=12245\
                    Train_Galaxy_Code.py --model_name "${model}" \
                    --batch_size 32 \
                    --weight_decay "${wd}" \
                    --lr "${lr}" \
                    --save_dir "results/ring-galaxy/${model}_pretrained_fp64/${model}_lr${lr}_wd${wd}" \
                    --epochs 1
        done
    done
done



MODELNAME=("swin_small_patch4_window7_224" "swin_base_patch4_window7_224" "swin_tiny_patch4_window7_224")
LR=("1e-5")
WD=("5e-4")

# LR=("1e-4" "5e-4" "1e-3" "5e-3" "1e-2" "5e-2")
# WD=("5e-4" "1e-3" "1e-2"  "5.0" "3.0" "1.0") 

for lr in "${LR[@]}"; do
    for model in "${MODELNAME[@]}"; do
        for wd in "${WD[@]}"; do
            echo "Training ${model} with lr=${lr} and wd=${wd}"
            CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1  --master_port=12245\
                    Train_Galaxy_Code.py --model_name "${model}" \
                    --batch_size 8  \
                    --weight_decay "${wd}" \
                    --lr "${lr}" \
                    --save_dir "results/ring-galaxy/${model}_pretrained_fp64/${model}_lr${lr}_wd${wd}" \
                    --epochs 1
        done
    done
done

MODELNAME=("vit_tiny_patch16_224" "vit_small_patch16_224" "vit_base_patch16_224" )
LR=("1e-4")
WD=("5e-4")
# LR=("1e-4" "5e-4" "1e-3" "5e-3" "1e-2" "5e-2")
# WD=("5e-4" "1e-3" "1e-2"  "5.0" "3.0" "1.0")

for lr in "${LR[@]}"; do
    for model in "${MODELNAME[@]}"; do
        for wd in "${WD[@]}"; do
            echo "Training ${model} with lr=${lr} and wd=${wd}"
            CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=3451\
                    Train_Galaxy_Code.py --model_name "${model}" \
                    --batch_size 8\
                    --weight_decay "${wd}" \
                    --lr "${lr}" \
                    --save_dir "results/ring-galaxy/${model}_pretrained_fp64/${model}_lr${lr}_wd${wd}"\
                    --epochs 1
          done
      done
  done



# MODELNAME=("efficientnet_b0")
# LR=("1e-3")
# WD=("5e-4")

# for lr in "${LR[@]}"; do
#     for model in "${MODELNAME[@]}"; do
#         for wd in "${WD[@]}"; do
#             echo "Training ${model} with lr=${lr} and wd=${wd}"
#             CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1  --master_port=12245\
#                     train_galaxy_resnet.py --model_name "${model}" \
#                     --batch_size 128  \
#                     --weight_decay "${wd}" \
#                     --lr "${lr}" \
#                     --save_dir "results/lbgs/${model}_pretrained/${model}_lr${lr}_wd${wd}" \
#                     --epochs 100
#         done
#     done
# done
