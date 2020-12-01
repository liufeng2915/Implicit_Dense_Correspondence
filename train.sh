

## ----------------------  chair  ----------------------------- ##
# stage1
CUDA_VISIBLE_DEVICES=0 python train_stage1.py --cate_name chair --epoch  50 --resolution 16 --batch_size 45 
CUDA_VISIBLE_DEVICES=0 python train_stage1.py --cate_name chair --epoch 100 --resolution 32 --batch_size 25 --pretrain_model TRUE --pretrain_model_name Corr-49.pth
CUDA_VISIBLE_DEVICES=0 python train_stage1.py --cate_name chair --epoch 300 --resolution 64 --batch_size  7 --pretrain_model TRUE --pretrain_model_name Corr-99.pth

# stage2
CUDA_VISIBLE_DEVICES=0 python train_stage2.py --cate_name chair --epoch 500 --resolution 64 --batch_size  4 --pretrain_model TRUE --pretrain_model_name stage1/Corr-299.pth