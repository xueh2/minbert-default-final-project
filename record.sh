# single task

python classifier.py --option finetune --use_gpu --batch_size 64 --epochs 20 --lr 1e-5

# multi-task

python multitask_classifier.py --option finetune --dp --use_gpu --without_para --without_sts --lr 1e-5 --batch_size 64 --epochs 20 --optimizer AdamW --scheduler ReduceLROnPlateau