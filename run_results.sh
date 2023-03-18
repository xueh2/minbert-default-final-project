# single task

# multi-task

export lr=2e-5
export epochs=15
export batch_size=64
export optimizer=AdamW
export scheduler=ReduceLROnPlateau
export StepLR_step_size=5

# sts
python3 multitask_classifier.py --option finetune --dp --use_gpu --without_para --without_sst --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler ${scheduler} --StepLR_step_size ${StepLR_step_size} --sts_train_method regression  --weight_decay 0.0 --experiment only_sts_weight_decay --wandb

# sst
python3 multitask_classifier.py --option finetune --dp --use_gpu --without_para --without_sts --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler ${scheduler} --StepLR_step_size ${StepLR_step_size} --sts_train_method regression  --weight_decay 0.0 --experiment only_sst_weight_decay --wandb

#para
python3 multitask_classifier.py --option finetune --dp --use_gpu --without_sts --without_sst --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler ${scheduler} --StepLR_step_size ${StepLR_step_size} --sts_train_method regression  --weight_decay 0.0 --experiment only_para_weight_decay --wandb

# all three
python3 multitask_classifier.py --option finetune --dp --use_gpu --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler ${scheduler} --StepLR_step_size ${StepLR_step_size} --sts_train_method regression  --weight_decay 1.0 --experiment all_weight_decay --wandb

python3 multitask_classifier.py --option finetune --dp --use_gpu --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler ${scheduler} --StepLR_step_size ${StepLR_step_size} --sts_train_method regression  --weight_decay 1.0 --experiment all_weight_decay_min_num_steps --num_steps min --wandb

# run reptile
python3 multitask_classifier_reptile.py --option finetune --dp --use_gpu --lr 1e-5 --optimizer AdamW --scheduler StepLR --sts_train_method classification --wandb --experiment reptile --meta_iter 30 --meta_optimizer SGD --meta_scheduler CosineAnnealingLR --meta_weight_decay 0.0 --meta_validate_every 1 --para_iter 10 --sst_iter 10 --sts_iter 10

python3 multitask_classifier_reptile.py --option finetune --dp --use_gpu --lr 2e-5 --optimizer AdamW --scheduler StepLR --sts_train_method regression --wandb --experiment reptile --meta_lr 0.1 --meta_iter 600 --meta_optimizer SGD  --meta_scheduler CosineAnnealingLR --meta_weight_decay 1.0 --meta_validate_every 20 --para_iter 10 --sst_iter 10 --sts_iter 10 --para_batch_size 32 --sst_batch_size 64 --sts_batch_size 64 --without_train_for_evaluation