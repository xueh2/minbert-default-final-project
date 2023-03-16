# single task

# multi-task

export lr=2e-5
export epochs=20
export batch_size=64
export optimizer=AdamW
export scheduler=ReduceLROnPlateau
export StepLR_step_size=5

# sts
python3 multitask_classifier.py --option finetune --dp --use_gpu --without_para --without_sst --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler ${scheduler} --StepLR_step_size ${StepLR_step_size} --sts_train_method regression  --weight_decay 1.0 --experiment only_sts_weight_decay --wandb

# sst
python3 multitask_classifier.py --option finetune --dp --use_gpu --without_para --without_sts --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler ${scheduler} --StepLR_step_size ${StepLR_step_size} --sts_train_method regression  --weight_decay 1.0 --experiment only_sst_weight_decay --wandb

#para
python3 multitask_classifier.py --option finetune --dp --use_gpu --without_sts --without_sst --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler ${scheduler} --StepLR_step_size ${StepLR_step_size} --sts_train_method regression  --weight_decay 1.0 --experiment only_para_weight_decay --wandb

# all three
python3 multitask_classifier.py --option finetune --dp --use_gpu --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler ${scheduler} --StepLR_step_size ${StepLR_step_size} --sts_train_method regression  --weight_decay 1.0 --experiment all_weight_decay --wandb

