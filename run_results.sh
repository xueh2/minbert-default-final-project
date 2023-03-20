# single task

# multi-task

export lr=2e-5
export epochs=20
export batch_size=256
export optimizer=AdamW
export scheduler=ReduceLROnPlateau
export StepLR_step_size=5

# sts
python3 multitask_classifier.py --option finetune --dp --use_gpu --without_para --without_sst --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler ${scheduler} --StepLR_step_size ${StepLR_step_size} --sts_train_method regression  --weight_decay 0.0 --percentage_data_for_train 100.0 --experiment only_sts_weight_decay --wandb

# sst
python3 multitask_classifier.py --option finetune --dp --use_gpu --without_para --without_sts --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler ${scheduler} --StepLR_step_size ${StepLR_step_size} --sts_train_method regression  --weight_decay 0.0 --percentage_data_for_train 100.0 --experiment only_sst_weight_decay --wandb

#para
python3 multitask_classifier.py --option finetune --dp --use_gpu --without_sts --without_sst --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler ${scheduler} --StepLR_step_size ${StepLR_step_size} --sts_train_method regression  --weight_decay 0.0 --percentage_data_for_train 100.0 --experiment only_para_weight_decay --wandb

# all three
python3 multitask_classifier.py --option finetune --dp --use_gpu --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler ${scheduler} --StepLR_step_size ${StepLR_step_size} --sts_train_method regression  --weight_decay 0.0 --experiment all_weight_decay_min_num_steps --num_steps min --percentage_data_for_train 100.0 --wandb --without_train_for_evaluation

# run reptile
python3 multitask_classifier_reptile.py --option finetune --dp --use_gpu --lr 1e-5 --optimizer AdamW --scheduler StepLR --sts_train_method classification --wandb --experiment reptile --meta_iter 30 --meta_optimizer SGD --meta_scheduler CosineAnnealingLR --meta_weight_decay 0.0 --meta_validate_every 1 --para_iter 10 --sst_iter 10 --sts_iter 10

python3 multitask_classifier_reptile.py --option finetune --dp --use_gpu --lr 2e-5 --optimizer AdamW --scheduler StepLR --sts_train_method regression --wandb --experiment reptile --meta_lr 0.1 --meta_iter 600 --meta_optimizer SGD  --meta_scheduler CosineAnnealingLR --meta_weight_decay 0.0 --meta_validate_every 20 --para_iter 10 --sst_iter 10 --sts_iter 10 --para_batch_size 256 --sst_batch_size 128 --sts_batch_size 128 --without_train_for_evaluation --task_sample_prob_para 0.2 --task_sample_prob_sst 0.4 --task_sample_prob_sts 0.4 --StepLR_step_size 30 --percentage_data_for_train 100.0 


# individual reptile

python3 multitask_classifier_reptile.py --option finetune --dp --use_gpu --lr 1e-5 --optimizer AdamW --scheduler StepLR --StepLR_step_size 10 --StepLR_gamma 0.8 --sts_train_method regression --wandb --meta_iter 300 --meta_optimizer SGD --meta_scheduler CosineAnnealingLR --meta_weight_decay 0.0 --meta_validate_every 30 --meta_lr 1.0 --para_iter 10 --sst_iter 10 --sts_iter 10 --without_train_for_evaluation --task_sample_prob_para 0.0 --task_sample_prob_sst 0.0 --task_sample_prob_sts 1.0 --experiment reptile_only_sts

python3 multitask_classifier_reptile.py --option finetune --dp --use_gpu --lr 1e-5 --optimizer AdamW --scheduler StepLR --StepLR_step_size 10 --StepLR_gamma 0.8 --sts_train_method regression --wandb --meta_iter 300 --meta_optimizer SGD --meta_scheduler CosineAnnealingLR --meta_weight_decay 0.0 --meta_validate_every 30 --meta_lr 1.0 --para_iter 10 --sst_iter 10 --sts_iter 10 --without_train_for_evaluation --task_sample_prob_para 0.0 --task_sample_prob_sst 1.0 --task_sample_prob_sts 0.0 --experiment reptile_only_sst

python3 multitask_classifier_reptile.py --option finetune --dp --use_gpu --lr 1e-5 --optimizer AdamW --scheduler StepLR --StepLR_step_size 10 --StepLR_gamma 0.8 --sts_train_method regression --wandb --meta_iter 300 --meta_optimizer SGD --meta_scheduler CosineAnnealingLR --meta_weight_decay 0.0 --meta_validate_every 30 --meta_lr 1.0 --para_iter 10 --sst_iter 10 --sts_iter 10 --without_train_for_evaluation --task_sample_prob_para 1.0 --task_sample_prob_sst 0.0 --task_sample_prob_sts 0.0 --experiment reptile_only_para