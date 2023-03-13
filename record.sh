# single task

# python classifier.py --option finetune --use_gpu --batch_size ${batch_size} --epochs 20 --lr 1e-5

# multi-task

export lr=2e-5
export epochs=60
export batch_size=64
export optimizer=AdamW
export scheduler=ReduceLROnPlateau
export StepLR_step_size=5

python3 multitask_classifier.py --option finetune --dp --use_gpu --without_para --without_sts --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler ${scheduler} --StepLR_step_size ${StepLR_step_size} --sts_train_method classification --experiment only_sst

python3 multitask_classifier.py --option finetune --dp --use_gpu --without_para --without_sts --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler ${scheduler} --StepLR_step_size ${StepLR_step_size} --sts_train_method classification --weight_decay 1e-5 --experiment only_sst_weight_decay

python3 multitask_classifier.py --option finetune --dp --use_gpu --without_sst --without_sts --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler ${scheduler} --StepLR_step_size ${StepLR_step_size} --sts_train_method classification --experiment only_para

python3 multitask_classifier.py --option finetune --dp --use_gpu --without_para --without_sst --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler ${scheduler} --StepLR_step_size ${StepLR_step_size} --sts_train_method classification --experiment only_sts_classification

python3 multitask_classifier.py --option finetune --dp --use_gpu --without_para --without_sst --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler ${scheduler} --StepLR_step_size ${StepLR_step_size} --sts_train_method regression --experiment only_sts_regression

python3 multitask_classifier.py --option finetune --dp --use_gpu --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler ${scheduler} --StepLR_step_size ${StepLR_step_size} --sts_train_method regression --experiment all_sts_regression

python3 multitask_classifier.py --option finetune --dp --use_gpu --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler StepLR --StepLR_step_size ${StepLR_step_size} --sts_train_method regression --experiment all_sts_regression_StepLR
