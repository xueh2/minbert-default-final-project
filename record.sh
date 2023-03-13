# single task

# python classifier.py --option finetune --use_gpu --batch_size ${batch_size} --epochs 20 --lr 1e-5

# multi-task

export lr=1e-5
export epochs=50
export batch_size=64
export optimizer=AdamW

python3 multitask_classifier.py --option finetune --dp --use_gpu --without_para --without_sts --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler StepLR --StepLR_step_size 4 --sts_train_method classification --experiment only_sst

python3 multitask_classifier.py --option finetune --dp --use_gpu --without_sst --without_sts --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler StepLR --StepLR_step_size 4 --sts_train_method classification --experiment only_para

python3 multitask_classifier.py --option finetune --dp --use_gpu --without_para --without_sst --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler StepLR --StepLR_step_size 4 --sts_train_method classification --experiment only_sts

python3 multitask_classifier.py --option finetune --dp --use_gpu --without_para --without_sts --lr ${lr} --batch_size ${batch_size} --epochs ${epochs} --optimizer ${optimizer} --scheduler StepLR --StepLR_step_size 4 --sts_train_method classification --weight_decay 1e-4 --experiment only_sst_weight_decay