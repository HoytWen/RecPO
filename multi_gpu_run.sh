CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node 4 --master_port=25642 sft.py \
        --model_name meta-llama/Llama-3.2-1B-Instruct  \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --prompt_path ./prompt/movie2.txt \
        --logging_dir log/ \
        --output_dir output/ \
        --learning_rate 1e-5 \
        --num_train_epochs 5 \
        --eval_step 0.1 \
        --report_to wandb \
        --wandb_project RecPO \
        --wandb_name SFT-4-gpu > log/sft.log


CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node 4 --master_port=25642 dpo.py \
        --model_name meta-llama/Llama-3.2-1B-Instruct  \
        --resume_from_checkpoint output/SFT-4-gpu/ \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --prompt_path ./prompt/movie2.txt \
        --logging_dir log/ \
        --output_dir output/ \
        --learning_rate 1e-5 \
        --beta 1 \
        --neg_num 1 \
        --num_train_epochs 5 \
        --eval_step 0.1 \
        --report_to wandb \
        --wandb_project RecPO \
        --wandb_name DPO-4-gpu > log/dpo.log


CUDA_VISIBLE_DEVICES=0,1,6,7 torchrun --nproc_per_node 4 --master_port=25641 cpo_simpo.py \
        --model_name meta-llama/Llama-3.2-1B-Instruct  \
        --resume_from_checkpoint output/SFT-4-gpu/ \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --prompt_path ./prompt/movie2.txt \
        --logging_dir log/ \
        --output_dir output/ \
        --learning_rate 1e-5 \
        --beta 1 \
        --loss_type simpo \
        --simpo_gamma 0.5 \
        --cpo_alpha 0 \
        --neg_num 1 \
        --num_train_epochs 5 \
        --eval_step 0.1 \
        --report_to wandb \
        --wandb_project RecPO \
        --wandb_name SimPO-4-gpu > log/simpo.log


CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node 4 --master_port=25642 softmax_dpo.py \
        --model_name meta-llama/Llama-3.2-1B-Instruct  \
        --resume_from_checkpoint output/SFT-4-gpu/ \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --prompt_path ./prompt/movie2.txt \
        --logging_dir log/ \
        --output_dir output/ \
        --learning_rate 1e-5 \
        --beta 1 \
        --neg_num 3 \
        --num_train_epochs 5 \
        --eval_step 0.1 \
        --report_to wandb \
        --wandb_project RecPO \
        --wandb_name SDPO-4-gpu > log/sdpo.log


CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node 4 --master_port=25642 rec_cpo.py \
        --model_name meta-llama/Llama-3.2-1B-Instruct  \
        --resume_from_checkpoint output/SFT-4-gpu/ \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --prompt_path ./prompt/movie2.txt \
        --logging_dir log/ \
        --output_dir output/ \
        --learning_rate 1e-5 \
        --beta 1 \
        --margin_lambda 0.5 \
        --neg_num 1 \
        --num_train_epochs 5 \
        --eval_step 0.1 \
        --report_to wandb \
        --wandb_project RecPO \
        --wandb_name RecPO-4-gpu-neg1 > log/recpo_neg1.log


CUDA_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node 4 --master_port=25642 rec_dpo.py \
        --model_name meta-llama/Llama-3.2-1B-Instruct  \
        --resume_from_checkpoint output/SFT-4-gpu/ \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --prompt_path ./prompt/movie2.txt \
        --logging_dir log/ \
        --output_dir output/ \
        --learning_rate 1e-5 \
        --beta 1 \
        --margin_lambda 0.5 \
        --neg_num 3 \
        --num_train_epochs 5 \
        --eval_step 0.1 \
        --user_score True \
        --report_to wandb \
        --wandb_project RecPO \
        --wandb_name RecDPO-4-gpu > log/recdpo.log