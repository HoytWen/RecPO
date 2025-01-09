CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25645 softmax_dpo.py \
        --model_name meta-llama/Llama-3.1-8B \
        --resume_from_checkpoint output/Base-8B-SFT-gpu4/ \
        --batch_size 4 \
        --train_dataset 10000 \
        --gradient_accumulation_steps 8 \
        --prompt_path ./prompt/movie_rating2.txt \
        --logging_dir log/ \
        --output_dir output/ \
        --learning_rate 1e-5 \
        --loss_type sigmoid \
        --beta 1 \
        --neg_num 3 \
        --num_train_epochs 3 \
        --eval_step 0.2 \
        --report_to wandb \
        --wandb_project RecPO \
        --wandb_name Base-8B-SDPO-gpu4 > log/sdpo.log


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25645 softmax_dpo.py \
        --model_name meta-llama/Llama-3.1-8B \
        --resume_from_checkpoint output/Base-8B-SFT-gpu4/ \
        --batch_size 4 \
        --train_dataset 10000 \
        --gradient_accumulation_steps 8 \
        --prompt_path ./prompt/movie_rating2.txt \
        --logging_dir log/ \
        --output_dir output/ \
        --learning_rate 1e-5 \
        --loss_type sigmoid \
        --beta 1 \
        --sft_weight 1 \
        --neg_num 3 \
        --num_train_epochs 3 \
        --eval_step 0.2 \
        --report_to wandb \
        --wandb_project RecPO \
        --wandb_name Base-8B-SDPO-wSFT-gpu4 > log/sdpo.log


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25643 rec_dpo.py \
        --model_name meta-llama/Llama-3.1-8B \
        --resume_from_checkpoint output/Base-8B-SFT-gpu4/ \
        --train_dataset 10000 \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --prompt_path ./prompt/movie_rating2.txt \
        --logging_dir log/ \
        --output_dir output/ \
        --learning_rate 1e-5 \
        --loss_type sigmoid \
        --beta 1 \
        --margin_lambda 0.5 \
        --neg_num 3 \
        --num_train_epochs 3 \
        --eval_step 0.2 \
        --user_score True \
        --report_to wandb \
        --wandb_project RecPO \
        --wandb_name Base-8B-RecDPO-gpu4 > log/recdpo.log


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25643 rec_dpo.py \
        --model_name meta-llama/Llama-3.1-8B \
        --resume_from_checkpoint output/Base-8B-SFT-gpu4/ \
        --train_dataset 10000 \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --prompt_path ./prompt/movie_rating2.txt \
        --logging_dir log/ \
        --output_dir output/ \
        --learning_rate 1e-5 \
        --loss_type sigmoid \
        --beta 1 \
        --sft_weight 1 \
        --margin_lambda 0.5 \
        --sft_weight 1 \
        --num_train_epochs 3 \
        --eval_step 0.2 \
        --user_score True \
        --report_to wandb \
        --wandb_project RecPO \
        --wandb_name Base-8B-RecDPO-wSFT-gpu4 > log/recdpo.log


##### RecCPO
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25643 rec_cpo.py \
#        --model_name meta-llama/Llama-3.2-1B-Instruct \
#        --resume_from_checkpoint output/Instruct-SFT-gpu4/ \
#        --train_dataset 10000 \
#        --batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/movie_rating2.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --loss_type sigmoid \
#        --beta 1 \
#        --sft_weight 1 \
#        --ln False \
#        --user_score True \
#        --margin_lambda 0.5 \
#        --neg_num 3 \
#        --num_train_epochs 5 \
#        --eval_step 0.1 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Instruct-RecCPO-gpu4 > log/reccpo.log
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25642 rec_cpo.py \
#        --model_name meta-llama/Llama-3.2-1B-Instruct \
#        --resume_from_checkpoint output/Instruct-SFT-gpu4/ \
#        --train_dataset 10000 \
#        --batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/movie_rating2.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --loss_type sigmoid \
#        --beta 1 \
#        --sft_weight 1 \
#        --simpo_gamma 0.0 \
#        --ln True \
#        --user_score True \
#        --margin_lambda 0.5 \
#        --neg_num 3 \
#        --num_train_epochs 5 \
#        --eval_step 0.1 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Instruct-RecCPO-wLN-gpu4 > log/reccpo.log
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25642 rec_cpo.py \
#        --model_name meta-llama/Llama-3.2-1B-Instruct \
#        --resume_from_checkpoint output/Instruct-SFT-gpu4/ \
#        --train_dataset 10000 \
#        --batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/movie_rating2.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --loss_type simpo \
#        --beta 1 \
#        --sft_weight 1 \
#        --simpo_gamma 0.5 \
#        --ln True \
#        --user_score True \
#        --margin_lambda 0.5 \
#        --neg_num 3 \
#        --num_train_epochs 5 \
#        --eval_step 0.1 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Instruct-RecSimPO-wSFT-gpu4 > log/recsimpo.log
