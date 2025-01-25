CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=25641 dpo.py \
        --model_name meta-llama/Llama-3.1-8B \
        --resume_from_checkpoint output/steam/Base-8B-SFT-gpu4/final_checkpoint/ \
        --train_dataset steam_10000 \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --prompt_path ./prompt/game_rating.txt \
        --logging_dir log/ \
        --output_dir output/ \
        --learning_rate 1e-5 \
        --loss_type sigmoid \
        --beta 1 \
        --neg_num 1 \
        --num_train_epochs 3 \
        --eval_step 0.2 \
        --report_to wandb \
        --wandb_project RecPO \
        --wandb_name Base-8B-DPO-gpu4 > log/dpo.log

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=25641 softmax_dpo.py \
        --model_name meta-llama/Llama-3.1-8B \
        --resume_from_checkpoint output/steam/Base-8B-SFT-gpu4/final_checkpoint/ \
        --batch_size 4 \
        --train_dataset steam_10000 \
        --gradient_accumulation_steps 8 \
        --prompt_path ./prompt/game_rating.txt \
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

#CUDA_VISIBLE_DEVICES=2,3,0,1 torchrun --nproc_per_node 4 --master_port=25641 sft.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --train_dataset amazon-books_10000 \
#        --batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/book_rating2.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --num_train_epochs 5 \
#        --eval_step 0.2 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-SFT-P2-gpu4 > log/base-sft.log


# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25641 rec_dpo.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --resume_from_checkpoint output/amazon-books/Base-8B-SFT-P2-gpu4/final_checkpoint/ \
#        --train_dataset amazon-books_10000 \
#        --batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/book_rating2.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --loss_type sigmoid \
#        --beta 1 \
#        --margin_lambda 2 \
#        --ratio True \
#        --neg_num 3 \
#        --num_train_epochs 3 \
#        --eval_step 0.2 \
#        --user_score True \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-RecDPO-P2-ratio-ml2-gpu4 > log/recdpo.log



#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25641 sft.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --train_dataset steam-origin_10000 \
#        --batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/game_rating.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --num_train_epochs 5 \
#        --eval_step 0.2 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-SFT-gpu4 > log/base-sft.log

#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25641 dpo.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --resume_from_checkpoint output/Base-8B-SFT-P2-gpu4/ \
#        --train_dataset 10000 \
#        --batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/movie_rating.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --loss_type sigmoid \
#        --beta 1 \
#        --neg_num 1 \
#        --num_train_epochs 3 \
#        --eval_step 0.1 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-DPO-P2-gpu4 > log/dpo.log

#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25643 rec_dpo.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --resume_from_checkpoint output/Base-8B-SFT-P1-gpu4/ \
#        --train_dataset 10000 \
#        --batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/movie_rating1.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --loss_type sigmoid \
#        --beta 1 \
#        --sft_weight 1 \
#        --margin_lambda 0.5 \
#        --sft_weight 1 \
#        --num_train_epochs 3 \
#        --eval_step 0.2 \
#        --user_score True \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-RecDPO-wSFT-P1-gpu4 > log/recdpo.log
#
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25643 rec_dpo.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --resume_from_checkpoint output/Base-8B-SFT-P2-gpu4/ \
#        --train_dataset 10000 \
#        --batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/movie_rating.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --loss_type sigmoid \
#        --beta 1 \
#        --sft_weight 1 \
#        --margin_lambda 0.5 \
#        --sft_weight 1 \
#        --num_train_epochs 3 \
#        --eval_step 0.2 \
#        --user_score True \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-RecDPO-wSFT-P2-gpu4 > log/recdpo.log


#CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=25641 dpo.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --resume_from_checkpoint output/Base-8B-SFT-gpu4/ \
#        --train_dataset 10000 \
#        --batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/movie_rating.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --loss_type sigmoid \
#        --beta 1 \
#        --neg_num 1 \
#        --num_train_epochs 3 \
#        --eval_step 0.1 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-DPO-gpu4 > log/dpo.log
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=25641 dpo.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --resume_from_checkpoint output/Base-8B-SFT-gpu4/ \
#        --train_dataset 10000 \
#        --batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/movie_rating.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --loss_type sigmoid \
#        --beta 1 \
#        --rpo_alpha 1 \
#        --neg_num 1 \
#        --num_train_epochs 3 \
#        --eval_step 0.1 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-DPO-wSFT-gpu4 > log/dpo.log


#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25642 cpo_simpo.py \
#        --model_name meta-llama/Llama-3.2-1B \
#        --resume_from_checkpoint output/Base-SFT-gpu4/ \
#        --train_dataset 10000 \
#        --batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/movie_rating.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --beta 1 \
#        --loss_type sigmoid \
#        --simpo_gamma 0 \
#        --cpo_alpha 1 \
#        --neg_num 1 \
#        --num_train_epochs 5 \
#        --eval_step 0.1 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-CPO-wSFT-gpu4 > log/cpo.log

#CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=25645 softmax_dpo.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --resume_from_checkpoint output/Base-8B-SFT-gpu4/ \
#        --batch_size 4 \
#        --train_dataset 10000 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/movie_rating.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --loss_type sigmoid \
#        --beta 1 \
#        --neg_num 3 \
#        --num_train_epochs 3 \
#        --eval_step 0.2 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-SDPO-gpu4 > log/sdpo.log
#
#
#CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=25645 softmax_dpo.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --resume_from_checkpoint output/Base-8B-SFT-gpu4/ \
#        --batch_size 4 \
#        --train_dataset 10000 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/movie_rating.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --loss_type sigmoid \
#        --beta 1 \
#        --sft_weight 1 \
#        --neg_num 3 \
#        --num_train_epochs 3 \
#        --eval_step 0.2 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-SDPO-wSFT-gpu4 > log/sdpo.log
