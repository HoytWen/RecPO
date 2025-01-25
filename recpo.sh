#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25645 rec_dpo.py \
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
#        --margin_lambda 0.5 \
#        --ratio True \
#        --neg_num 3 \
#        --negative_selection random \
#        --num_train_epochs 3 \
#        --eval_step 0.2 \
#        --user_score True \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-RecDPO-ratio-gpu4 > log/recdpo.log


#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25645 rec_dpo.py \
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
#        --margin_lambda 0.5 \
#        --ratio True \
#        --neg_num 3 \
#        --num_train_epochs 3 \
#        --eval_step 0.2 \
#        --user_score True \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-RecDPO-ratio-gpu4 > log/recdpo.log
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25645 rec_dpo.py \
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
#        --margin_lambda 1 \
#        --ratio True \
#        --neg_num 3 \
#        --num_train_epochs 3 \
#        --eval_step 0.2 \
#        --user_score True \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-RecDPO-ratio-ml1-gpu4 > log/recdpo.log

#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25645 rec_dpo.py \
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
#        --margin_lambda 1 \
#        --ratio True \
#        --neg_num 3 \
#        --negative_selection random \
#        --num_train_epochs 3 \
#        --eval_step 0.2 \
#        --user_score True \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-RecDPO-random-ratio-ml1-gpu4 > log/recdpo.log


#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25645 rec_dpo.py \
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
#        --margin_lambda 1.5 \
#        --ratio True \
#        --neg_num 3 \
#        --negative_selection random \
#        --num_train_epochs 3 \
#        --eval_step 0.2 \
#        --user_score True \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-RecDPO-random-ratio-ml1.5-gpu4 > log/recdpo.log
#
#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25645 rec_dpo.py \
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
#        --margin_lambda 1.5 \
#        --ratio True \
#        --neg_num 3 \
#        --num_train_epochs 3 \
#        --eval_step 0.2 \
#        --user_score True \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-RecDPO-ratio-ml1.5-gpu4 > log/recdpo.log

#CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=25645 rec_dpo.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --resume_from_checkpoint output/amazon-books/Base-8B-SFT-gpu4/final_checkpoint/ \
#        --train_dataset amazon-books_10000 \
#        --batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/book_rating1.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --loss_type sigmoid \
#        --beta 1 \
#        --margin_lambda 1 \
#        --ratio True \
#        --neg_num 3 \
#        --num_train_epochs 3 \
#        --eval_step 0.2 \
#        --user_score True \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-RecDPO-ratio-ml1-gpu4 > log/recdpo.log

#CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node 4 --master_port=25645 rec_dpo.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --resume_from_checkpoint output/amazon-books/Base-8B-SFT-gpu4/final_checkpoint/ \
#        --train_dataset amazon-books_10000 \
#        --batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/book_rating1.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --loss_type sigmoid \
#        --beta 2 \
#        --margin_lambda 2 \
#        --ratio True \
#        --neg_num 3 \
#        --num_train_epochs 3 \
#        --eval_step 0.2 \
#        --user_score True \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-RecDPO-ratio-beta2-ml2-gpu4 > log/recdpo.log

#CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25641 rec_dpo.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --resume_from_checkpoint output/amazon-books/Base-8B-SFT-gpu4/final_checkpoint/ \
#        --train_dataset amazon-books_10000 \
#        --batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/book_rating1.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --loss_type sigmoid \
#        --beta 1.5 \
#        --margin_lambda 1 \
#        --ratio True \
#        --neg_num 3 \
#        --num_train_epochs 3 \
#        --eval_step 0.2 \
#        --user_score True \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-RecDPO-ratio-beta1.5-ml1-gpu4 > log/recdpo.log

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 --master_port=25643 rec_dpo.py \
        --model_name meta-llama/Llama-3.1-8B \
        --resume_from_checkpoint output/amazon-books/Base-8B-SFT-gpu4/final_checkpoint/ \
        --train_dataset amazon-books_10000 \
        --batch_size 4 \
        --gradient_accumulation_steps 8 \
        --prompt_path ./prompt/book_rating1.txt \
        --logging_dir log/ \
        --output_dir output/ \
        --learning_rate 1e-5 \
        --loss_type sigmoid \
        --beta 1 \
        --margin_lambda 2 \
        --ratio True \
        --sft_weight 1 \
        --neg_num 3 \
        --num_train_epochs 3 \
        --eval_step 0.2 \
        --user_score True \
        --report_to wandb \
        --wandb_project RecPO \
        --wandb_name Base-8B-RecDPO-wSFT-ratio-ml2-gpu4 > log/recdpo.log