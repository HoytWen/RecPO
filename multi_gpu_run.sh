#CUDA_VISIBLE_DEVICES=0,1,6,7 torchrun --nproc_per_node 4 --master_port=25642 dpo.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --resume_from_checkpoint output/beeradvocate/Base-8B-SFT-gpu4/final_checkpoint/ \
#        --train_dataset beeradvocate_10000 \
#        --batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/beer_rating1.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --loss_type sigmoid \
#        --beta 1 \
#        --neg_num 1 \
#        --num_train_epochs 3 \
#        --eval_step 0.2 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-DPO-gpu4 > log/dpo.log
#
#CUDA_VISIBLE_DEVICES=0,1,6,7 torchrun --nproc_per_node 4 --master_port=25642 softmax_dpo.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --resume_from_checkpoint output/beeradvocate/Base-8B-SFT-gpu4/final_checkpoint/ \
#        --batch_size 4 \
#        --train_dataset beeradvocate_10000 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/beer_rating1.txt \
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
#CUDA_VISIBLE_DEVICES=0,1,6,7 torchrun --nproc_per_node 4 --master_port=25641 sft.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --train_dataset beeradvocate_10000 \
#        --batch_size 4 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/beer_rating1.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --num_train_epochs 5 \
#        --eval_step 0.2 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-SFT-P1-gpu4 > log/base-sft.log

#CUDA_VISIBLE_DEVICES=2,3,6,7 torchrun --nproc_per_node 4 --master_port=25641 sft.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --train_dataset amazon-books_10000 \
#        --batch_size 2 \
#        --gradient_accumulation_steps 16 \
#        --prompt_path ./prompt/book_rating.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --num_train_epochs 5 \
#        --eval_step 0.2 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-SFT-seq1024-gpu4 > log/base-sft
#
#CUDA_VISIBLE_DEVICES=2,3,6,7 torchrun --nproc_per_node 4 --master_port=25641 sft.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --train_dataset amazon-books-wolow_10000 \
#        --batch_size 2 \
#        --gradient_accumulation_steps 16 \
#        --prompt_path ./prompt/book.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --num_train_epochs 5 \
#        --eval_step 0.2 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-SFT-woLow-woRating-gpu4 > log/base-sft.log
#
#CUDA_VISIBLE_DEVICES=2,3,6,7 torchrun --nproc_per_node 4 --master_port=25641 sft.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --train_dataset amazon-books-wolow_10000 \
#        --batch_size 2 \
#        --gradient_accumulation_steps 16 \
#        --prompt_path ./prompt/book_rating.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --num_train_epochs 5 \
#        --eval_step 0.2 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-SFT-woLow-gpu4 > log/base-sft.log


#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port=25641 softmax_dpo.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --resume_from_checkpoint output/amazon-books/Base-8B-SFT-gpu8/final_checkpoint/ \
#        --batch_size 2 \
#        --train_dataset amazon-books_10000 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/book_rating.txt \
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
#        --wandb_name Base-8B-SDPO-gpu8 > log/sdpo.log


#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port=25641 rec_dpo.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --resume_from_checkpoint output/amazon-books/Base-8B-SFT-gpu8/final_checkpoint/ \
#        --train_dataset amazon-books_10000 \
#        --batch_size 2 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/book_rating.txt \
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
#        --prompt_cutoff_len 924 \
#        --cutoff_len 1024 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-RecDPO-ratio-ml2-gpu8 > log/recdpo


#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 torchrun --nproc_per_node 8 --master_port=25641 sft.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --train_dataset movielens_10000 \
#        --batch_size 2 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/movie_rating1.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --num_train_epochs 5 \
#        --eval_step 0.2 \
#        --cutoff_len 768 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-SFT-gpu8 > log/base-sft


#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 torchrun --nproc_per_node 8 --master_port=25641 rec_dpo.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --resume_from_checkpoint output/movielens/Base-8B-SFT-gpu8/final_checkpoint/ \
#        --train_dataset movielens_10000 \
#        --batch_size 2 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/movie_rating1.txt \
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
#        --prompt_cutoff_len 704 \
#        --cutoff_len 768 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-RecDPO-ratio-ml2-gpu8 > log/recdpo
#
#
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 torchrun --nproc_per_node 8 --master_port=25641 softmax_dpo.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --resume_from_checkpoint output/movielens/Base-8B-SFT-gpu8/final_checkpoint/ \
#        --batch_size 2 \
#        --train_dataset movielens_10000 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/movie_rating1.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --loss_type sigmoid \
#        --beta 1 \
#        --neg_num 3 \
#        --num_train_epochs 3 \
#        --eval_step 0.2 \
#        --prompt_cutoff_len 704 \
#        --cutoff_len 768 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-SDPO-gpu8 > log/sdpo.log
#
#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 torchrun --nproc_per_node 8 --master_port=25641 softmax_dpo.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --resume_from_checkpoint output/amazon-books/Base-8B-SFT-gpu8/final_checkpoint/ \
#        --batch_size 2 \
#        --train_dataset amazon-books_10000 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/book_rating.txt \
#        --logging_dir log/ \
#        --output_dir output/ \
#        --learning_rate 1e-5 \
#        --loss_type sigmoid \
#        --beta 1 \
#        --neg_num 3 \
#        --num_train_epochs 3 \
#        --eval_step 0.2 \
#        --prompt_cutoff_len 924 \
#        --cutoff_len 1024 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-SDPO-gpu8 > log/sdpo.log



# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 torchrun --nproc_per_node 8 --master_port=25642 cpo_simpo.py \
#         --model_name meta-llama/Llama-3.1-8B \
#         --resume_from_checkpoint output/movielens/Base-8B-SFT-gpu8/final_checkpoint/ \
#         --train_dataset movielens_10000 \
#         --batch_size 2 \
#         --gradient_accumulation_steps 8 \
#         --prompt_path ./prompt/movie_rating1.txt \
#         --logging_dir log/ \
#         --output_dir output/ \
#         --learning_rate 1e-5 \
#         --beta 1 \
#         --loss_type simpo \
#         --simpo_gamma 2 \
#         --cpo_alpha 0 \
#         --neg_num 1 \
#         --num_train_epochs 3 \
#         --eval_step 0.2 \
#         --prompt_cutoff_len 704 \
#         --cutoff_len 768 \
#         --report_to wandb \
#         --wandb_project RecPO \
#         --wandb_name Base-SimPO-gpu8 > log/simpo.log


# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 torchrun --nproc_per_node 8 --master_port=25642 cpo_simpo.py \
#         --model_name meta-llama/Llama-3.1-8B \
#         --resume_from_checkpoint output/amazon-books/Base-8B-SFT-gpu8/final_checkpoint/ \
#         --train_dataset amazon-books_10000 \
#         --batch_size 2 \
#         --gradient_accumulation_steps 8 \
#         --prompt_path ./prompt/book_rating.txt \
#         --logging_dir log/ \
#         --output_dir output/ \
#         --learning_rate 1e-5 \
#         --beta 1 \
#         --loss_type simpo \
#         --simpo_gamma 2 \
#         --cpo_alpha 0 \
#         --neg_num 1 \
#         --num_train_epochs 3 \
#         --eval_step 0.2 \
#         --prompt_cutoff_len 924 \
#         --cutoff_len 1024 \
#         --report_to wandb \
#         --wandb_project RecPO \
#         --wandb_name Base-SimPO-gpu8 > log/simpo.log


# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 torchrun --nproc_per_node 8 --master_port=25642 cpo_simpo.py \
#         --model_name meta-llama/Llama-3.1-8B \
#         --resume_from_checkpoint output/beeradvocate/Base-8B-SFT-gpu8/final_checkpoint/ \
#         --train_dataset beeradvocate_10000 \
#         --batch_size 2 \
#         --gradient_accumulation_steps 8 \
#         --prompt_path ./prompt/beer_rating1.txt \
#         --logging_dir log/ \
#         --output_dir output/ \
#         --learning_rate 1e-5 \
#         --beta 1 \
#         --loss_type simpo \
#         --simpo_gamma 2 \
#         --cpo_alpha 0 \
#         --neg_num 1 \
#         --num_train_epochs 3 \
#         --prompt_cutoff_len 924 \
#         --cutoff_len 1024 \
#         --eval_step 0.2 \
#         --report_to wandb \
#         --wandb_project RecPO \
#         --wandb_name Base-SimPO-gpu8 > log/simpo.log


 CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 torchrun --nproc_per_node 8 --master_port=25641 sft.py \
        --model_name Qwen/Qwen2.5-7B \
        --train_dataset amazon-books_10000 \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --prompt_path ./prompt/book_rating.txt \
        --logging_dir log/ \
        --output_dir output/ \
        --learning_rate 1e-5 \
        --num_train_epochs 5 \
        --eval_step 0.2 \
        --cutoff_len 1024 \
        --report_to none \
        --wandb_project RecPO \
        --wandb_name Base-qwen-7B-SFT-gpu8 > log/base-sft
#
#
 CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 torchrun --nproc_per_node 8 --master_port=25641 sft.py \
        --model_name Qwen/Qwen2.5-7B \
        --train_dataset beeradvocate_10000 \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --prompt_path ./prompt/beer_rating.txt \
        --logging_dir log/ \
        --output_dir output/ \
        --learning_rate 1e-5 \
        --num_train_epochs 5 \
        --eval_step 0.2 \
        --cutoff_len 1024 \
        --report_to none \
        --wandb_project RecPO \
        --wandb_name Base-qwen-7B-SFT-gpu8 > log/base-sft


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port=25642 dpo.py \
       --model_name Qwen/Qwen2.5-7B \
       --resume_from_checkpoint output/amazon-books/Base-qwen-7B-SFT-gpu8/final_checkpoint/ \
       --train_dataset amazon-books_10000 \
       --batch_size 2 \
       --gradient_accumulation_steps 16 \
       --prompt_path ./prompt/book_rating.txt \
       --logging_dir log/ \
       --output_dir output/ \
       --learning_rate 1e-5 \
       --loss_type sigmoid \
       --beta 1 \
       --neg_num 1 \
       --num_train_epochs 3 \
       --prompt_cutoff_len 924 \
       --cutoff_len 1024 \
       --eval_step 0.2 \
       --report_to none \
       --wandb_project RecPO \
       --wandb_name Base-qwen-7B-DPO-gpu4 > log/dpo.log

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port=25642 dpo.py \
       --model_name Qwen/Qwen2.5-7B \
       --resume_from_checkpoint output/beeradvocate/Base-qwen-7B-SFT-gpu8/final_checkpoint/ \
       --train_dataset beeradvocate_10000 \
       --batch_size 2 \
       --gradient_accumulation_steps 16 \
       --prompt_path ./prompt/beer_rating.txt \
       --logging_dir log/ \
       --output_dir output/ \
       --learning_rate 1e-5 \
       --loss_type sigmoid \
       --beta 1 \
       --neg_num 1 \
       --num_train_epochs 3 \
       --prompt_cutoff_len 924 \
       --cutoff_len 1024 \
       --eval_step 0.2 \
       --report_to none \
       --wandb_project RecPO \
       --wandb_name Base-qwen-7B-DPO-gpu4 > log/dpo.log


#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 torchrun --nproc_per_node 8 --master_port=25641 rec_dpo.py \
#        --model_name Qwen/Qwen2.5-7B \
#        --resume_from_checkpoint output/movielens/Base-qwen-7B-SFT-gpu8/final_checkpoint/ \
#        --train_dataset movielens_10000 \
#        --batch_size 2 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/movie_rating.txt \
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
#        --prompt_cutoff_len 704 \
#        --cutoff_len 768 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-qwen-7B-RecDPO-ratio-ml2-gpu8 > log/recdpo



#CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 torchrun --nproc_per_node 8 --master_port=25641 rec_dpo.py \
#        --model_name Qwen/Qwen2.5-7B \
#        --resume_from_checkpoint output/steam/Base-qwen-7B-SFT-gpu8/final_checkpoint/ \
#        --train_dataset steam_10000 \
#        --batch_size 2 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/game_rating.txt \
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
#        --prompt_cutoff_len 704 \
#        --cutoff_len 768 \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-qwen-7B-RecDPO-ratio-ml2-gpu8 > log/recdpo


CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 torchrun --nproc_per_node 8 --master_port=25641 rec_dpo.py \
        --model_name Qwen/Qwen2.5-7B \
        --resume_from_checkpoint output/beeradvocate/Base-qwen-7B-SFT-gpu8/final_checkpoint/ \
        --train_dataset beeradvocate_10000 \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --prompt_path ./prompt/beer_rating.txt \
        --logging_dir log/ \
        --output_dir output/ \
        --learning_rate 1e-5 \
        --loss_type sigmoid \
        --beta 1 \
        --margin_lambda 2 \
        --ratio True \
        --neg_num 3 \
        --num_train_epochs 3 \
        --eval_step 0.2 \
        --user_score True \
        --prompt_cutoff_len 924 \
        --cutoff_len 1024 \
        --report_to none \
        --wandb_project RecPO \
        --wandb_name Base-qwen-7B-RecDPO-ratio-ml2-gpu8 > log/recdpo


CUDA_VISIBLE_DEVICES=2,3,4,5,6,7,0,1 torchrun --nproc_per_node 8 --master_port=25641 rec_dpo.py \
        --model_name Qwen/Qwen2.5-7B \
        --resume_from_checkpoint output/amazon-books/Base-qwen-7B-SFT-gpu8/final_checkpoint/ \
        --train_dataset amazon-books_10000 \
        --batch_size 2 \
        --gradient_accumulation_steps 8 \
        --prompt_path ./prompt/book_rating.txt \
        --logging_dir log/ \
        --output_dir output/ \
        --learning_rate 1e-5 \
        --loss_type sigmoid \
        --beta 1 \
        --margin_lambda 2 \
        --ratio True \
        --neg_num 3 \
        --num_train_epochs 3 \
        --eval_step 0.2 \
        --user_score True \
        --prompt_cutoff_len 924 \
        --cutoff_len 1024 \
        --report_to none \
        --wandb_project RecPO \
        --wandb_name Base-qwen-7B-RecDPO-ratio-ml2-gpu8 > log/recdpo