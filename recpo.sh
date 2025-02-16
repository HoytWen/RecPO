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
#CUDA_VISIBLE_DEVICES=0,1,6,7 torchrun --nproc_per_node 4 --master_port=25645 rec_dpo.py \
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
#        --margin_lambda 1 \
#        --ratio True \
#        --neg_num 3 \
#        --num_train_epochs 3 \
#        --eval_step 0.2 \
#        --user_score True \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-RecDPO-ratio-ml1-gpu4 > log/recdpo.log
#
#CUDA_VISIBLE_DEVICES=0,1,6,7 torchrun --nproc_per_node 4 --master_port=25645 rec_dpo.py \
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
#        --margin_lambda 2 \
#        --ratio True \
#        --neg_num 3 \
#        --num_train_epochs 3 \
#        --eval_step 0.2 \
#        --user_score True \
#        --report_to wandb \
#        --wandb_project RecPO \
#        --wandb_name Base-8B-RecDPO-ratio-ml2-gpu4 > log/recdpo.log
#
#
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 --master_port=25645 rec_dpo.py \
#        --model_name meta-llama/Llama-3.1-8B \
#        --resume_from_checkpoint output/beeradvocate/Base-8B-SFT-gpu4/final_checkpoint/ \
#        --train_dataset beeradvocate_10000 \
#        --batch_size 2 \
#        --gradient_accumulation_steps 8 \
#        --prompt_path ./prompt/beer_rating1.txt \
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
#        --wandb_name Base-8B-RecDPO-ratio-ml0.5-gpu4 > log/recdpo.log