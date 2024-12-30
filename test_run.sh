#cd /home/ericwen/S-DPO
#python sft.py --prompt_path ./prompt/movie2.txt  --wandb_name SFT-woRating-HighRate-Full

cd /home/ericwen/Rec-PO
#python sft.py --prompt_path ./prompt/movie_rating2.txt  --wandb_name SFT-wRating-HighRate
#python rec_po.py --prompt_path ./prompt/movie_rating2.txt  --wandb_name CPO --resume_from_checkpoint output/sft_checkpoint/ --loss_type sigmoid
#python rec_po.py --prompt_path ./prompt/movie_rating2.txt  --wandb_name CPO-SimPO --resume_from_checkpoint output/sft_checkpoint/ --loss_type simpo
#python rec_po.py --prompt_path ./prompt/movie_rating2.txt  --wandb_name SimPO --resume_from_checkpoint output/sft_checkpoint/ --cpo_alpha 0 --loss_type simpo
python rec_po.py --prompt_path ./prompt/movie_rating2.txt  --wandb_name RecPO --resume_from_checkpoint output/sft_checkpoint/ --cpo_alpha 0 --loss_type sigmoid --margin_lambda 0.5