#cd /home/ericwen/S-DPO
#python sft.py --prompt_path ./prompt/movie2.txt  --report_to none
#python sft.py --prompt_path ./prompt/movie2.txt  --wandb_name SFT-woRating-HighRate-Full

cd /home/ericwen/Rec-PO
#python sft.py --prompt_path ./prompt/movie2.txt  --report_to none
#python sft.py --prompt_path ./prompt/movie_rating2.txt  --wandb_name SFT-wRating-HighRate

## CPO experiment
#python rec_po.py --prompt_path ./prompt/movie_rating2.txt  --wandb_name CPO --resume_from_checkpoint output/sft_checkpoint/ --loss_type sigmoid --cpo_alpha 1 --ln False

## SimPO Experiment
#python rec_po.py --prompt_path ./prompt/movie_rating2.txt  --wandb_name SimPO --resume_from_checkpoint output/sft_checkpoint/ --loss_type simpo --cpo_alpha 0 --simpo_gamma 0.5 --ln True

## CPO-SimPO experiment
#python rec_po.py --prompt_path ./prompt/movie_rating2.txt  --wandb_name CPO-SimPO --resume_from_checkpoint output/sft_checkpoint/ --loss_type simpo --cpo_alpha 1 --simpo_gamma 0.5 -- ln True

## RecPO experiment
python rec_po.py --prompt_path ./prompt/movie_rating2.txt --report_to wandb  --wandb_name RecPO --resume_from_checkpoint output/sft_checkpoint/ --neg_num 1 --loss_type sigmoid --cpo_alpha 0 --margin_lambda 0.5  --ln True --use_score True