cd /home/ericwen/S-DPO
python sft.py --prompt_path ./prompt/movie.txt  --wandb_name SFT-woRating-Next
python sft.py --prompt_path ./prompt/movie2.txt  --wandb_name SFT-woRating-HighRate

cd /home/ericwen/Rec-PO
python sft.py --prompt_path ./prompt/movie_rating.txt  --wandb_name SFT-wRating-Next
python sft.py --prompt_path ./prompt/movie_rating2.txt  --wandb_name SFT-wRating-HighRate