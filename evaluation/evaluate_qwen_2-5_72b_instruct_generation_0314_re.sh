
#source /map-vepfs/miniconda3/bin/activate
#conda activate yubo_lf

export CUDA_VISIBLE_DEVICES=4,5,6,7

python evaluate_qwen_2-5_72b_instruct_generation_re_0314.py
