
source /map-vepfs/miniconda3/bin/activate
conda activate yubo_lf

export CUDA_VISIBLE_DEVICES=5,7

python evaluate_qwen_2-5_72b_instruct_generation_0226.py