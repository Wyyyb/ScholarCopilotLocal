
source /map-vepfs/miniconda3/bin/activate
conda activate yubo_lf

export CUDA_VISIBLE_DEVICES=7

python evaluate_qwen_2-5_7b_instruct_generation_0305_gt.py
