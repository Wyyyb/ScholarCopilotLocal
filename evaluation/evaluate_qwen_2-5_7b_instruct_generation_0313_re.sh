
#source /map-vepfs/miniconda3/bin/activate
#conda activate yubo_lf

export CUDA_VISIBLE_DEVICES=1

cd ..

python evaluation/evaluate_qwen_2-5_7b_instruct_generation_re_0313.py
