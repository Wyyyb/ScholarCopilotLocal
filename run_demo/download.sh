mkdir -p ../data
cd ../data

huggingface-cli download TIGER-Lab/ScholarCopilot-Data-v1 --local-dir . --repo-type dataset

wget https://huggingface.co/datasets/ubowang/scholarcopilot_re_eval_data_0226/resolve/main/eval_re_data_1k_0225.json

mkdir -p ../model_v1208
cd ../model_v1208

huggingface-cli download TIGER-Lab/ScholarCopilot-v1 --local-dir . --repo-type model

