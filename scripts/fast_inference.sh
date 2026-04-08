run_name=${1:-first_run}
sample_len=${2:-100}
num_samples=${3:-1}
temperature=${4:-0.9}

model_path=exp/model_ckpt/$run_name
inference_path=exp/model_generation/${run_name}.jsonl
log_path=exp/logs/$run_name

mkdir -p exp/model_generation
mkdir -p $log_path

echo "--------------------Fast Inferencing--------------------"
echo "Model: $model_path"
echo "Samples: $sample_len, Num generations: $num_samples, Temp: $temperature"

python3 src/test/inference.py \
  --pretrained-path $model_path \
  --in-path data/sl_data/test.json \
  --out-path $inference_path \
  --num-samples $num_samples \
  --sample-len $sample_len \
  --temperature $temperature \
  --batch-size 64 \
  --model-name llama3 2>&1 | tee $log_path/inference.txt

echo "--------------------Done--------------------"
echo "Output: $inference_path"
