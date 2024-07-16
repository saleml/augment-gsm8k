### Config
```
# Create and activate the environment
conda create -n gfnlm python=3.10
conda activate gfnlm

# Install main dependencies
conda install -c pytorch pytorch
conda install -c huggingface transformers
conda install -c conda-forge datasets
conda install pandas
conda install tqdm
conda install -c conda-forge pyarrow

# Install additional dependencies via pip
pip install accelerate
```

### Launching data agumentation

To use the script, you can run:
```
python augment_gsm8k.py --model gpt-neo-2.7B --num_examples 3 --output_file augmented_dataset --output_format csv --max_samples 1000 --batch_size 10 --use_gpu --dataset gsm8k --dataset_config main --dataset_split train --checkpoint_interval 100 --num_processes 4
```