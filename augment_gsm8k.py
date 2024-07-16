import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import pandas as pd
import argparse
from tqdm import tqdm
import logging
import json
from pathlib import Path
import multiprocessing as mp
import os
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate similar reasoning examples based on input dataset.")
    parser.add_argument("--model", type=str, default="gpt-neo-2.7B", help="Model name to use for generation")
    parser.add_argument("--num_examples", type=int, default=3, help="Number of examples to generate for each input")
    parser.add_argument("--output_file", type=str, default="augmented_dataset", help="Path to save the output file (without extension)")
    parser.add_argument("--output_format", type=str, choices=['csv', 'json', 'parquet'], default='csv', help="Output file format")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--dataset", type=str, default="gsm8k", help="Dataset to use (default: gsm8k)")
    parser.add_argument("--dataset_config", type=str, default="main", help="Dataset configuration")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split to use")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="Save checkpoint every N samples")
    parser.add_argument("--num_processes", type=int, default=1, help="Number of processes to use for multiprocessing")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def generate_prompt(example, num_examples, use_system_prompt=False):
    system_prompt = ("You are a highly efficient assistant, who generates examples of reasoning problems and their solution steps. "
                     "This process will be used to create a high quality dataset on which a reward model is trained. "
                     "This reward function later on optimized by an LLM to improve its reasoning abilities."
                     )
    base_prompt = (
        f"Generate {num_examples} reasoning problems and solutions similar to the following example. "
        "You can modify the style, the numbers (but adjust the math accordingly), the objects, the characters... "
        "The format should be: question (ending with a question mark), followed by reasoning steps separated by full stops. "
        f"Provide {num_examples} problems and solutions in the same format, separated by two line breaks. "
        f"Make sure your output contains exactly {num_examples} blocks of text separated by two line breaks, no quotes, no extra spaces, no additional new lines.\n\n"
    )
    return (system_prompt if use_system_prompt else '') + base_prompt + example['question'] + "\n" + example['answer'] + "\n\n"

def generate_examples_batch(model, tokenizer, prompts, num_examples):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=2048, num_return_sequences=1)
    responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    all_examples = []
    for response in responses:
        generated_examples = response.split('\n\n')[:num_examples]
        all_examples.extend(generated_examples)
    
    return all_examples

def process_batch(args):
    batch, model, tokenizer, num_examples = args
    prompts = [generate_prompt(example, num_examples) for example in batch]
    generated_examples = generate_examples_batch(model, tokenizer, prompts, num_examples)
    
    augmented_data = []
    for gen_example in generated_examples:
        parts = gen_example.split('?', 1)
        if len(parts) == 2:
            question = parts[0] + '?'
            answer = parts[1].strip()
            augmented_data.append({'question': question, 'answer': answer})
    
    return augmented_data

def save_checkpoint(data, output_path, format):
    if format == 'csv':
        pd.DataFrame(data).to_csv(output_path, index=False)
    elif format == 'json':
        with open(output_path, 'w') as f:
            json.dump(data, f)
    elif format == 'parquet':
        pd.DataFrame(data).to_parquet(output_path, index=False)

def process_dataset(model, tokenizer, dataset, args):
    augmented_data = []
    dataset = dataset[:args.max_samples] if args.max_samples else dataset
    
    checkpoint_path = f"{args.output_file}_checkpoint.pkl"
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        augmented_data = checkpoint['data']
        start_index = checkpoint['index']
        logging.info(f"Resuming from checkpoint at index {start_index}")
    else:
        start_index = 0

    with mp.Pool(processes=args.num_processes) as pool:
        for i in tqdm(range(start_index, len(dataset), args.batch_size), desc="Processing examples"):
            batch = dataset[i:i+args.batch_size]
            batch_args = (batch, model, tokenizer, args.num_examples)
            results = pool.apply(process_batch, (batch_args,))
            augmented_data.extend(results)

            if (i + args.batch_size) % args.checkpoint_interval == 0:
                checkpoint = {'data': augmented_data, 'index': i + args.batch_size}
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint, f)
                logging.info(f"Checkpoint saved at index {i + args.batch_size}")

    # Remove the checkpoint file after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return augmented_data

def main():
    args = parse_arguments()
    
    if args.config:
        config = load_config(args.config)
        args.__dict__.update(config)
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    logging.info(f"Using device: {device}")
    
    model_cache_path = f"{args.model}_cache.pkl"
    if os.path.exists(model_cache_path):
        logging.info("Loading model from cache")
        with open(model_cache_path, 'rb') as f:
            model, tokenizer = pickle.load(f)
    else:
        try:
            logging.info(f"Loading model: {args.model}")
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
            with open(model_cache_path, 'wb') as f:
                pickle.dump((model, tokenizer), f)
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return

    logging.info(f"Loading dataset: {args.dataset}")
    try:
        dataset = load_dataset(args.dataset, args.dataset_config, split=args.dataset_split)
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        return
    
    logging.info(f"Generating {args.num_examples} examples for each input")
    augmented_data = process_dataset(model, tokenizer, dataset, args)
    
    output_path = Path(f"{args.output_file}.{args.output_format}")
    logging.info(f"Saving augmented dataset to: {output_path}")
    save_checkpoint(augmented_data, output_path, args.output_format)
    
    logging.info("Done!")

if __name__ == "__main__":
    main()