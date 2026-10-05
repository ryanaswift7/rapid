import argparse
import sys

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Configuration constants
TOKENIZER_ID = "EleutherAI/pythia-70m"
MASTER_SEED = 3
BUFFER_SIZE = 10_000
VERSIONS = ["v1"]

# Dataset mapping
DATASET_CONFIGS = {
    "fineweb-edu": {
        "hf_path": "HuggingFaceFW/fineweb-edu",
        "subset": "sample-10BT",
        "base_repo": "ryanaswift7/fineweb-edu-pythia-kdfp-subsets"
    },
    "wikipedia": {
        "hf_path": "wikimedia/wikipedia",
        "subset": "20231101.en",
        "base_repo": "ryanaswift7/wikipedia-pythia-kdfp-subsets"
    },
    "dolma3": {
        "hf_path": "allenai/dolma3_mix-150B-1025",
        "subset": "default",
        "revision": "refs/convert/parquet", # Points directly to the auto-converted Parquet branch
        "base_repo": "ryanaswift7/dolma3-pythia-kdfp-subsets"
    }
}

def parse_size(size_str):
    size_str = size_str.lower().strip()
    multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    if size_str[-1] in multipliers:
        try:
            return int(float(size_str[:-1]) * multipliers[size_str[-1]])
        except ValueError: pass
    try: return int(size_str)
    except ValueError: raise ValueError(f"Invalid size format: {size_str}")

def get_next_chunk(iterator, tokenizer, target_tokens, version_label, split_name, max_tokens):
    buffer = []
    current_count = 0
    desc = f"Collecting {version_label} [{split_name}]"
    pbar = tqdm(total=target_tokens, desc=desc, unit="tok")

    while current_count < target_tokens:
        try:
            sample = next(iterator)
            text = sample.get("text", "")
            if not text: continue

            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            
            if len(tokens) > max_tokens:
                tokens = tokens[:max_tokens]
                text = tokenizer.decode(tokens)
            
            num_tokens = len(tokens)

            buffer.append({
                "text": text,
                "token_count": num_tokens,
                "id": str(sample.get("id") or sample.get("doc_id") or ""),
                "version_label": version_label,
                "split_name": split_name,
            })

            current_count += num_tokens
            pbar.update(num_tokens)

        except StopIteration:
            print(f"\nCRITICAL: Dataset ran out of data during {version_label}!")
            break

    pbar.close()
    return buffer

def main():
    parser = argparse.ArgumentParser(description="Upload dataset subsets based on token count.")
    parser.add_argument("dataset", choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument("train_size", type=str)
    parser.add_argument("--seq-len", type=int, default=2048)

    args = parser.parse_args()
    config = DATASET_CONFIGS[args.dataset]

    try:
        train_target = parse_size(args.train_size)
    except ValueError as e:
        print(f"Error: {e}"); sys.exit(1)

    val_target, test_target = int(train_target * 0.10), int(train_target * 0.10)
    final_repo_id = f"{config['base_repo']}-{args.train_size.lower().strip()}"
    if args.seq_len != 2048: final_repo_id += f"-seq{args.seq_len}"

    print(f"--- Processing {args.dataset} -> {final_repo_id} ---")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    
    # Extract the revision (defaults to "main" if not specified in config)
    target_revision = config.get("revision", "main")

    # Standard load_dataset works perfectly now because Parquet handles the schema
    ds_stream = load_dataset(
        config['hf_path'], 
        name=config['subset'], 
        split="train", 
        streaming=True,
        revision=target_revision
    )

    shuffled_stream = ds_stream.shuffle(seed=MASTER_SEED, buffer_size=BUFFER_SIZE)
    ds_iterator = iter(shuffled_stream)

    for version in VERSIONS:
        print(f"\n=== {version} ===")
        train_data = get_next_chunk(ds_iterator, tokenizer, train_target, version, "train", args.seq_len)
        val_data = get_next_chunk(ds_iterator, tokenizer, val_target, version, "validation", args.seq_len)
        test_data = get_next_chunk(ds_iterator, tokenizer, test_target, version, "test", args.seq_len)

        final_dataset_dict = DatasetDict({
            "train": Dataset.from_list(train_data),
            "validation": Dataset.from_list(val_data),
            "test": Dataset.from_list(test_data)
        })

        final_dataset_dict.push_to_hub(final_repo_id, config_name=version)
        print(f"  ✅ {version} uploaded.")

if __name__ == "__main__":
    main()