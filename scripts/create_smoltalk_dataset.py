#!/usr/bin/env python3
"""Build a 40k, 70/30 MetaMathQA/Numina-CoT SmolTalk dataset."""

import argparse
from functools import partial

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer


DATASET_NAME = "HuggingFaceTB/smoltalk"
TOKENIZER_NAME = "google/gemma-3-270m-it"
MAX_TOKENS = 1024
SEED = 3
DEFAULT_REPO_ID = "ryanaswift7/smoltalk-reasoning-gemma3-40k"

SOURCE_CONFIGS = {
    "metamathqa": {"subset": "metamathqa-50k", "count": 28_000},
    "numina_cot": {"subset": "numina-cot-100k", "count": 12_000},
}


def add_token_counts(examples, tokenizer, max_tokens):
    token_counts = []
    for messages in examples["messages"]:
        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        token_counts.append(len(token_ids))
    return {"token_count": token_counts}


def load_and_filter_source(source_name, config, tokenizer, max_tokens, seed):
    dataset = load_dataset(
        DATASET_NAME,
        config["subset"],
        split="train",
    )
    dataset = dataset.map(
        partial(
            add_token_counts,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
        ),
        batched=True,
        desc=f"Counting tokens: {source_name}",
    )
    dataset = dataset.filter(
        lambda token_count: token_count <= max_tokens,
        input_columns=["token_count"],
        desc=f"Filtering tokens: {source_name}",
    )

    required = config["count"]
    if len(dataset) < required:
        raise RuntimeError(
            f"{source_name} has only {len(dataset):,} samples after filtering, "
            f"but {required:,} are required."
        )

    dataset = dataset.shuffle(seed=seed).select(range(required))
    if "source" in dataset.column_names:
        dataset = dataset.remove_columns("source")
    return dataset.add_column("source", [source_name] * len(dataset))


def split_source(dataset, train_count, validation_count):
    train = dataset.select(range(train_count))
    validation_start = train_count
    validation = dataset.select(
        range(validation_start, validation_start + validation_count)
    )
    test = dataset.select(range(validation_start + validation_count, len(dataset)))
    return train, validation, test


def build_dataset(tokenizer, max_tokens, seed):
    sources = [
        load_and_filter_source(name, config, tokenizer, max_tokens, seed)
        for name, config in SOURCE_CONFIGS.items()
    ]

    # 90/5/5 splits while retaining the exact 70/30 source composition.
    split_counts = {
        "metamathqa": (25_200, 1_400),
        "numina_cot": (10_800, 600),
    }
    source_splits = [
        split_source(dataset, *split_counts[name])
        for name, dataset in zip(SOURCE_CONFIGS, sources)
    ]

    return DatasetDict(
        {
            split_name: concatenate_datasets(
                [source_split[split_index] for source_split in source_splits]
            ).shuffle(seed=seed)
            for split_index, split_name in enumerate(
                ("train", "validation", "test")
            )
        }
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create and push a filtered SmolTalk dataset."
    )
    parser.add_argument(
        "repo_id",
        nargs="?",
        default=DEFAULT_REPO_ID,
        help="Hugging Face Hub repository, for example username/smoltalk-40k-gemma3.",
    )
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    dataset = build_dataset(tokenizer, args.max_tokens, args.seed)

    print(
        f"Pushing {sum(len(split) for split in dataset.values()):,} samples to "
        f"{args.repo_id}"
    )
    print({split: len(rows) for split, rows in dataset.items()})
    print(
        {
            split: dataset[split].to_pandas()["source"].value_counts().to_dict()
            for split in dataset
        }
    )
    dataset.push_to_hub(args.repo_id, private=args.private)


if __name__ == "__main__":
    main()
