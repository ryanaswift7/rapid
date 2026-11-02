import json
import os
import random
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import load_dataset, load_dataset_builder
from pydantic import BaseModel, field_validator, model_validator
from torch.utils.data import ConcatDataset, Dataset

from rapid.schemas.tasks import TaskInstance


DEFAULT_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def _normalize_chat_message(message: dict) -> dict:
    role = message.get("role") or message.get("from") or "user"
    if role in {"human", "user"}:
        role = "user"
    elif role in {"gpt", "assistant", "bot"}:
        role = "assistant"
    elif role != "system":
        role = "user"

    content = message.get("content")
    if content is None:
        content = message.get("value", "")

    return {"role": role, "content": content}


def _render_conversation_text(conversation, tokenizer):
    messages = [_normalize_chat_message(msg) for msg in conversation]
    chat_template = getattr(tokenizer, "chat_template", None) or DEFAULT_CHAT_TEMPLATE
    return tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template,
        tokenize=False,
        add_generation_prompt=False,
    )


def _preprocess(examples, seq_len, tokenizer, token_rename: bool):
    # Preprocess the examples
    if "conversations" in examples:
        texts = []
        for conversation in examples["conversations"]:
            texts.append(_render_conversation_text(conversation, tokenizer))
    elif "messages" in examples:
        texts = []
        for conversation in examples["messages"]:
            texts.append(_render_conversation_text(conversation, tokenizer))
    elif "text" in examples:
        texts = examples["text"]
    else:
        raise ValueError(
            "Expected either a 'text' or 'conversations' field in dataset batch."
        )

    tokenized_examples = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=seq_len,
    )

    # Rename 'input_ids' to 'tokens'
    if token_rename:
        tokenized_examples["tokens"] = tokenized_examples.pop("input_ids")

    return tokenized_examples


def load_data(
    tokenizer,
    seq_len: int,
    dataset_name: str,
    streaming: bool,
    subset: Optional[str] = None,
    split: str = "train",
    token_rename: bool = False,
    n_samples: Optional[int] = None,
):
    if subset:
        try:
            dataset = load_dataset(
                dataset_name,
                subset,
                split=split,
                trust_remote_code=True,
                streaming=streaming,
            )
        except Exception:
            dataset = None

    else:
        try:
            dataset = load_dataset(
                dataset_name,
                split=split,
                trust_remote_code=True,
                streaming=streaming,
            )
        except Exception:
            dataset = None

    if dataset is None:
        builder = load_dataset_builder(
            dataset_name,
            subset,
            trust_remote_code=True,
        )
        if not builder.info.splits or "train" not in builder.info.splits:
            raise ValueError(
                f"Dataset {dataset_name} does not provide split '{split}' and has no train split to fall back to."
            )

        train_count = builder.info.splits["train"].num_examples
        val_count = max(1, train_count // 10)
        test_count = max(1, train_count // 10)
        train_count = max(1, train_count - val_count - test_count)

        train_dataset = load_dataset(
            dataset_name,
            subset,
            split="train",
            trust_remote_code=True,
            streaming=streaming,
        )

        if split == "train":
            dataset = train_dataset.take(train_count)
        elif split == "validation":
            dataset = train_dataset.skip(train_count).take(val_count)
        elif split == "test":
            dataset = train_dataset.skip(train_count + val_count).take(test_count)
        else:
            raise ValueError(
                f"Unsupported split '{split}' for dataset {dataset_name}."
            )

    if n_samples:
        dataset = dataset.take(n_samples)

    use_chat_template = bool(
        {"conversations", "messages"} & set(getattr(dataset, "column_names", []))
    )

    if use_chat_template:
        print(f"Using tokenizer chat template for dataset {dataset_name}")

    preprocess_fn = partial(
        _preprocess, seq_len=seq_len, tokenizer=tokenizer, token_rename=token_rename
    )
    dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )

    if token_rename:
        dataset.set_format(type="torch", columns=["tokens"])
    # else:
    #     dataset.set_format(type="torch")
    return dataset


def _format_arc(examples, split: str):
    """
    Formats ARC examples for either training or NLL-based evaluation.

    Args:
        examples: A batch of examples from the ARC dataset.
        split (str): The dataset split, e.g., 'train' or 'validation'.
    """
    # Get the number of examples in the batch
    num_examples = len(examples["id"])

    if split == "train":
        # For TRAINING: Create a single, correct question-answer pair.
        texts = []
        for i in range(num_examples):
            question = examples["question"][i]
            choices = examples["choices"][i]["text"]
            choice_labels = examples["choices"][i][
                "label"
            ]  # e.g., ['A', 'B', 'C', 'D']
            answer_key = examples["answerKey"][i]  # e.g., 'C'

            # Find the 0-indexed integer for the correct answer
            correct_idx = choice_labels.index(answer_key)
            correct_answer_text = choices[correct_idx]

            # Format as a complete and correct QA pair
            text = f"Question: {question}\nAnswer: {correct_answer_text}"
            texts.append(text)

        return texts, None  # No separate labels needed for training

    else:
        # For VALIDATION: Create all possible QA pairs for each example
        # and return them in a flattened list for efficient batch tokenization.
        all_choice_texts = []
        labels = []
        for i in range(num_examples):
            question = examples["question"][i]
            choices = examples["choices"][i]["text"]
            choice_labels = examples["choices"][i]["label"]
            answer_key = examples["answerKey"][i]

            # Find the 0-indexed label
            label_idx = choice_labels.index(answer_key)
            labels.append(label_idx)

            # Create a formatted string for each of the choices
            for choice in choices:
                text = f"Question: {question}\nAnswer: {choice}"
                all_choice_texts.append(text)

        return all_choice_texts, labels


def _format_hellaswag(examples, split: str):
    """
    Prepares texts and labels for HellaSwag.

    For split='train': Returns a list of single correct sentences.
    For split='validation'/'test': Returns all 4 possible sentences for each example,
                                   flattened into a single list, plus the labels.
    """
    # Get the number of examples in the batch
    num_examples = len(examples["ctx"])

    if split == "train":
        texts = []
        for i in range(num_examples):
            context = examples["ctx"][i]
            correct_ending = examples["endings"][i][int(examples["label"][i])]
            texts.append(context + " " + correct_ending)
        # No labels needed for training
        return texts, None
    else:
        # For validation, we create all 4 options for each example
        all_choice_texts = []
        labels = []
        for i in range(num_examples):
            context = examples["ctx"][i]
            for choice in examples["endings"][i]:
                all_choice_texts.append(context + " " + choice)
            labels.append(int(examples["label"][i]))
        return all_choice_texts, labels


def _format_piqa(examples, split: str):
    """
    Formats PIQA examples for either training or NLL-based evaluation.

    Args:
        examples: A batch of examples from the PIQA dataset.
        split (str): The dataset split, e.g., 'train' or 'validation'.
    """
    # Get the number of examples in the batch
    num_examples = len(examples["goal"])

    if split == "train":
        # For TRAINING: Create a single, correct goal-solution pair.
        texts = []
        for i in range(num_examples):
            goal = examples["goal"][i]
            sol1 = examples["sol1"][i]
            sol2 = examples["sol2"][i]
            # The 'label' is already a 0-indexed integer (0 or 1)
            correct_label = examples["label"][i]
            correct_solution = sol1 if correct_label == 0 else sol2

            # Format as a complete and correct goal-solution pair
            text = f"Goal: {goal}\nSolution: {correct_solution}"
            texts.append(text)

        return texts, None  # No separate labels needed for training

    else:
        # For VALIDATION: Create both possible goal-solution pairs for each example
        # and return them in a flattened list for efficient batch tokenization.
        all_choice_texts = []
        labels = []
        for i in range(num_examples):
            goal = examples["goal"][i]
            sol1 = examples["sol1"][i]
            sol2 = examples["sol2"][i]
            label_idx = examples["label"][i]

            labels.append(label_idx)

            # Create a formatted string for both possible solutions
            text1 = f"Goal: {goal}\nSolution: {sol1}"
            text2 = f"Goal: {goal}\nSolution: {sol2}"
            all_choice_texts.extend([text1, text2])

        return all_choice_texts, labels


def _format_boolq(examples, split: str):
    """
    Formats BoolQ examples for either training or NLL-based evaluation.

    Args:
        examples: A batch of examples from the BoolQ dataset.
        split (str): The dataset split, e.g., 'train' or 'validation'.
    """
    # Get the number of examples in the batch
    num_examples = len(examples["question"])

    # Define the fixed choices and their mapping from the boolean label
    # This ensures consistency: False -> 0 -> "no", True -> 1 -> "yes"
    choices = ["no", "yes"]

    if split == "train":
        # For TRAINING: Create a single, correct question-answer pair.
        texts = []
        for i in range(num_examples):
            context = examples["passage"][i]
            question = examples["question"][i]
            # Determine the correct answer string based on the boolean label
            correct_answer_text = "yes" if examples["label"][i] else "no"

            # Format as a complete and correct QA pair
            text = f"Passage: {context}\nQuestion: {question}?\nAnswer: {correct_answer_text}"
            texts.append(text)

        return texts, None  # No separate labels needed for training

    else:
        # For VALIDATION: Create both possible QA pairs for each example
        # and return them in a flattened list for efficient batch tokenization.
        all_choice_texts = []
        labels = []
        for i in range(num_examples):
            context = examples["passage"][i]
            question = examples["question"][i]

            # Determine the 0-indexed label (0 for "no", 1 for "yes")
            label_idx = 1 if examples["label"][i] else 0
            labels.append(label_idx)

            # Create a formatted string for both of the fixed choices ("no", "yes")
            for choice in choices:
                text = f"Passage: {context}\nQuestion: {question}?\nAnswer: {choice}"
                all_choice_texts.append(text)

        return all_choice_texts, labels


def _format_logiqa(examples, split: str):
    """
    Formats LogiQA examples for either training or NLL-based evaluation.

    Args:
        examples: A batch of examples from the LogiQA dataset.
        split (str): The dataset split, e.g., 'train' or 'validation'.
    """
    # Get the number of examples in the batch
    num_examples = len(examples["id"])

    if split == "train":
        # For TRAINING: Create a single, correct question-answer pair.
        texts = []
        for i in range(num_examples):
            context = examples["text"][i]
            question = examples["question"][i]
            choices = examples["options"][i]
            # The 'answer' is already a 0-indexed integer
            correct_answer_idx = examples["answer"][i]
            correct_answer_text = choices[correct_answer_idx]

            # Format as a complete and correct QA pair
            text = f"Passage: {context}\nQuestion: {question}\nAnswer: {correct_answer_text}"
            texts.append(text)

        return texts, None  # No separate labels needed for training

    else:
        # For VALIDATION: Create four possible QA pairs for each example
        # and return them in a flattened list for efficient batch tokenization.
        all_choice_texts = []
        labels = []
        for i in range(num_examples):
            context = examples["text"][i]
            question = examples["question"][i]
            choices = examples["options"][i]
            label_idx = examples["answer"][i]

            labels.append(label_idx)

            # Create a formatted string for each of the four choices
            for choice in choices:
                text = f"Passage: {context}\nQuestion: {question}\nAnswer: {choice}"
                all_choice_texts.append(text)

        return all_choice_texts, labels


def _format_sciq(examples, split: str):
    """
    Formats SciQ examples for either training or NLL-based evaluation.

    Args:
        examples: A batch of examples from the SciQ dataset.
        split (str): The dataset split, e.g., 'train' or 'validation'.
    """
    # Get the number of examples in the batch
    num_examples = len(examples["question"])

    if split == "train":
        # For TRAINING: Create a single, correct question-answer pair.
        texts = []
        for i in range(num_examples):
            context = examples["support"][i]
            question = examples["question"][i]
            correct_answer = examples["correct_answer"][i]

            # Format as a complete and correct QA pair
            text = f"Context: {context}\nQuestion: {question}\nAnswer: {correct_answer}"
            texts.append(text)

        return texts, None  # No separate labels needed for training

    else:
        # For VALIDATION: Create four possible QA pairs for each example
        # and return them in a flattened list for efficient batch tokenization.
        all_choice_texts = []
        labels = []
        for i in range(num_examples):
            context = examples["support"][i]
            question = examples["question"][i]
            correct_answer = examples["correct_answer"][i]

            # Assemble all four choices
            choices = [
                correct_answer,
                examples["distractor1"][i],
                examples["distractor2"][i],
                examples["distractor3"][i],
            ]

            # Shuffle the choices to avoid positional bias
            random.shuffle(choices)

            # Find the new 0-indexed label of the correct answer
            label_idx = choices.index(correct_answer)
            labels.append(label_idx)

            # Create a formatted string for each of the four shuffled choices
            for choice in choices:
                text = f"Context: {context}\nQuestion: {question}\nAnswer: {choice}"
                all_choice_texts.append(text)

        return all_choice_texts, labels


def _format_winogrande(examples, split: str):
    """
    Formats Winogrande examples for either training or NLL-based evaluation.

    Args:
        examples: A batch of examples from the Winogrande dataset.
        split (str): The dataset split, e.g., 'train' or 'validation'.
    """
    # Get the number of examples in the batch
    num_examples = len(examples["sentence"])

    if split == "train":
        # For TRAINING: Create a single, correctly filled-in sentence for each example.
        texts = []
        for i in range(num_examples):
            sentence = examples["sentence"][i]
            option1 = examples["option1"][i]
            option2 = examples["option2"][i]
            # The 'answer' in the dataset is 1-based, so we determine the correct text
            correct_option = option1 if examples["answer"][i] == "1" else option2

            # Replace the pronoun placeholder '_' with the correct option
            filled_sentence = sentence.replace("_", correct_option, 1)
            texts.append(filled_sentence)

        return texts, None  # No separate labels needed for training

    else:
        # For VALIDATION: Create all possible sentences (2 for each example)
        # and return them in a flattened list for efficient batch tokenization.
        all_choice_texts = []
        labels = []
        for i in range(num_examples):
            sentence = examples["sentence"][i]
            option1 = examples["option1"][i]
            option2 = examples["option2"][i]

            # Create both possible sentences
            text_option1 = sentence.replace("_", option1, 1)
            text_option2 = sentence.replace("_", option2, 1)
            all_choice_texts.extend([text_option1, text_option2])

            # The 'answer' is 1 or 2; convert it to a 0-indexed label
            label_idx = int(examples["answer"][i]) - 1
            labels.append(label_idx)

        return all_choice_texts, labels


def _preprocess_task_ft(
    examples,
    split: str,
    task: TaskInstance,
    seq_len: int,
    tokenizer,
):
    if "hellaswag" in task.name:
        texts, labels = _format_hellaswag(examples, split=split)
    elif "winogrande" in task.name:
        texts, labels = _format_winogrande(examples, split=split)
    elif "sciq" in task.name:
        texts, labels = _format_sciq(examples, split=split)
    elif "logiqa" in task.name:
        texts, labels = _format_logiqa(examples, split=split)
    elif "boolq" in task.name:
        texts, labels = _format_boolq(examples, split=split)
    elif "piqa" in task.name:
        texts, labels = _format_piqa(examples, split=split)
    elif "arc" in task.name:
        texts, labels = _format_arc(examples, split=split)
    else:
        raise ValueError(f"Task {task.name} is not currently supported.")

    # Tokenize all text inputs at once
    tokenized = tokenizer(
        texts,
        max_length=seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    # if split == 'train':
    #     # For training, labels are the input_ids shifted
    #     tokenized['labels'] = tokenized['input_ids'].clone()
    #     return tokenized
    # else:

    if split != "train":
        assert labels is not None, "Labels should not be None for val or test."
        # For validation, reshape the tokenized outputs
        num_examples = len(labels)
        num_choices = 4 if task.task_type == "multiclass" else 2

        # Reshape from [batch_size * 4, seq_len] to [batch_size, 4, seq_len]
        for key in tokenized:
            tokenized[key] = tokenized[key].view(num_examples, num_choices, -1)

        # Add the labels to the dictionary
        tokenized["label"] = torch.tensor(labels, dtype=torch.long)
    return tokenized


def load_task_data(
    task: "TaskInstance",
    tokenizer,
    seq_len: int,
    split: str,
    streaming: bool,
    n_samples: Optional[int] = None,
):
    """
    Loads and preprocesses a dataset for a given task.

    If task.test_available is False, it reserves 10% of the 'train' split
    for validation/testing purposes. `actual_split` is the split used for
    loading the data from HuggingFace, and `split` is used to decide
    how to preprocess and format the data.
    """
    actual_split = split

    # If a dedicated test set is not provided by the dataset, create a
    # holdout set from the training data.
    if not task.test_available:
        if split == "train":
            # When training, use the first 90% of the original training data.
            actual_split = "train"
        elif split == "validation":
            # When 'test' or 'validation' is requested, use the last 10% of
            # the training data as a substitute validation set.
            actual_split = "train"
        elif split == "test":
            actual_split = "validation"

    if task.subset:
        dataset = load_dataset(
            task.dataset_name,
            task.subset,
            split=actual_split,
            trust_remote_code=True,
            streaming=streaming,
        )
    else:
        dataset = load_dataset(
            task.dataset_name,
            split=actual_split,
            trust_remote_code=True,
            streaming=streaming,
        )
    if not task.test_available and task.ninety_percent is not None:
        if split == "train":
            dataset = dataset.take(task.ninety_percent)
        elif split == "validation":
            dataset = dataset.skip(task.ninety_percent)
    else:
        raise ValueError("If test_available is False, ninety_percent must be set.")

    if n_samples:
        dataset = dataset.take(n_samples)

    preprocess_fn = partial(
        _preprocess_task_ft,
        split=split,  # the originally requested split name
        task=task,
        seq_len=seq_len,
        tokenizer=tokenizer,
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return dataset


class CachedActivationsDataset(Dataset):
    """
    Optimized Dataset for working with cached activations from TransformerLens.

    Performance Note:
    To avoid the "initialization explosion" (re-reading CSVs for every layer for every worker),
    this class expects metadata (mappings) to be pre-calculated and passed in.
    """

    def __init__(
        self,
        cache_dir: str,
        sample_to_batch: Dict[int, int],
        sample_to_position: Dict[int, int],
        num_samples: int,
        activation_shape: Optional[Tuple] = None,
        target_data_path: Optional[str] = None,
        preload: bool = False,
        transform=None,
    ):
        """
        Args:
            cache_dir: Directory containing the cached activation files.
            sample_to_batch: Pre-computed dict mapping sample_index -> batch_index.
            sample_to_position: Pre-computed dict mapping sample_index -> index_inside_batch.
            num_samples: Total number of samples.
            activation_shape: (Optional) Shape of the activations (excluding batch dim).
            target_data_path: Path to targets/labels (CSV or pickle).
            preload: Whether to load all .pt files into RAM on init.
            transform: Optional transform to apply to the activations.
        """
        self.cache_dir = cache_dir
        self.transform = transform

        # Shared Metadata (Passed by reference, minimal memory overhead per instance)
        self.sample_to_batch = sample_to_batch
        self.sample_to_position = sample_to_position
        self.num_samples = num_samples
        self.activation_shape = activation_shape

        # Lazy load targets
        self.targets = None
        if target_data_path:
            # We load this lazily or only once if passed from outside
            if target_data_path.endswith(".csv"):
                self.targets = pd.read_csv(target_data_path)
            elif target_data_path.endswith(".pkl"):
                self.targets = pd.read_pickle(target_data_path)
            else:
                raise ValueError("Target data must be CSV or pickle format")

        # Preload logic
        self.preloaded_activations = {}
        if preload:
            print(f"Preloading activations from {os.path.basename(cache_dir)}...")
            # Use the set from the dictionary values to avoid scanning directory
            unique_batches = set(self.sample_to_batch.values())
            for batch_idx in unique_batches:
                batch_path = os.path.join(cache_dir, f"batch_{batch_idx:05d}.pt")
                # Load to CPU memory
                self.preloaded_activations[batch_idx] = torch.load(
                    batch_path, map_location="cpu", weights_only=True
                )

    @staticmethod
    def load_metadata(cache_dir: str) -> Tuple[Dict, Dict, int]:
        """
        Static helper to load metadata ONCE.
        Call this in HookCache and pass the results to the dataset constructor.
        """
        mapping_path = os.path.join(cache_dir, "batch_mapping.txt")
        if not os.path.exists(mapping_path):
            # If mapping doesn't exist, assume standard 1:1 or generate it
            # For now, we raise strictly as per previous code logic
            raise FileNotFoundError(f"Batch mapping not found: {mapping_path}")

        # Fast CSV read
        df = pd.read_csv(mapping_path)

        sample_to_batch = {}
        sample_to_position = {}

        # Optimized Loop: Zip is ~10x faster than df.iterrows()
        # We assume rows define contiguous ranges
        for b_idx, start, end in zip(
            df["batch_idx"], df["start_sample_idx"], df["end_sample_idx"]
        ):
            # Create a range of indices for this batch
            indices = range(start, end)

            # Map global index -> batch index
            for i, sample_idx in enumerate(indices):
                sample_to_batch[sample_idx] = b_idx
                sample_to_position[sample_idx] = i

        if not sample_to_batch:
            return {}, {}, 0

        # Max index + 1 is the count (assuming 0-indexed contiguous)
        num_samples = max(sample_to_batch.keys()) + 1

        print(
            f"Loaded metadata from {mapping_path}: {num_samples} samples across "
            f"{len(set(sample_to_batch.values()))} batches."
        )
        return sample_to_batch, sample_to_position, num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 1. Resolve Location
        try:
            batch_idx = self.sample_to_batch[idx]
            position = self.sample_to_position[idx]
        except KeyError:
            raise IndexError(
                f"Index {idx} out of range for dataset of size {self.num_samples}"
            )

        # 2. Retrieve Data (RAM or Disk)
        if batch_idx in self.preloaded_activations:
            batch_activations = self.preloaded_activations[batch_idx]
        else:
            batch_path = os.path.join(self.cache_dir, f"batch_{batch_idx:05d}.pt")
            # map_location='cpu' prevents CUDA initialization in DataLoader workers
            batch_activations = torch.load(
                batch_path, map_location="cpu", weights_only=True
            )

        # 3. Extract Slice
        if position < batch_activations.shape[0]:
            activation = batch_activations[position]
        else:
            raise IndexError(
                f"Position {position} out of bounds for batch {batch_idx} "
                f"with shape {batch_activations.shape}"
            )

        # 4. Transform
        if self.transform:
            activation = self.transform(activation)

        # 5. Return with Target (if exists)
        if self.targets is not None:
            # .iloc is relatively slow; if targets are huge, convert to Tensor in __init__
            target = torch.tensor(self.targets.iloc[idx]["target"])
            return activation, target
        else:
            return activation


class HookCache(BaseModel):
    cache_dir: str
    hook_list: List[str]

    # Dataset fields
    embed: Optional[CachedActivationsDataset] = None
    attn_out: Optional[List[CachedActivationsDataset]] = None
    mlp_out: Optional[List[CachedActivationsDataset]] = None
    resid: Optional[List[CachedActivationsDataset]] = None
    output_logits: Optional[CachedActivationsDataset] = None

    model_config = {
        "extra": "forbid",  # Raises error on unrecognized fields
        "arbitrary_types_allowed": True,
    }

    @field_validator("cache_dir")
    def ensure_cache_exists(cls, v: str) -> str:
        """Validate that the cache directory exists."""
        if not os.path.isdir(v):
            raise ValueError(f"Directory {v} does not exist.")
        return v

    @field_validator("hook_list", mode="after")
    def ensure_hooks_in_cache(cls, v: List[str], values) -> List[str]:
        """Validate that all hook directories exist in the cache."""
        if "cache_dir" not in values.data:
            return v  # cache_dir validation failed, skip this check

        cache_dir = values.data["cache_dir"]
        for hp in v:
            if not os.path.isdir(os.path.join(cache_dir, hp)):
                raise ValueError(f"Cache for hook {hp} not found in {cache_dir}.")
        return v

    @model_validator(mode="after")
    def setup_datasets(self) -> "HookCache":
        """Initialize the dataset objects efficiently using shared metadata."""

        # ---------------------------------------------------------
        # 1. OPTIMIZATION: Load Metadata ONCE for all datasets
        # ---------------------------------------------------------
        # We load the batch_mapping.txt from the root cache_dir once.
        # s2b = sample_to_batch, s2p = sample_to_position
        try:
            s2b, s2p, n_samples = CachedActivationsDataset.load_metadata(self.cache_dir)
        except Exception as e:
            raise ValueError(f"Failed to load metadata from {self.cache_dir}: {str(e)}")

        # ---------------------------------------------------------
        # 2. Helper to create datasets with shared metadata
        # ---------------------------------------------------------
        def create_ds(hook_subfolder: str) -> CachedActivationsDataset:
            return CachedActivationsDataset(
                cache_dir=os.path.join(self.cache_dir, hook_subfolder),
                sample_to_batch=s2b,  # Pass by reference (Shared)
                sample_to_position=s2p,  # Pass by reference (Shared)
                num_samples=n_samples,  # Pass by value
                # Optional: You can pass target_data_path here if it exists in root
                # target_data_path=os.path.join(self.cache_dir, "targets.csv")
            )

        # ---------------------------------------------------------
        # 3. Categorize Hooks
        # ---------------------------------------------------------
        hook_dict = {
            "embed": [],
            "attn_out": [],
            "mlp_out": [],
            "resid": [],
            "output_logits": [],
        }

        for hook_name in hook_dict.keys():
            # Find all hooks that match this category
            matches = [s for s in self.hook_list if hook_name in s]
            hook_dict[hook_name] = matches

        # ---------------------------------------------------------
        # 4. Verify all hooks are accounted for
        # ---------------------------------------------------------
        all_matched_hooks = [hook for sublist in hook_dict.values() for hook in sublist]

        # Check for duplicates or missing hooks
        if len(set(all_matched_hooks)) != len(self.hook_list):
            unrecognized_hooks = set(self.hook_list) - set(all_matched_hooks)
            # This logic might need adjustment if a hook matches multiple keys,
            # but usually hook names are distinct.
            if unrecognized_hooks:
                raise ValueError(
                    f"Some hooks were not recognized: {unrecognized_hooks}"
                )

        # ---------------------------------------------------------
        # 5. Initialize Datasets
        # ---------------------------------------------------------
        try:
            # A. Layer-wise lists (attn, mlp, resid)
            layer_hook_type_list = ["attn_out", "mlp_out", "resid"]

            for hook_type in layer_hook_type_list:
                # Create a dataset for every hook in this category
                # We sort them to ensure layer 0 comes before layer 1, etc.
                # Assuming hook names contain "blocks.X." or similar for sorting if needed.
                # The current code relies on the order in self.hook_list or basic list append.

                # If you need specific sorting by layer index:
                # hooks_sorted = sorted(hook_dict[hook_type], key=lambda x: int(x.split('.')[1]))
                hooks_for_type = hook_dict[hook_type]

                ds_list = [create_ds(h) for h in hooks_for_type]
                setattr(self, hook_type, ds_list)

            # B. Singletons (embed, output_logits)
            if hook_dict["embed"]:
                # Assuming "hook_embed" is the specific folder name or the first match
                # Using the actual matched hook string from the list
                self.embed = create_ds(hook_dict["embed"][0])

            if hook_dict["output_logits"]:
                self.output_logits = create_ds(hook_dict["output_logits"][0])

            # ---------------------------------------------------------
            # 6. Consistency Checks
            # ---------------------------------------------------------
            # Ensure we have the same number of layers for all main components
            # Only check if the lists are not empty
            active_layer_lists = [
                getattr(self, ht)
                for ht in layer_hook_type_list
                if getattr(self, ht) and len(getattr(self, ht)) > 0
            ]

            if active_layer_lists:
                reference_len = len(active_layer_lists[0])
                for lst in active_layer_lists[1:]:
                    if len(lst) != reference_len:
                        raise ValueError(
                            f"Inconsistent layer counts detected. "
                            f"Expected {reference_len}, found {len(lst)}."
                        )

        except Exception as e:
            raise ValueError(f"Failed to initialize datasets: {str(e)}")

        return self


class ProjectorDataset(Dataset):
    def __init__(self, teacher_cache_dir: str, student_cache_dir: str):
        self.teacher_cache_dir = os.path.abspath(teacher_cache_dir)
        self.student_cache_dir = os.path.abspath(student_cache_dir)

        # -------------------------------------------------------------
        # 1. Load Metadata & Config
        # -------------------------------------------------------------
        self.teacher_config = self._load_json(self.teacher_cache_dir)
        self.student_config = self._load_json(self.student_cache_dir)

        # print("Loading Teacher Metadata...")
        self.teacher_meta = CachedActivationsDataset.load_metadata(
            self.teacher_cache_dir
        )

        # print("Loading Student Metadata...")
        self.student_meta = CachedActivationsDataset.load_metadata(
            self.student_cache_dir
        )

        # -------------------------------------------------------------
        # 2. VALIDATION: Ensure strict alignment
        # -------------------------------------------------------------
        # A. Resolve the specific hooks to be used
        self.teacher_hooks = self._get_hook_list(self.teacher_config)
        self.student_hooks = self._get_hook_list(self.student_config)

        # Check 1: Hooks must be identical (same hooks, same order)
        if len(self.teacher_hooks) != len(self.student_hooks):
            # diff_t = set(self.teacher_hooks) - set(self.student_hooks)
            # diff_s = set(self.student_hooks) - set(self.teacher_hooks)
            raise ValueError(
                f"Hook mismatch detected!\n"
                f"Teacher has {len(self.teacher_hooks)} hooks, Student has {len(self.student_hooks)} hooks.\n"
                # f"In Teacher but not Student: {diff_t}\n"
                # f"In Student but not Teacher: {diff_s}"
            )

        # Check 2: Number of raw samples must be identical
        # metadata tuple is (sample_to_batch, sample_to_position, num_samples)
        teacher_samples = self.teacher_meta[2]
        student_samples = self.student_meta[2]

        if teacher_samples != student_samples:
            raise ValueError(
                f"Sample count mismatch! Teacher has {teacher_samples} samples, "
                f"but Student has {student_samples} samples."
            )

        # -------------------------------------------------------------
        # 3. Calculate Total Length
        # -------------------------------------------------------------
        # Length = (Number of Samples) * (Number of Hooks)
        self.num_rows = teacher_samples * len(self.teacher_hooks)

        # -------------------------------------------------------------
        # 4. Lazy Initialization Placeholders
        # -------------------------------------------------------------
        self._teacher_ds = None
        self._student_ds = None

    def _load_json(self, cache_dir: str) -> Dict:
        path = Path(cache_dir) / "caching_summary.json"
        if not path.exists():
            raise FileNotFoundError(f"Config not found at {path}")
        with open(path, "r") as f:
            return json.load(f)

    def _get_hook_list(self, config: Dict) -> List[str]:
        """
        Parses the config to determine exactly which hooks will be turned into datasets.
        """
        hook_list = config.get("lm_hook_points", [])
        valid_hooks = []

        # 1. Embeddings
        embed_hooks = [h for h in hook_list if "embed" in h]
        if embed_hooks:
            valid_hooks.append(embed_hooks[0])

        # 2. Layer-wise Activations
        for hook_type in ["attn_out", "mlp_out", "resid"]:
            type_hooks = [h for h in hook_list if hook_type in h]
            try:
                # Sort by layer number: blocks.{layer}.{type}
                type_hooks.sort(key=lambda x: int(x.split(".")[1]))
            except (IndexError, ValueError):
                type_hooks.sort()

            valid_hooks.extend(type_hooks)

        return valid_hooks

    @property
    def teacher_ds(self):
        if self._teacher_ds is None:
            # print(f"Building Teacher Dataset from: {self.teacher_cache_dir}")
            self._teacher_ds = self._build_concat_dataset(
                self.teacher_cache_dir, self.teacher_hooks, self.teacher_meta
            )
        return self._teacher_ds

    @property
    def student_ds(self):
        if self._student_ds is None:
            # print(f"Building Student Dataset from: {self.student_cache_dir}")
            self._student_ds = self._build_concat_dataset(
                self.student_cache_dir, self.student_hooks, self.student_meta
            )
        return self._student_ds

    def _build_concat_dataset(
        self, cache_dir: str, hook_paths: List[str], metadata: Tuple
    ):
        """
        Instantiates a list of datasets for the provided hook paths.
        """
        s2b, s2p, n_samples = metadata

        datasets = []
        for hook_path in hook_paths:
            # --- PRINT HOOK NAME AS IT LOADS ---
            # print(f"Loading cached hook dataset: {hook_path}")

            ds = CachedActivationsDataset(
                cache_dir=os.path.join(cache_dir, hook_path),
                sample_to_batch=s2b,
                sample_to_position=s2p,
                num_samples=n_samples,
            )
            datasets.append(ds)

        return ConcatDataset(datasets)

    def lock(self) -> str:
        return f"/tmp/projector_dataset_{hash(self.teacher_cache_dir + self.student_cache_dir)}.lock"

    def __len__(self):
        return self.num_rows

    def __getitem__(self, index):
        X_teacher = self.teacher_ds[index]
        X_student = self.student_ds[index]
        return X_teacher, X_student
