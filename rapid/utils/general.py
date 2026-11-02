import datetime
import gc
import os
from pathlib import Path
import random
import re
import unicodedata

import numpy as np
import torch
from tabulate import tabulate
from transformers import set_seed


def set_all_seeds(seed: int, reproducible_backend: bool = True):
    random.seed(seed)  # Python random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch
    set_seed(seed)  # Hugging Face transformers
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU

    if reproducible_backend:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_directory(directory_path):
    """
    Create a directory if it doesn't already exist.

    :param directory_path: The path of the directory to create.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")


def set_vram_limit(gb_limit):
    # Get total and currently used VRAM
    total_vram = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).total_memory
    # Convert GB to bytes
    target_bytes = int(gb_limit * 1024**3)

    # Calculate what fraction of total VRAM this represents
    target_fraction = target_bytes / total_vram

    # Set the limit (using fraction, not byte value)
    torch.cuda.set_per_process_memory_fraction(target_fraction)

    print(f"Set VRAM limit to {target_bytes / 1024**3:.2f} GB")


def cleanup_resources(verbose=True):
    """Cleans up GPU resources and shows memory stats in GB"""
    if not torch.cuda.is_available():
        gc.collect()
        return
    # Conversion factor
    GB = 1024**3

    # Get memory stats before cleanup
    allocated_before = torch.cuda.memory_allocated() / GB
    reserved_before = torch.cuda.memory_reserved() / GB
    max_allocated = torch.cuda.max_memory_allocated() / GB
    total_memory = torch.cuda.get_device_properties(0).total_memory / GB

    # Perform cleanup
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                del obj
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()

    # Get memory stats after cleanup
    allocated_after = torch.cuda.memory_allocated() / GB
    reserved_after = torch.cuda.memory_reserved() / GB

    if verbose:
        print("\n=== GPU Memory Report ===")
        print(f"{'Metric':<25} {'Before Cleanup':>12} {'After Cleanup':>12}")
        print(
            f"{'Allocated Memory (GB)':<25} {allocated_before:>12.3f} {allocated_after:>12.3f}"
        )
        print(
            f"{'Reserved Memory (GB)':<25} {reserved_before:>12.3f} {reserved_after:>12.3f}"
        )
        print("\nAdditional Stats:")
        print(f"{'Peak Memory Usage (GB):':<25} {max_allocated:>12.3f}")
        print(f"{'Total GPU Memory (GB):':<25} {total_memory:>12.3f}")
        print(
            f"{'Memory Recovered (GB):':<25} {(allocated_before - allocated_after):>12.3f}"
        )


def print_args_table(args):
    table = tabulate(vars(args).items(), headers=["Argument", "Value"], tablefmt="grid")
    print("Current arguments:")
    print(table)
    return table


def log_training_setup(
    component: str,
    mode: str,
    visible_gpus: int,
    num_workers: int,
    num_gpus_per_trial: float,
    num_cpus_per_trial: int,
):
    print(
        f"[{component}] mode={mode} visible_gpus={visible_gpus} "
        f"workers={num_workers} gpus_per_trial={num_gpus_per_trial} "
        f"cpus_per_trial={num_cpus_per_trial}"
    )


def sanitize_filename(filename, replacement="_"):
    """
    Sanitize a string to be used as a filename.

    Args:
        filename (str): The original string
        replacement (str): Character to replace invalid chars with (default '_')

    Returns:
        str: Sanitized filename
    """
    # Normalize unicode characters (convert accented chars to ASCII equivalents)
    filename = (
        unicodedata.normalize("NFKD", filename)
        .encode("ascii", "ignore")
        .decode("ascii")
    )

    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*\x00-\x1F]', replacement, filename)

    # Remove leading/trailing spaces and dots (invalid on Windows)
    filename = filename.strip(". ")

    # Replace sequences of replacement characters with a single one
    filename = re.sub(r"[" + re.escape(replacement) + r"]+", replacement, filename)

    return filename


def set_device(device: str):
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device != "cuda":
            gpu: int = int(device.split(":")[1])
            torch.cuda.set_device(gpu)
        print(f"Using CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    elif device == "cpu":
        print("Using CPU")
    else:
        raise ValueError(f"Unsupported device: {device}")


def get_timestamp() -> str:
    now = datetime.datetime.now()
    return now.strftime("%Y%m%d_%H%M%S")


def get_hook_block_idx(hook_name: str) -> int:
    return int(hook_name.split(".")[1])

def is_valid_model_dir(model_path: str | Path) -> bool:
    """Checks if the directory contains a valid Hugging Face model."""
    path = Path(model_path)
    return path.exists() and (path / "config.json").exists()
