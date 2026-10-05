import json
import math
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

from kdfp.schemas.experiments.experimentConfig import ExperimentConfig
from kdfp.schemas.models import DistillationModelConfig
from kdfp.schemas.trainingParams import TrainingParams

from .data import load_data
from .general import cleanup_resources

# NOTE: Data streaming is hardcoded to True


def prepare_lm_and_dataloader(
    lm_name: str,
    dataset_name: str,
    dataset_subset: Optional[str],
    split: str,
    seq_len: int,
    batch_size: int,
    torch_dtype: torch.dtype,
    token_rename: bool = False,
    shuffle: bool = False,
    lm_dir: Optional[str] = None,
    total_tokens: Optional[int] = None,
    num_workers: int = 0,  # needs to be 0 for caching to work for some reason
):
    """Load model, sae, and dataset based on args."""
    if lm_dir:
        print(f"Loading model {lm_name} from local version in {lm_dir}")

    lm = HookedTransformer.from_pretrained_no_processing(lm_name, dtype=torch_dtype)
    tokenizer = lm.tokenizer

    if lm_dir:
        lm.load_state_dict(torch.load(f"{lm_dir}/state_dict.pt", weights_only=True))

    # load selected number of data samples
    n_samples = math.floor(total_tokens / seq_len) if total_tokens else None

    dataset = load_data(
        dataset_name=dataset_name,
        subset=dataset_subset,
        split=split,
        tokenizer=tokenizer,
        seq_len=seq_len,
        token_rename=token_rename,
        n_samples=n_samples,
        streaming=True,
    )
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return lm, dl, n_samples


def create_caching_hook(batch_idx, save_dir):
    """Create a hook function that saves activations to disk."""

    def caching_hook(act, hook):
        # Create a filename based on batch index
        filename = os.path.join(save_dir, f"batch_{batch_idx:05d}.pt")

        # Move activation to CPU and detach from computation graph
        act_cpu = act.detach().cpu()

        # Save the activation tensor
        torch.save(act_cpu, filename)

        # Return the activation unchanged for the forward pass
        return act

    return caching_hook


def cache_activations(
    model,
    dataloader,
    cache_outputs: bool,
    hook_points: List[str],
    save_dir,
    device,
    n_samples: Optional[int],
    token_rename: bool = False,
):
    """Process dataset and cache activations at specified hook point."""

    print("N SAMPLES: ", n_samples)
    print("\nCaching activations at:")
    for hp in hook_points:
        print(hp)
    # Create a mapping file to track what's in each batch
    batch_map = []
    total_samples_cached = 0

    if cache_outputs:
        output_logits_dir = f"{save_dir}/output_logits"
        os.makedirs(output_logits_dir, exist_ok=True)

    model.to(device)
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                if token_rename:
                    input_ids = batch["tokens"].to(device)
                else:
                    input_ids = batch["input_ids"].to(device)

                # Check if we need to truncate this batch to meet n_samples limit
                current_batch_size = input_ids.shape[0]
                if n_samples is not None:
                    remaining_samples = n_samples - total_samples_cached
                    if remaining_samples <= 0:
                        break  # We've already cached enough samples
                    if current_batch_size > remaining_samples:
                        # Truncate the batch to only process the remaining samples needed
                        input_ids = input_ids[:remaining_samples]
                        current_batch_size = remaining_samples

                # create full list of hooks and set their cache dirs
                fwd_hooks = [
                    (hp, create_caching_hook(batch_idx, f"{save_dir}/{hp}"))
                    for hp in hook_points
                ]

                # Register the hook for caching
                output_logits = model.run_with_hooks(
                    input_ids,
                    fwd_hooks=fwd_hooks,
                )

                if cache_outputs:
                    logits_filename = os.path.join(
                        output_logits_dir, f"batch_{batch_idx:05d}.pt"
                    )
                    torch.save(output_logits.detach().cpu(), logits_filename)

                # Save mapping information (what samples are in this batch)
                start_idx = total_samples_cached
                end_idx = start_idx + current_batch_size
                batch_map.append((batch_idx, start_idx, end_idx))
                total_samples_cached += current_batch_size

                # Break if we've reached the desired number of samples
                if n_samples is not None and total_samples_cached >= n_samples:
                    break

        # Save the batch mapping for future reference
        with open(os.path.join(save_dir, "batch_mapping.txt"), "w") as f:
            f.write("batch_idx,start_sample_idx,end_sample_idx\n")
            for batch_idx, start_idx, end_idx in batch_map:
                f.write(f"{batch_idx},{start_idx},{end_idx}\n")

        print(
            f"Finished caching {len(batch_map)} batches with {total_samples_cached} total samples."
        )
        if n_samples is not None:
            print(
                f"Target samples: {n_samples}, Actual samples cached: {total_samples_cached}"
            )

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("Your process hit the memory limit - exiting gracefully")
            cleanup_resources()
            sys.exit(1)
        else:
            print(e)
            raise
    except Exception as e:
        print(f"Error in run_with_hooks: {e}")
        raise


def is_cache_complete(
    cache_dir: str,
    # expected_batches: int,
    n_samples: Optional[int] = None,
) -> bool:
    """Check if cache directory contains all expected batches and samples."""
    if not os.path.exists(cache_dir):
        print(f"Directory {cache_dir} not found.")
        return False

    # Check mapping file
    mapping_file = os.path.join(Path(cache_dir).parent, "batch_mapping.txt")
    if not os.path.exists(mapping_file):
        # print(f"Mapping file not found in {Path(cache_dir).parent}.")
        return False

    # If n_samples is specified, check if we have at least that many samples
    if n_samples is not None:
        try:
            with open(mapping_file, "r") as f:
                lines = f.readlines()[1:]  # Skip header
                total_samples = 0
                for line in lines:
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        # start_idx = int(parts[1])
                        end_idx = int(parts[2])
                        total_samples = max(total_samples, end_idx)

                if total_samples < n_samples:
                    # print(f"Error: Cache contains only {total_samples} samples, but {n_samples} samples are required.")
                    return False
                # else:
                #     print(f"Cache contains {total_samples} samples (required: {n_samples}).")

        except Exception as e:
            print(f"Error reading mapping file: {e}")
            return False

    return True


# NOTE: seq_len overridden to 256 for caching to save disk space
def activation_caching(
    lm_cfg: DistillationModelConfig,
    cache_dir: str,
    train_params: TrainingParams,
    cache_outputs: bool = False,
    lm_dir: Optional[str] = None,
) -> Tuple[str, int]:
    lm_act_cache_dir = f"{cache_dir}/{lm_cfg.name.split('/')[1]}/{train_params.offline_projection.dataset.name.split('/')[1]}"
    os.makedirs(lm_act_cache_dir, exist_ok=True)

    lm, token_dl, n_samples = prepare_lm_and_dataloader(
        lm_name=lm_cfg.name,
        dataset_name=train_params.offline_projection.dataset.name,
        dataset_subset=train_params.offline_projection.dataset.subset,
        split=train_params.offline_projection.dataset.split,
        seq_len=256,  # train_params.seq_len,
        batch_size=train_params.offline_projection.cache_batch_size,
        torch_dtype=train_params.torch_dtype,
        token_rename=train_params.offline_projection.dataset.token_rename,
        shuffle=train_params.offline_projection.dataset.shuffle,
        lm_dir=lm_dir,
        total_tokens=train_params.offline_projection.total_tokens,
        # num_workers=train_params.resources.projector.num_data_workers,
    )

    # token_dl_len = len(token_dl)

    lm_hook_points = lm_cfg.hook_list
    hook_caches = [f"{lm_act_cache_dir}/{hp}" for hp in lm_hook_points]

    if train_params.offline_projection.force_cache_all:
        hooks_needed = lm_hook_points
    else:
        hooks_needed = []
        for idx, cache in enumerate(hook_caches):
            os.makedirs(cache, exist_ok=True)

            if not is_cache_complete(cache, n_samples=n_samples):
                hooks_needed.append(lm_hook_points[idx])

    if hooks_needed:
        print("\nStarting model activation caching...")
        print("Requested hook points:")
        for hp in lm_hook_points:
            print(hp)
        if hooks_needed != lm_hook_points:
            print("\nSome hooks have already been cached!")
        else:
            print("\nNo existing cache found. Caching all requested hook points.")
        # print("Caching activations for:")
        # print(f"{hp}\n" for hp in hooks_needed)
        cache_activations(
            model=lm,
            dataloader=token_dl,
            cache_outputs=cache_outputs,
            hook_points=hooks_needed,
            save_dir=lm_act_cache_dir,
            device=train_params.device,
            n_samples=n_samples,
            token_rename=train_params.offline_projection.dataset.token_rename,
        )
        print(
            f"\nModel activation caching complete. Batch file for all hook points saved at {lm_act_cache_dir}/batch_mapping.txt"
        )
    else:
        print(f"\nUsing existing lm activations cache at {lm_act_cache_dir}.")

    if n_samples is not None:
        cached_tokens = n_samples * train_params.seq_len
    else:
        # cached_tokens = train_params.seq_len * train_params.offline_projection.cache_batch_size * token_dl_len
        raise NotImplementedError(
            "Total cached tokens calculation not implemented for streaming datasets without n_samples."
        )

    summary = {
        "lm_hook_points": lm_hook_points,
        "lm_name": lm_cfg.name,
        "dataset_name": train_params.offline_projection.dataset.name,
        "dataset_subset": train_params.offline_projection.dataset.subset,
        "split": train_params.offline_projection.dataset.split,
        "seq_len": train_params.seq_len,
        "batch_size": train_params.offline_projection.cache_batch_size,
        "lm_act_cache_dir": lm_act_cache_dir,
        "token_rename": train_params.offline_projection.dataset.token_rename,
        "shuffle": train_params.offline_projection.dataset.shuffle,
        "lm_dir": lm_dir,
        "total_requested_tokens": train_params.offline_projection.total_tokens,
        "total_cached_tokens": cached_tokens,
        "num_workers": train_params.resources.projector.num_data_workers,
        "force_cache_all": train_params.offline_projection.force_cache_all,
        "cache_outputs": cache_outputs,
        "device": train_params.device,
    }

    # print("====== SUMMARY ======")
    # table = print_args_table(args)

    # Save summary to appropriate directory
    # output_dir = args.sae_act_cache_dir if args.sae_act_cache_dir else args.lm_act_cache_dir
    with open(f"{lm_act_cache_dir}/caching_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    return lm_act_cache_dir, lm_cfg.d_model


def cache_student_and_teacher_acts(
    cfg: ExperimentConfig,
    training_params: TrainingParams,
    cache_dir: str,
    cache_outputs: bool = False,
    local_teacher_dir: Optional[str] = None,
    local_student_dir: Optional[str] = None,
) -> Tuple[str, int, str, int]:
    """Cache both student and teacher activations based on experiment config."""
    # Cache teacher activations
    teacher_cache_dir, teacher_dim = activation_caching(
        lm_cfg=cfg.distill_pair.teacher,
        cache_dir=cache_dir,
        train_params=training_params,
        cache_outputs=cache_outputs,
        lm_dir=local_teacher_dir,
    )

    # Cache student activations
    student_cache_dir, student_dim = activation_caching(
        lm_cfg=cfg.distill_pair.student,
        cache_dir=cache_dir,
        train_params=training_params,
        cache_outputs=cache_outputs,
        lm_dir=local_student_dir,
    )

    return (
        teacher_cache_dir,
        teacher_dim,
        student_cache_dir,
        student_dim,
    )
