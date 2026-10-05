from pathlib import Path
from typing import Optional, Union

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, Subset, random_split

from kdfp.utils.data import ProjectorDataset

class ProjectorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        teacher_cache_dir: Union[str, Path],
        student_cache_dir: Union[str, Path],
        batch_size: int,
        seed: int,
        num_workers: int,
        seq_len: int,  # Added fixed sequence length parameter
        val_split: float = 0.1,
        test_split: float = 0.1,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
        total_tokens: Optional[int] = None,
    ):
        super().__init__()
        self.teacher_cache_dir = teacher_cache_dir
        self.student_cache_dir = student_cache_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Token limiting variables
        self.total_tokens = total_tokens
        self.seq_len = seq_len

        # Optimization variables
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        self.persistent_workers = True if num_workers > 0 else False

        self.dataset = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: Optional[str] = None):
        if self.dataset is not None:
            return

        self.dataset = ProjectorDataset(
            teacher_cache_dir=str(self.teacher_cache_dir),
            student_cache_dir=str(self.student_cache_dir),
        )

        total_size = len(self.dataset)
        print(f"======================= Total dataset size: {total_size} samples =======================")
        
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size - test_size

        if train_size + val_size + test_size != total_size:
            train_size = total_size - val_size - test_size

        generator = torch.Generator().manual_seed(self.seed)

        # 1. Split normally to keep validation/test sets consistent
        self.train_ds, self.val_ds, self.test_ds = random_split(
            self.dataset, [train_size, val_size, test_size], generator=generator
        )

        # 2. Apply Token Limit to the Training Set
        if self.total_tokens is not None:
            if self.seq_len is not None:
                # --- FAST PATH: Math-based slicing (O(1)) ---
                max_samples_allowed = self.total_tokens // self.seq_len
                samples_to_keep = min(max_samples_allowed, len(self.train_ds))
                keep_indices = list(range(samples_to_keep))
                cumulative_tokens = samples_to_keep * self.seq_len
                print("=== FAST Token Limit Applied (Using fixed seq_len) ===")
                
            else:
                # --- SLOW PATH: Iterative disk reading (O(N)) ---
                print(f"Warning: No seq_len provided. Reading from disk to count tokens to reach {self.total_tokens}...")
                cumulative_tokens = 0
                keep_indices = []
                
                for i in range(len(self.train_ds)):
                    x_teacher, x_student = self.train_ds[i]
                    token_count = x_teacher.shape[0]  # Assumes first dim is sequence length

                    if cumulative_tokens + token_count <= self.total_tokens:
                        cumulative_tokens += token_count
                        keep_indices.append(i)
                    else:
                        break
                print("=== Iterative Token Limit Applied ===")

            # Apply the calculated indices to the training dataset
            self.train_ds = Subset(self.train_ds, keep_indices)
            
            print(f"    Tokens allowed: {self.total_tokens}")
            print(f"    Tokens used:    {cumulative_tokens}")
            print(f"    Samples kept:   {len(self.train_ds)} out of {train_size}")
            print(f"======================================================\n")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )