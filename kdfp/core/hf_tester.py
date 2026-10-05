from typing import Optional, Union

import lightning.pytorch as pl
import torch
import torchmetrics
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
from transformers import AutoModelForCausalLM, AutoTokenizer

from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.mixins import DtypeSerializerMixin, MCEvalMixin
from kdfp.schemas.tasks import TaskInstance
from kdfp.utils.data import load_task_data


class HFTesterModuleConfig(ConfigBase, DtypeSerializerMixin):
    model_path: str
    tokenizer_name: str
    task: TaskInstance
    seq_len: int
    batch_size: int
    seed: int
    torch_dtype: Union[str, torch.dtype]
    num_data_workers: Optional[int] = None
    n_samples: Optional[int] = None


class HFTesterModule(pl.LightningModule, MCEvalMixin):
    def __init__(self, train_config):
        super().__init__()
        if isinstance(train_config, dict):
            self.train_config = HFTesterModuleConfig(**train_config)
        else:
            self.train_config = train_config
        self.save_hyperparameters({"train_config": self.train_config.model_dump()})

        self.model = AutoModelForCausalLM.from_pretrained(
            self.train_config.model_path, dtype=self.train_config.torch_dtype
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.train_config.tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.train_config.task.task_type == "multiclass":
            acc = Accuracy(task=self.train_config.task.task_type, num_classes=4)
            f1 = F1Score(task=self.train_config.task.task_type, num_classes=4)
        else:  # binary
            acc = Accuracy(task=self.train_config.task.task_type)
            f1 = F1Score(task=self.train_config.task.task_type)

        # Eval metrics
        self.test_metrics = torchmetrics.MetricCollection(
            {
                "acc": acc,
                "f1": f1,
            },
            prefix="test_",
        )

    def _format_params(self, num_params: int) -> str:
        """Formats a number into a human-readable string (e.g., 1.2M, 2.5B)."""
        if num_params >= 1_000_000_000:
            return f"{num_params / 1_000_000_000:.2f}B"
        elif num_params >= 1_000_000:
            return f"{num_params / 1_000_000:.2f}M"
        elif num_params >= 1_000:
            return f"{num_params / 1_000:.2f}K"
        else:
            return f"{num_params}"

    # Assuming 'self.model' is defined within a class context
    def print_param_count(self, name: Optional[str]) -> None:
        """Prints the total and trainable parameter counts in a human-readable format."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print("\n" + "=" * 60)
        if name:
            print(f"{name.upper()} PARAMETER COUNTS:")

        # Use the helper function to format the output
        print(f"Total parameters:     {self._format_params(total_params)}")
        print(f"Trainable parameters: {self._format_params(trainable_params)}")
        print("=" * 60)

    def test_step(self, batch, batch_idx):
        preds, true_labels, correct_choice_loss = self._shared_eval_step(
            model=self.model, batch=batch, batch_idx=batch_idx
        )
        self.test_metrics.update(preds, true_labels)
        self.log("test_loss", correct_choice_loss, sync_dist=True)
        return {"test_loss": correct_choice_loss}

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self.test_metrics.reset()

    def test_dataloader(self):
        dataset = load_task_data(
            task=self.train_config.task,
            tokenizer=self.tokenizer,
            seq_len=self.train_config.seq_len,
            split="test",
            n_samples=self.train_config.n_samples,
        )

        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=self.train_config.num_data_workers,
            pin_memory=True,
        )
