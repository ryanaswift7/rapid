import argparse
import gc
import json
import os

import lm_eval
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.minillm import MiniLLMConfig, MiniLLMTrainer
from kdfp.comparisons.ted import EVAL_TASKS, run_evaluation

# --- Configuration ---
STUDENT_MODEL_ID = "EleutherAI/pythia-1.4b-deduped"
TEACHER_MODEL_ID = "EleutherAI/pythia-2.8b-deduped"
DATASET_ID = "ryanaswift7/fineweb-edu-pythia-kdfp-subsets-100m"
DATASET_CONFIG = "v1"
OUTPUT_DIR = "results/minillm"

# MiniLLM / Training Hyperparameters
MAX_LENGTH = 512
COMPLETION_LENGTH = 128
PROMPT_LENGTH = MAX_LENGTH - COMPLETION_LENGTH

# Eval Tasks
# EVAL_TASKS = [
#     "hellaswag",
#     "arc_challenge",
#     "arc_easy",
#     "boolq",
#     "triviaqa",
#     "piqa",
#     "sciq",
#     "winogrande",
#     "mmlu",
#     "bbh",
#     "drop",
# ]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_limit", type=int, default=None)
    args = parser.parse_args()

    print("--- Checking Model Status ---")
    model_already_trained = os.path.exists(os.path.join(OUTPUT_DIR, "config.json"))

    if model_already_trained:
        print(f"✅ Model already found at '{OUTPUT_DIR}'. Skipping training...")
    else:
        print("--- Starting Official MiniLLM Training ---")

        # 1. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token

        # 2. Load and Format Dataset
        print(f"Loading dataset: {DATASET_ID}")
        dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train")
        def format_for_minillm(example):
            full_tokens = tokenizer(
                example["text"], truncation=True, max_length=MAX_LENGTH
            )["input_ids"]

            # Set prompt length to max allowed, ensuring at least 10 tokens are left for completion
            prompt_len = min(PROMPT_LENGTH, len(full_tokens) - 10)

            # If the calculated prompt length is less than 10, the document is too short overall
            if prompt_len < 10:
                return {"prompt": ""}  

            prompt_tokens = full_tokens[:prompt_len]
            prompt_text = tokenizer.decode(prompt_tokens)

            return {"prompt": prompt_text}

        print("Formatting dataset (Splitting Prompts)...")
        dataset = dataset.map(format_for_minillm, remove_columns=dataset.column_names)
        
        # Filter out the empty strings we just created
        dataset = dataset.filter(lambda x: x["prompt"] != "")
        
        print(f"Training on {len(dataset)} documents.")

        # # 3. Load Models
        # print("Loading Teacher (2.8B)...")
        # teacher_model = AutoModelForCausalLM.from_pretrained(
        #     TEACHER_MODEL_ID,
        #     torch_dtype=torch.bfloat16,
        #     device_map="auto",
        #     attn_implementation="flash_attention_2",
        # )
        # teacher_model.eval()
        # teacher_model.requires_grad_(False)

        # print("Loading Student (1.4B)...")
        # student_model = AutoModelForCausalLM.from_pretrained(
        #     STUDENT_MODEL_ID,
        #     torch_dtype=torch.bfloat16,
        #     device_map="auto",
        #     attn_implementation="flash_attention_2",
        # )

        # 4. Configure MiniLLM
        minillm_config = MiniLLMConfig(
            model_init_kwargs=None,
            output_dir=OUTPUT_DIR,
            learning_rate=5e-6,
            num_train_epochs=1,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=8,
            bf16=True,
            logging_steps=100,
            save_strategy="epoch",
            report_to="none",
            gradient_checkpointing=True,
            ddp_find_unused_parameters=False,
            # model=STUDENT_MODEL_ID,
            max_completion_length=COMPLETION_LENGTH,
            # max_prompt_length=PROMPT_LENGTH,
            beta=0.5,
            rkl_advantage=True,
            single_step_decomposition=True,
            length_normalization=True,
            max_grad_norm=0.2,
            # use_vllm=True,
            # vllm_mode="colocate",
        )

        # 5. Initialize Trainer
        trainer = MiniLLMTrainer(
            model=STUDENT_MODEL_ID,
            teacher_model=TEACHER_MODEL_ID,
            train_dataset=dataset,
            # tokenizer=tokenizer,
            args=minillm_config,
        )

        # 6. Train & Save
        print("Starting Training...")
        trainer.train()
        print("Saving model...")
        trainer.save_model(OUTPUT_DIR)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print("✅ Training complete.")

        # --- NEW: Aggressive Memory Cleanup ---
        print("Wiping VRAM to make room for vLLM...")
        del trainer
        # del teacher_model
        # del student_model
        gc.collect()
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # --- EVALUATION PHASE ---
    # ---------------------------------------------------------
    run_evaluation(model_path=OUTPUT_DIR, tasks=EVAL_TASKS, eval_limit=args.eval_limit)
    # scores_only = {}

    # for shot_count in [0, 5]:
    #     print(f"\n--- Running {shot_count}-Shot Evaluation ---")

    #     # We point vLLM directly to the OUTPUT_DIR where the model was just saved
    #     results = lm_eval.simple_evaluate(
    #         model="vllm",
    #         model_args=f"pretrained={OUTPUT_DIR},dtype=auto,gpu_memory_utilization=0.8",
    #         tasks=EVAL_TASKS,
    #         num_fewshot=shot_count,
    #         batch_size="auto",
    #     )

    #     shot_key = f"{shot_count}_shot"
    #     scores_only[shot_key] = {}

    #     print(f"\nResults for {shot_count}-Shot:")
    #     if "results" in results:
    #         for task, metrics in results["results"].items():
    #             acc = metrics.get("acc,none", metrics.get("acc"))
    #             norm = metrics.get("acc_norm,none", metrics.get("acc_norm"))
    #             em = metrics.get("exact_match,none", metrics.get("exact_match"))
    #             f1 = metrics.get("f1,none", metrics.get("f1"))

    #             task_scores = {}
    #             if acc is not None:
    #                 task_scores["acc"] = acc
    #             if norm is not None:
    #                 task_scores["acc_norm"] = norm
    #             if em is not None:
    #                 task_scores["exact_match"] = em
    #             if f1 is not None:
    #                 task_scores["f1"] = f1

    #             scores_only[shot_key][task] = task_scores

    #             print_str = f"{task:<20} | "
    #             if norm is not None:
    #                 print_str += f"Norm: {norm:.4f}  "
    #             if acc is not None:
    #                 print_str += f"Acc: {acc:.4f}  "
    #             if em is not None:
    #                 print_str += f"EM: {em:.4f}  "
    #             if f1 is not None:
    #                 print_str += f"F1: {f1:.4f}  "

    #             print(print_str.strip())

    # # Save the consolidated scores to a JSON file next to the model
    # eval_output_path = os.path.join(OUTPUT_DIR, "eval_scores.json")
    # with open(eval_output_path, "w") as f:
    #     json.dump(scores_only, f, indent=4)
    # print(f"\n✅ Evaluation complete. Scores saved to {eval_output_path}")


if __name__ == "__main__":
    main()
