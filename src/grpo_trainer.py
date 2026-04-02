import torch
import sys
import os
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoProcessor
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, '..')))

from model.lora_setup import apply_lora_to_quantized_model
from src.rewards import format_reward_func, accuracy_reward_func
from src.utils import prepare_scienceqa_for_grpo


def train_quan_grpo(model_dir: str, train_data, output_dir: str):
    model_dir  = os.path.abspath(model_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading processor from: {model_dir}")
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)

    print(f"Loading model from: {model_dir}")
    peft_model = apply_lora_to_quantized_model(model_dir)
    peft_model.print_trainable_parameters()

    print("Preparing dataset...")
    grpo_dataset = prepare_scienceqa_for_grpo(train_data)
    print(f"Dataset size: {len(grpo_dataset)} samples")

    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=1e-6,
        lr_scheduler_type="cosine",
        logging_steps=1,
        max_steps=500,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        num_generations=4,
        max_prompt_length=512,
        max_completion_length=1024,
        bf16=False,  
        fp16=True,  
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=peft_model,
        processing_class=processor,
        reward_funcs=[format_reward_func, accuracy_reward_func],
        args=training_args,
        train_dataset=grpo_dataset,
    )

    print("Starting GRPO training...")
    trainer.train()

    print(f"\nĐang lưu mô hình tại: {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    print("Done.")


if __name__ == "__main__":
    MODEL_DIR  = os.path.join(BASE_DIR, "..", "weights", "Qwen2.5-VL-7B-Instruct-GPTQ-Int4")
    OUTPUT_DIR = os.path.join(BASE_DIR, "..", "checkpoints", "quan_grpo_7b")

    print("Loading ScienceQA dataset...")
    raw_scienceqa = load_dataset("derek-thomas/ScienceQA", split="validation")

    train_quan_grpo(MODEL_DIR, raw_scienceqa, OUTPUT_DIR)
