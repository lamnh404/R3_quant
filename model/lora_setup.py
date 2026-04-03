import torch
from transformers import Qwen2VLForConditionalGeneration, GPTQConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

def apply_lora_to_quantized_model(model_path: str):

    gptq_config = GPTQConfig(
        bits=4,
        disable_exllama=True,  
    )

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        quantization_config=gptq_config,
        device_map="auto",
        torch_dtype=torch.float16,  
        local_files_only=True,
    )

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        exclude_modules=["visual"], 
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, lora_config)


    for name, param in peft_model.named_parameters():
        if "visual" in name:
            param.requires_grad = False

    peft_model.print_trainable_parameters()

    visual_is_training = any(
        p.requires_grad for name, p in peft_model.named_parameters() if "visual" in name
    )
    print(f"Vision Encoder training? -> {'⚠️ Đang train!' if visual_is_training else '✅ Đã freeze'}")

    return peft_model
