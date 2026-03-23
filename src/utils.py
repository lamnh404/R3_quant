import torch
from datasets import Dataset

def build_scienceqa_prompt(question: str, choices: list) -> str:
    prompt = f"{question}\n\nChoices:\n"
    labels = ["A", "B", "C", "D", "E"]
    
    if not choices:
        prompt += (
            "\nThink step by step and reason based on the image. "
            "Enclose your reasoning process within <think> </think> tags "
            "and provide your FINAL ANSWER within <answer> </answer> tags."
        )
        return prompt

    for i, choice in enumerate(choices):
        prompt += f"{labels[i]}. {choice}\n"
        
    valid_labels = labels[:len(choices)]
    if len(valid_labels) > 1:
        label_str = ", ".join(valid_labels[:-1]) + f" or {valid_labels[-1]}"
    else:
        label_str = valid_labels[0]
        
    prompt += (
        "\nThink step by step and reason based on the image. "
        "Enclose your reasoning process within <think> </think> tags "
        f"and provide your FINAL ANSWER (strictly write 1 letter: {label_str}) within <answer> </answer> tags."
    )
    return prompt

def prepare_scienceqa_for_grpo(raw_dataset, max_samples=None):
    formatted_data = {
        "prompt": [],    
        "ground_truth": [],
    }
    
    labels = ["A", "B", "C", "D", "E"]
    count = 0 
    
    for item in raw_dataset:
        if max_samples and count >= max_samples:
            break
            
        if item["image"] is None:
            continue
            
        text_prompt = build_scienceqa_prompt(item["question"], item["choices"])
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": item["image"]}, 
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]
        
        correct_index = item["answer"]
        correct_letter = labels[correct_index]
        
        formatted_data["prompt"].append(messages)
        formatted_data["ground_truth"].append(correct_letter)
        
        count += 1 
        
    return Dataset.from_dict(formatted_data)

def prepare_scienceqa_for_sft(raw_dataset, max_samples=None):
    formatted_data = {"messages": []} 
    
    labels = ["A", "B", "C", "D", "E"]
    count = 0 
    
    for item in raw_dataset:
        if max_samples and count >= max_samples:
            break
            
        if item["image"] is None:
            continue
            
        text_prompt = build_scienceqa_prompt(item["question"], item["choices"])
        user_message = {
            "role": "user",
            "content": [
                {"type": "image", "image": item["image"]}, 
                {"type": "text", "text": text_prompt}
            ]
        }
        
        correct_letter = labels[item["answer"]]
        solution_text = item.get("solution", "Suy luận logic dựa trên hình ảnh.")
        
        assistant_text = (
            f"<think>\n{solution_text}\n</think>\n"
            f"<answer>{correct_letter}</answer>"
        )
        assistant_message = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": assistant_text}
            ]
        }
        
        formatted_data["messages"].append([user_message, assistant_message])
        count += 1 
        
    return Dataset.from_dict(formatted_data)