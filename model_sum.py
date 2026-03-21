import torch
from transformers import Qwen2_5_VLForConditionalGeneration
import os

def print_model_info(model_path, name):
    print(f"\n{'='*60}")
    print(f"🔍 ĐANG SOI CHI TIẾT MODEL: {name}")
    print(f"{'='*60}")
    
    # Load model lên CPU để không tốn VRAM lúc soi
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        device_map="cpu", 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
    )
    
    # 1. Xem dung lượng thực tế model chiếm dụng
    mem_bytes = model.get_memory_footprint()
    print(f"📦 Dung lượng bộ nhớ : {mem_bytes / (1024**3):.2f} GB")
    
    # 2. Đếm số lượng tham số
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🔢 Tổng số tham số   : {total_params / 1e9:.2f} Tỷ (Billion)")
    
    # 3. Soi data type của một vài lớp Linear điển hình
    print("\n🔬 CẤU TRÚC LỚP ẨN (Layer 0):")
    
    # Trích xuất thử một layer transformer đầu tiên để xem nó bị biến đổi thế nào
    layer_0 = model.model.layers[0]
    
    for name, module in layer_0.named_modules():
        # Chỉ in các lớp tính toán chính (như Linear hoặc QuantLinear)
        if "Linear" in str(type(module)):
            print(f"  - Tên module: {name}")
            print(f"    Loại      : {type(module).__name__}")
            # In ra thông tin các tensor bên trong module đó
            for param_name, param in module.named_parameters():
                print(f"    + {param_name}: shape={list(param.shape)}, dtype={param.dtype}")
            print("-" * 40)

if __name__ == "__main__":
    BASE_MODEL = r"./weights/Qwen2.5-VL-3B-Instruct"
    QUANT_MODEL = r"./weights/Qwen2.5-VL-3B-Instruct-GPTQ-Int3"
    
    print_model_info(BASE_MODEL, "BẢN GỐC (16-BIT)")
    print_model_info(QUANT_MODEL, "BẢN QUANTIZED (3-BIT)")