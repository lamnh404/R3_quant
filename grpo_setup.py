from huggingface_hub import snapshot_download
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "..", "weights", "Qwen2.5-VL-7B-Instruct-GPTQ-Int4")

snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct-GPTQ-Int4",
    local_dir=WEIGHTS_DIR,
)
