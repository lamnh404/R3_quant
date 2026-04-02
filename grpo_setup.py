from huggingface_hub import snapshot_download, login
from kaggle_secrets import UserSecretsClient
import os

# Lấy token từ Kaggle Secrets
secrets = UserSecretsClient()
hf_token = secrets.get_secret("lalala")
login(token=hf_token)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights", "Qwen2.5-VL-7B-Instruct-GPTQ-Int4")

snapshot_download(
    repo_id="Qwen/Qwen2.5-VL-7B-Instruct-GPTQ-Int4",
    local_dir=WEIGHTS_DIR,
    local_dir_use_symlinks=False,
    token=hf_token,
)
