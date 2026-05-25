from huggingface_hub import HfApi, login

login()
api = HfApi()

repo_id = "sanganaka/Qwen3.5-9B-padamitra-lora-1b"

print(f"Creating repository under organization: {repo_id}...")

api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

print("Uploading LoRA adapter weights...")
api.upload_folder(
    folder_path="./qwen3_5-it-glossary-1b-best-model",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload of Qwen3.5 instruct model's LoRA adapter for Padamitra"
)

print(f"Success! Model uploaded to: https://huggingface.co/{repo_id}")