from huggingface_hub import HfApi, login

login()
api = HfApi()

repo_id = "sanganaka/unsloth-gemma3ne4b-it-padamitra-lora-1a"

print(f"Creating repository under organization: {repo_id}...")

api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)

print("Uploading LoRA adapter weights...")
api.upload_folder(
    folder_path="./unsloth-gemma-3ne4b-it-glossary-1a-best-model",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload of Gemma3ne4b instruct model's LoRA adapter for Padamitra"
)

print(f"Success! Model uploaded to: https://huggingface.co/{repo_id}")