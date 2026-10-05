import os
import argparse
from huggingface_hub import HfApi

def push_directory_to_hub(local_dir: str, repo_id: str, hf_token: str = None):
    """
    Pushes a local directory to a Hugging Face Hub repository.
    """
    if not os.path.exists(local_dir):
        raise FileNotFoundError(f"The directory '{local_dir}' does not exist.")

    api = HfApi()

    # Step 1: Create the repository if it doesn't exist yet
    print(f"Ensuring repository '{repo_id}' exists...")
    api.create_repo(
        repo_id=repo_id, 
        repo_type="model", 
        exist_ok=True, 
        token=hf_token
    )

    # Step 2: Upload the folder contents
    print(f"Uploading contents from '{local_dir}' to '{repo_id}'...")
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        token=hf_token
    )
    
    print(f"✅ Success! Model uploaded to: https://huggingface.co/{repo_id}")

def main():
    """The entry point for the CLI command."""
    parser = argparse.ArgumentParser(
        description="Push a local model directory to the Hugging Face Hub."
    )
    parser.add_argument("path", type=str, help="Path to the local directory.")
    parser.add_argument("repo", type=str, help="The target repository on the Hub.")
    parser.add_argument("--token", "-t", type=str, default=None, help="Optional HF token.")

    args = parser.parse_args()

    push_directory_to_hub(
        local_dir=args.path, 
        repo_id=args.repo, 
        hf_token=args.token
    )

if __name__ == "__main__":
    main()