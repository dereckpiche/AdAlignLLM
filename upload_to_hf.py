"""
File: upload_to_hf.py
Summary: Uploads experiment artifacts to Hugging Face Hub.
"""

import argparse
import os

from huggingface_hub import HfApi

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model directory to upload.",
    )
    args.add_argument(
        "--hf_model_name",
        type=str,
        default=None,
        help="Name of the model on Hugging Face Hub.",
    )
    args = args.parse_args()
    model_path = args.model_path
    hf_model_name = args.hf_model_name
    api = HfApi()
    api.create_repo(repo_id=hf_model_name, repo_type="model", exist_ok=True)
    api.upload_large_folder(
        folder_path=model_path,
        repo_id=hf_model_name,
        repo_type="model",
        ignore_patterns=["**/*.render.*", "**/checkpoints/**", "**/iteration_**/**"],
    )
