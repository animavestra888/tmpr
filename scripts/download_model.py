from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    repo_id = "Qwen/Qwen3-VL-2B-Instruct"
    local_dir = Path("models/Qwen3-VL-2B-Instruct")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Model downloaded to: {local_dir}")


if __name__ == "__main__":
    main()
