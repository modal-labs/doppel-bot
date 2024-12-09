import subprocess
from pathlib import Path
from typing import Optional

import modal

from .common import (
    MODEL_PATH,
    WANDB_PROJECT,
    app,
    output_vol,
    user_data_path,
    user_model_path,
    VOL_MOUNT_PATH,
)

REMOTE_CONFIG_PATH = Path("/llama3_1_8B_qlora.yaml")

image = (
    modal.Image.debian_slim()
    .pip_install("wandb", "torch", "torchao", "torchvision")
    .apt_install("git")
    .pip_install(
        "git+https://github.com/pytorch/torchtune.git@06a837953a89cdb805c7538ff5e0cc86c7ab44d9"
    )
    .add_local_file(
        Path(__file__).parent / "llama3_1_8B_qlora.yaml", REMOTE_CONFIG_PATH.as_posix()
    )
)


def download_model():
    subprocess.run(
        [
            "tune",
            "download",
            "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "--output-dir",
            MODEL_PATH.as_posix(),
            "--ignore-patterns",
            "original/consolidated.00.pth",
        ]
    )


secrets = [modal.Secret.from_name("huggingface-secret", environment_name="main")]

if WANDB_PROJECT:
    secrets.append(modal.Secret.from_name("my-wandb-secret"))


@app.function(
    image=image,
    gpu="H100",
    volumes={VOL_MOUNT_PATH: output_vol},
    timeout=60 * 60 * 2,
    secrets=secrets,
)
def finetune(user: str, team_id: Optional[str] = None):
    if not MODEL_PATH.exists():
        # More robust way to check if it's downloaded.
        print("Downloading model...")
        download_model()
        output_vol.commit()

    data_path = user_data_path(user, team_id)

    output_dir = user_model_path(user, team_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "tune",
            "run",
            "lora_finetune_single_device",
            "--config",
            REMOTE_CONFIG_PATH,
            f"output_dir={output_dir.as_posix()}",
            f"dataset_path={data_path.as_posix()}",
            f"model_path={MODEL_PATH.as_posix()}",
        ]
    )

    # # Delete scraped data after fine-tuning
    # os.remove(data_path)
