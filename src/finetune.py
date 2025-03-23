import os
import subprocess
from pathlib import Path

import modal

from .common import (
    MODEL_NAME,
    MODEL_PATH,
    VOL_MOUNT_PATH,
    WANDB_PROJECT,
    app,
    get_user_data_path,
    get_user_model_path,
    output_vol,
)

MINUTES = 60  # seconds
HOURS = 60 * MINUTES

REMOTE_CONFIG_PATH = Path("/llama3_1_8B_lora.yaml")

image = (
    modal.Image.debian_slim()
    .pip_install("wandb", "torch", "torchao", "torchvision")
    .apt_install("git")
    .pip_install("git+https://github.com/pytorch/torchtune.git@06a837953a89cdb805c7538ff5e0cc86c7ab44d9")
    .add_local_file(Path(__file__).parent / "llama3_1_8B_lora.yaml", REMOTE_CONFIG_PATH.as_posix())
)


def download_model():
    subprocess.run(
        [
            "tune",
            "download",
            MODEL_NAME,
            "--output-dir",
            MODEL_PATH.as_posix(),
            "--ignore-patterns",
            "original/consolidated.00.pth",
        ]
    )


secrets = [modal.Secret.from_name("huggingface-secret")]
wandb_args = []

if WANDB_PROJECT:
    secrets.append(modal.Secret.from_name("my-wandb-secret"))
    wandb_args = [
        "metric_logger._component_=torchtune.training.metric_logging.WandBLogger",
        f"metric_logger.project={WANDB_PROJECT}",
    ]
else:
    wandb_args = []


@app.function(
    image=image,
    gpu="H100",
    volumes={VOL_MOUNT_PATH: output_vol},
    timeout=2 * HOURS,
    secrets=secrets,
)
def finetune(user: str, team_id: str = None, recipe_args: str = None, cleanup: bool = True):
    """Fine-tune a model on the user from the provided team with torchtune.

    Args:
        user: The real or display username of a Slack user.
        team_id: Identifier for a Slack workspace.
        recipe_args: Additional arguments to pass to the fine-tuning recipe.
        cleanup: Remove user data after fine-tuning. On by default.
    """
    import shlex

    if not MODEL_PATH.exists():
        print("Downloading model...")
        download_model()
        output_vol.commit()

    data_path = get_user_data_path(user, team_id)

    output_dir = get_user_model_path(user, team_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    if recipe_args is not None:
        recipe_args = shlex.split(recipe_args)
    else:
        recipe_args = []

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
            *wandb_args,
        ]
        + recipe_args
    )

    if cleanup and user != "test":
        # Delete scraped data after fine-tuning
        os.remove(data_path)
