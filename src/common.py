import modal
import random
from typing import Optional
from pathlib import Path

VOL_MOUNT_PATH = Path("/vol")

MULTI_WORKSPACE_SLACK_APP = False

WANDB_PROJECT = ""

MODEL_PATH = VOL_MOUNT_PATH / "model"


def download_models():
    from transformers import LlamaForCausalLM, LlamaTokenizer

    model_name = "openlm-research/open_llama_7b"

    model = LlamaForCausalLM.from_pretrained(model_name)
    model.save_pretrained(MODEL_PATH)

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(MODEL_PATH)


openllama_image = (
    modal.Image.micromamba()
    .micromamba_install(
        "cudatoolkit=11.7",
        "cudnn=8.1.0",
        "cuda-nvcc",
        channels=["conda-forge", "nvidia"],
    )
    .apt_install("git")
    .pip_install(
        "accelerate==0.18.0",
        "bitsandbytes==0.37.0",
        "bitsandbytes-cuda117==0.26.0.post2",
        "datasets==2.14.6",
        "fire==0.5.0",
        "gradio==3.23.0",
        "peft @ git+https://github.com/huggingface/peft.git@e536616888d51b453ed354a6f1e243fecb02ea08",
        "transformers @ git+https://github.com/huggingface/transformers.git@a92e0ad2e20ef4ce28410b5e05c5d63a5a304e65",
        "torch==2.0.0",
        "torchvision==0.15.1",
        "sentencepiece==0.1.97",
    )
    .run_function(download_models)
    .pip_install("wandb==0.15.0")
)

app = modal.App(name="doppel-bot")

slack_image = (
    modal.Image.debian_slim()
    .pip_install("slack-sdk", "slack-bolt")
    .apt_install("wget")
    .run_commands(
        "sh -c 'echo \"deb http://apt.postgresql.org/pub/repos/apt bullseye-pgdg main\" > /etc/apt/sources.list.d/pgdg.list'",
        "wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -",
    )
    .apt_install("libpq-dev")
    .pip_install("psycopg2")
)

output_vol = modal.Volume.from_name("doppelbot-vol", create_if_missing=True)


def user_data_path(user: str, team_id: Optional[str] = None) -> Path:
    return VOL_MOUNT_PATH / (team_id or "data") / user / "data.json"


def user_model_path(
    user: str, team_id: Optional[str] = None, checkpoint: Optional[str] = None
) -> Path:
    path = VOL_MOUNT_PATH / (team_id or "data") / user / "model"
    if checkpoint:
        path = path / checkpoint
    return path


def get_user_for_team_id(team_id: Optional[str], users: list[str]) -> Optional[str]:
    # Dumb: for now, we only allow one user per team.
    path = VOL_MOUNT_PATH / (team_id or "data")
    filtered = []
    for p in path.iterdir():
        # Check if finished fine-tuning.
        if (path / p / "adapter_config.json").exists() and p.name in users:
            filtered.append(p.name)
    if not filtered:
        return None
    user = random.choice(filtered)
    print(f"Randomly picked {user} out of {filtered}.")
    return user
