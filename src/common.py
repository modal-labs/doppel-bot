from modal import Image, SharedVolume, Stub
import random
from typing import Optional
from pathlib import Path

VOL_MOUNT_PATH = Path("/vol")

MULTI_WORKSPACE_SLACK_APP = False

WANDB_PROJECT = ""


BASE_MODEL = "openlm-research/open_llama_7b_preview_300bt"
SUBFOLDER = "open_llama_7b_preview_300bt_transformers_weights"


def download_models():
    from transformers import LlamaForCausalLM, LlamaTokenizer

    LlamaForCausalLM.from_pretrained(BASE_MODEL, subfolder=SUBFOLDER)
    LlamaTokenizer.from_pretrained(BASE_MODEL, subfolder=SUBFOLDER)


openllama_image = (
    Image.micromamba()
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
        "datasets==2.10.1",
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

stub = Stub(name="doppel-bot", image=openllama_image)

stub.slack_image = (
    Image.debian_slim()
    .pip_install("slack-sdk", "slack-bolt")
    .apt_install("wget")
    .run_commands(
        "sh -c 'echo \"deb http://apt.postgresql.org/pub/repos/apt bullseye-pgdg main\" > /etc/apt/sources.list.d/pgdg.list'",
        "wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -",
    )
    .apt_install("libpq-dev")
    .pip_install("psycopg2")
)

output_vol = SharedVolume(cloud="gcp").persist("slack-finetune-vol")


def generate_prompt(user, input, output=""):
    return f"""You are {user}, employee at a fast-growing startup. Below is an input conversation that takes place in the company's internal Slack. Write a response that appropriately continues the conversation.

### Input:
{input}

### Response:
{output}"""


def user_data_path(user: str, team_id: Optional[str] = None) -> Path:
    return VOL_MOUNT_PATH / (team_id or "data") / user / "data.json"


def user_model_path(user: str, team_id: Optional[str] = None, checkpoint: Optional[str] = None) -> Path:
    path = VOL_MOUNT_PATH / (team_id or "data") / user
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
