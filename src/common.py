import modal
import random
from typing import Optional
from pathlib import Path

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

VOL_MOUNT_PATH = Path("/vol")

MULTI_WORKSPACE_SLACK_APP = False

WANDB_PROJECT = ""

MODEL_PATH = VOL_MOUNT_PATH / "model"

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
