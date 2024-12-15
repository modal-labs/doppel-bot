import modal
import random
from typing import Optional, Callable
from pathlib import Path

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = """You are an employee at a fast-growing startup. Below is an input conversation that takes place in the company's internal Slack. Continue the conversation appropriately in the same tone and style.

<@BOT> is your handle."""

VOL_MOUNT_PATH = Path("/vol")

MULTI_WORKSPACE_SLACK_APP = False

WANDB_PROJECT = "modal-labs/slack-finetune"

MODEL_PATH = VOL_MOUNT_PATH / "model"

app = modal.App(name="doppel-bot")

slack_image = (
    modal.Image.debian_slim()
    .pip_install("slack-sdk", "slack-bolt")
    .pip_install("fastapi")
    # .apt_install("wget")
    # .run_commands(
    #     "sh -c 'echo \"deb http://apt.postgresql.org/pub/repos/apt bullseye-pgdg main\" > /etc/apt/sources.list.d/pgdg.list'",
    #     "wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add -",
    # )
    # .apt_install("libpq-dev")
    # .pip_install("psycopg2")
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
        # TODO: fix this
        if (
            p / "model" / "epoch_0" / "adapter_config.json"
        ).exists() and p.name in users:
            filtered.append(p.name)
    if not filtered:
        return None
    user = random.choice(filtered)
    print(f"Randomly picked {user} out of {filtered}.")
    return user


MAX_INPUT_LENGTH = 4096  # characters, not tokens.


def get_messages_for_slack_thread(
    thread: list[dict], identity: dict, is_assistant: Callable[[dict], bool]
) -> list[dict]:
    # Go backwards and fetch messages until we hit the max input length.
    # Returns a conversation in the OpenAI chat dataset format.
    messages = []
    total = 0
    bot_user_id = identity["user_id"]
    bot_id = identity["bot_id"]

    current_message = []
    last_turn = "system"

    for message in reversed(thread):
        role = "assistant" if is_assistant(message) else "user"

        if role == "assistant":
            id = bot_id
        elif "user" in message:
            id = message["user"]
        elif "bot_id" in message:
            id = message["bot_id"]
        else:
            continue

        # Replace bot user id with bot id for consistency.
        text = message["text"].replace(f"<@{bot_user_id}>", f"<@{bot_id}>")
        text = f"{id}: {text}"

        if last_turn == role:
            current_message.append(text)
        else:
            if current_message:
                messages.append(
                    dict(role=last_turn, content="\n".join(reversed(current_message)))
                )
            current_message = [text]
            last_turn = role

        total += len(text)

        if total > MAX_INPUT_LENGTH:
            break

    if current_message:
        messages.append(
            dict(role=last_turn, content="\n".join(reversed(current_message)))
        )

    if last_turn == "assistant":
        # Special case because Llama doesn't like assistant messages right after system.
        messages.append(dict(role="user", content="\n"))

    return [
        dict(
            role="system",
            content=SYSTEM_PROMPT.replace("<@BOT>", f"<@{bot_id}>"),
        )
    ] + list(reversed(messages))
