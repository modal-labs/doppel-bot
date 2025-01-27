import modal
import random
from typing import Literal, Optional, Callable
from pathlib import Path

MODEL_NAME = "meta-llama/Meta-Llama-3.1-8B-Instruct"

SYSTEM_PROMPT = """You are {NAME}, an employee at a fast-growing startup. Below is an input conversation that takes place in the company's internal Slack. Continue the conversation appropriately in the same tone and style.

<@BOT> is your handle."""

VOL_MOUNT_PATH = Path("/vol")

MULTI_WORKSPACE_SLACK_APP = False

WANDB_PROJECT = ""

MODEL_PATH = VOL_MOUNT_PATH / "model"

app = modal.App(name="doppel-bot")

slack_image = (
    modal.Image.debian_slim()
    .pip_install("slack-sdk", "slack-bolt", "fastapi", "requests", "Pillow")
    .add_local_file(Path(__file__).parent / "disguise.png", remote_path="/disguise.png")
)

output_vol = modal.Volume.from_name("doppelbot-vol", create_if_missing=True)


def get_user_data_path(user: str, team_id: Optional[str] = None) -> Path:
    return VOL_MOUNT_PATH / (team_id or "data") / user / "data.json"


def get_user_model_path(user: str, team_id: Optional[str] = None) -> Path:
    return VOL_MOUNT_PATH / (team_id or "data") / user / "model"


def get_user_checkpoint_path(
    user: str, team_id: Optional[str] = None, version: Optional[int] = None
) -> Path:
    user_model_path = get_user_model_path(user, team_id)
    if version is None:
        version = find_latest_version(user_model_path)
    else:
        version = f"epoch_{int(version)}"

    return user_model_path / version


def find_latest_version(directory: Path) -> Path:
    import re

    pattern = re.compile(r"^epoch_(\d+)$")

    largest = -1

    for entry in directory.iterdir():
        if entry.is_dir():
            match = pattern.match(entry.name)
            if match:
                value = int(match.group(1))  # Extract the integer suffix
                if value > largest:
                    largest = value

    return f"epoch_{largest}"


def get_user_for_team_id(team_id: Optional[str], users: list[str]) -> Optional[str]:
    # Dumb: for now, we only allow one user per team.
    path = VOL_MOUNT_PATH / (team_id or "data")
    filtered = []
    for p in path.iterdir():
        adapter_path = get_user_checkpoint_path(p.name, team_id) / "adapter_config.json"
        if adapter_path.exists() and p.name in users:
            filtered.append(p.name)

    if not filtered:
        return None
    user = random.choice(filtered)
    print(f"Randomly picked {user} out of {filtered}.")
    return user


MAX_INPUT_LENGTH = 4096  # characters, not tokens.

Message = dict[str, str]  # OpenAI Message format -- role and content keys
Conversation = dict[Literal["messages"] : list[Message]]


def get_messages_for_slack_thread(
    thread: list[dict],
    identity: dict,
    target_user: str,
    is_assistant: Callable[[dict], bool],
) -> list[Message]:
    # Go backwards and fetch messages until we hit the max input length.
    # Returns a conversation in the OpenAI message format.
    messages = []
    total = 0
    bot_user_id = identity["user_id"]

    current_message = []
    last_turn = "system"

    for message in reversed(thread):
        role = "assistant" if is_assistant(message) else "user"

        if role == "assistant":
            id = "BOT"
        elif "user" in message:
            id = message["user"]
        elif "bot_id" in message:
            id = message["bot_id"]
        else:
            continue

        # Replace bot user id with bot id for consistency.
        text = message["text"].replace(f"<@{bot_user_id}>", "<@BOT>")
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
        dict(role="system", content=SYSTEM_PROMPT.replace("{NAME}", target_user))
    ] + list(reversed(messages))
