import json
import modal
from typing import Optional
from datetime import datetime, timedelta, timezone
import os
from typing import Iterable

from .common import (
    app,
    Conversation,
    VOL_MOUNT_PATH,
    output_vol,
    get_user_data_path,
    get_messages_for_slack_thread,
)

scraper_kwargs = dict(
    image=modal.Image.debian_slim().pip_install("slack-sdk"),
    secrets=[modal.Secret.from_name("slack-finetune-secret")],
)

# Cache for slack threads.
slack_cache = modal.Dict.from_name("slack-conversation-dict", create_if_missing=True)

MINUTES = 60  # seconds
HOURS = 60 * MINUTES


def make_slack_client(bot_token: str):
    from slack_sdk import WebClient
    from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler

    class CustomRetryHandler(RateLimitErrorRetryHandler):
        def prepare_for_next_attempt(self, **kwargs):
            super().prepare_for_next_attempt(**kwargs)
            print("Retrying...", kwargs["request"].url, kwargs["state"].current_attempt)

    client = WebClient(token=bot_token)
    client.retry_handlers.append(CustomRetryHandler(max_retry_count=8))

    return client


def get_thread_replies_cached(client, ts: str, channel_id: str):
    if ts in slack_cache:
        return slack_cache[ts]

    result = client.conversations_replies(channel=channel_id, ts=ts, limit=1000)
    messages = result["messages"]
    slack_cache[ts] = messages
    return messages


ChannelId = str
ChannelName = str
Channel = tuple[ChannelId, ChannelName]


@app.function(**scraper_kwargs)
def get_channels(bot_token: str) -> Iterable[Channel]:
    client = make_slack_client(bot_token)
    result = client.conversations_list(limit=1000)
    channels = result["channels"]
    for c in channels:
        if not c["is_shared"] and not c["is_archived"]:
            yield c["id"], c["name"]


UserDisplayName = UserRealName = str
UserIdMap = dict[str, tuple[UserDisplayName, UserRealName]]


@app.function(**scraper_kwargs)
def get_user_id_map(bot_token: str) -> UserIdMap:
    """Produce a dictionary mapping user id to (display name, real name)."""
    client = make_slack_client(bot_token)
    cursor = None
    user_id_map = {}
    while True:
        response = client.users_list(limit=1000, cursor=cursor)
        for user in response["members"]:
            user_id_map[user["id"]] = (
                user["profile"]["display_name"],
                user["profile"]["real_name"],
            )

        if not response["has_more"]:
            break

        cursor = response["response_metadata"]["next_cursor"]

    return user_id_map


@app.function(
    **scraper_kwargs,
    timeout=2 * HOURS,
    concurrency_limit=10,  # Slack API applies rate limits
)
def get_conversations(
    channel: Channel,
    names: UserIdMap,
    target_user: UserDisplayName | UserRealName,
    min_message_length: int,
    cutoff_days: int,
    bot_token: str,
    team_id: Optional[str],
) -> list[Conversation]:
    """Fetch conversations via the Slack API for a given target_user in a given channel.

    Args:
        channel: A channel to scrape, provided in the format `(id, name)`
        names: A dictionary mapping user IDs to user display and real names.
        target_user: The display or real username to fetch conversations for.
        min_message_length: Minimum length of message by target user to include in scrape.
        cutoff_days: Number of days in the past to fetch conversations from.
        bot_token: Slack bot token for authentication.
        team_id: The workspace to scrape.
    """
    channel_id, channel_name = channel

    client = make_slack_client(bot_token)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=cutoff_days)).timestamp()

    identity = client.auth_test(team_id=team_id)

    # Join channel
    client.conversations_join(channel=channel_id)
    cursor = None
    threads: list[str] = []
    while True:
        result = client.conversations_history(
            channel=channel_id, oldest=cutoff, cursor=cursor, limit=1000
        )

        for message in result["messages"]:
            if "bot_id" in message:
                # Ignore threads from automations to avoid scrape taking too long.
                continue

            if "reply_count" in message and message["reply_count"] > 0:
                threads.append(message["ts"])

        if not result["has_more"]:
            break

        cursor = result["response_metadata"]["next_cursor"]

    print(f"Found {len(threads)} threads in {channel_name}.")

    conversations = []

    def is_target(message: dict) -> bool:
        if "user" not in message or "text" not in message:
            return False

        user = message["user"]
        try:
            display_name, real_name = names[user]
        except KeyError:
            return False

        return (display_name == target_user) or (real_name == target_user)

    for ts in threads:
        messages = get_thread_replies_cached(client, ts, channel_id)
        messages.sort(key=lambda m: m["ts"])

        messages_so_far = []
        for message in messages:
            messages_so_far.append(message)

            if is_target(message) and len(message["text"]) > min_message_length:
                conversation = get_messages_for_slack_thread(
                    messages_so_far, identity, target_user, is_target
                )
                conversations.append({"messages": conversation})

    return conversations


@app.function(
    **scraper_kwargs,
    retries=modal.Retries(
        max_retries=3,
        initial_delay=5.0,
        backoff_coefficient=2.0,
    ),
    timeout=3 * HOURS,
    volumes={VOL_MOUNT_PATH: output_vol},
)
def scrape(
    user: UserDisplayName | UserRealName,
    team_id: Optional[str] = None,
    bot_token: Optional[str] = None,
    min_message_length: int = 100,
    cutoff_days: int = 365,
):
    """Scrapes the Slack API for a target user's conversations and saves them in OpenAI Chat format.

    Args:
        user (str): The display or real Slack username to target with the scrape.

    Other arguments are passed to get_conversations. See that function's docstring for details.
    """
    print(f"Beginning scrape for {user}...")
    bot_token = bot_token or os.environ["SLACK_BOT_TOKEN"]
    conversations = []
    channels = list(get_channels.remote_gen(bot_token))
    users = get_user_id_map.remote(bot_token)

    for c in get_conversations.map(
        channels,
        kwargs=dict(
            names=users,
            target_user=user,
            min_message_length=min_message_length,
            cutoff_days=cutoff_days,
            bot_token=bot_token,
            team_id=team_id,
        ),
    ):
        conversations.extend(c)

    path = get_user_data_path(user, team_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Limit to (most recent) 30,000 samples.
    conversations = conversations[-30_000:]

    with open(path, "w") as f:
        json.dump(conversations, f, indent=2)

    output_vol.commit()

    samples = len(conversations)
    print(f"Finished scrape for {user} ({samples} samples found).")
    return samples
