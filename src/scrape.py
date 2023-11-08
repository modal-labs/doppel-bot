import json
from datetime import datetime, timedelta, timezone
import os
from typing import Iterable

from modal import Dict, Image, Secret, Retries

from .common import (
    stub,
    VOL_MOUNT_PATH,
    output_vol,
    user_data_path,
)

scraper_kwargs = dict(
    image=Image.debian_slim().pip_install("slack-sdk"),
    secrets=[Secret.from_name("slack-finetune-secret")],
)

# Cache for slack threads.
stub.slack_cache = Dict.persisted("slack-conversation-dict")


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
    if ts in stub.slack_cache:
        return stub.slack_cache[ts]

    result = client.conversations_replies(channel=channel_id, ts=ts, limit=1000)
    messages = result["messages"]
    stub.slack_cache[ts] = messages
    return messages


@stub.function(**scraper_kwargs)
def get_channel_ids(bot_token: str) -> Iterable[str]:
    client = make_slack_client(bot_token)
    result = client.conversations_list(limit=1000)
    channels = result["channels"]
    for c in channels:
        if not c["is_shared"] and not c["is_archived"]:
            yield c["id"]


@stub.function(**scraper_kwargs)
def get_user_id_map(bot_token: str) -> dict[str, tuple[str, str]]:
    """Map of user id to (display name, real name)."""
    client = make_slack_client(bot_token)
    cursor = None
    user_id_map = {}
    while True:
        response = client.users_list(limit=1000, cursor=cursor)
        for user in response["members"]:
            user_id_map[user["id"]] = (user["profile"]["display_name"], user["profile"]["real_name"])

        if not response["has_more"]:
            break

        cursor = response["response_metadata"]["next_cursor"]

    return user_id_map


@stub.function(
    **scraper_kwargs,
    timeout=3000,
    concurrency_limit=3,
)
def get_question_response_pairs(
    channel_id: str,
    names: dict[str, tuple[str, str]],
    target_users: list[str],
    min_message_length: int,
    cutoff_days: int,
    bot_token: str,
) -> list[tuple[str, str]]:
    client = make_slack_client(bot_token)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=cutoff_days)).timestamp()

    # Join channel
    client.conversations_join(channel=channel_id)
    cursor = None
    threads: list[str] = []
    while True:
        result = client.conversations_history(channel=channel_id, oldest=cutoff, cursor=cursor, limit=1000)

        for message in result["messages"]:
            if "reply_count" in message and message["reply_count"] > 0:
                threads.append(message["ts"])

        if not result["has_more"]:
            break

        cursor = result["response_metadata"]["next_cursor"]

    pairs = []

    for ts in threads:
        messages = get_thread_replies_cached(client, ts, channel_id)
        messages.sort(key=lambda m: m["ts"])

        # Construct pairs of contiguous non-target followed by target messages.
        input = ""
        output = ""
        for message in messages:
            if "user" not in message or "text" not in message:
                continue

            # user = names[message["user"]]
            user = message["user"]
            try:
                display_name, real_name = names[user]
            except KeyError:
                continue

            if (display_name in target_users) or (real_name in target_users):
                output += f"{user}: {message['text']}\n"
            else:
                if output:
                    if len(output) > min_message_length:
                        pairs.append((input, output))
                    input = ""
                    output = ""
                input += f"{user}: {message['text']}\n"

        if output and len(output) > min_message_length:
            pairs.append((input, output))

    return pairs


@stub.function(
    **scraper_kwargs,
    retries=Retries(
        max_retries=3,
        initial_delay=5.0,
        backoff_coefficient=2.0,
    ),
    timeout=60 * 60 * 2,
    network_file_systems={VOL_MOUNT_PATH.as_posix(): output_vol},
    cloud="gcp",
)
def scrape(
    user: str,
    # TODO: make Optional when supported by `modal run`.
    team_id: str = "",
    bot_token: str = "",
    min_message_length: int = 80,
    cutoff_days: int = 365,
):
    print(f"Beginning scrape for {user} in {team_id}...")
    bot_token = bot_token or os.environ["SLACK_BOT_TOKEN"]
    fine_tune_data = []
    channel_ids = list(get_channel_ids.remote_gen(bot_token))
    users = get_user_id_map.remote(bot_token)

    for pairs in get_question_response_pairs.map(
        channel_ids,
        kwargs=dict(
            names=users,
            target_users=[user],
            min_message_length=min_message_length,
            cutoff_days=cutoff_days,
            bot_token=bot_token,
        ),
    ):
        for question, response in pairs:
            fine_tune_data.append({"input": question, "output": response, "user": user})

    path = user_data_path(user, team_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Limit to 30,000 samples.
    fine_tune_data = fine_tune_data[:30_000]

    with open(path, "w") as f:
        json.dump(fine_tune_data, f, indent=2)

    samples = len(fine_tune_data)
    print(f"Finished scrape for {user} in {team_id} ({samples} samples found).")
    return samples
