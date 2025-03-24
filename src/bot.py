import os
import time
import uuid

import modal

from .common import (
    MULTI_WORKSPACE_SLACK_APP,
    VOL_MOUNT_PATH,
    app,
    get_active_user_for_team_id,
    get_messages_for_slack_thread,
    get_user_checkpoint_path,
    output_vol,
    slack_image,
    update_active_user,
)
from .finetune import finetune
from .inference import Inference
from .scrape import scrape

with slack_image.imports():
    from fastapi import FastAPI, Request, Response

    from .img_utils import overlay_disguise


# Ephemeral caches
users_cache = modal.Dict.from_name("doppel-bot-users-cache", create_if_missing=True)
self_cache = modal.Dict.from_name("doppel-bot-self-cache", create_if_missing=True)
avatar_cache = modal.Dict.from_name("doppel-bot-avatar-cache", create_if_missing=True)

MAX_INPUT_LENGTH = 512  # characters, not tokens.


def get_users(team_id: str, client) -> dict[str, tuple[str, str]]:
    """Returns a mapping from display name to user ID and avatar."""
    try:
        users = users_cache[team_id]
    except KeyError:
        # TODO: lower TTL when we support it.
        users = {}
        cursor = None
        while True:
            result = client.users_list(limit=1000, cursor=cursor)
            for user in result["members"]:
                users[user["profile"]["display_name"]] = (
                    user["id"],
                    user["profile"]["image_512"],
                )
                users[user["profile"]["real_name"]] = (
                    user["id"],
                    user["profile"]["image_512"],
                )

            if not result["has_more"]:
                break

            cursor = result["response_metadata"]["next_cursor"]
        users_cache[team_id] = users
    return users


def get_identity(team_id: str, client) -> str:
    try:
        # TODO: lower TTL when we support it.
        return self_cache[team_id]
    except KeyError:
        self_cache[team_id] = self_id = client.auth_test(team_id=team_id)
        return self_id


def get_oauth_settings():
    from slack_bolt.oauth.oauth_settings import OAuthSettings
    from slack_sdk.oauth.installation_store import FileInstallationStore
    from slack_sdk.oauth.state_store import FileOAuthStateStore

    return OAuthSettings(
        client_id=os.environ["SLACK_CLIENT_ID"],
        client_secret=os.environ["SLACK_CLIENT_SECRET"],
        scopes=[
            "app_mentions:read",
            "channels:history",
            "channels:join",
            "channels:read",
            "chat:write",
            "chat:write.customize",
            "commands",
            "groups:history",
            "users.profile:read",
            "users:read",
        ],
        install_page_rendering_enabled=False,
        installation_store=FileInstallationStore(base_dir=VOL_MOUNT_PATH / "slack" / "installation"),
        state_store=FileOAuthStateStore(expiration_seconds=600, base_dir=VOL_MOUNT_PATH / "slack" / "state"),
    )


def post_to_slack(text, client, channel_id, thread_ts, icon_url, username):
    if text == "":
        return

    print(f"Sending message: {text}")
    client.chat_postMessage(
        channel=channel_id,
        text=text,
        thread_ts=thread_ts,
        icon_url=icon_url,
        username=username,
    )


def get_or_create_avatar_url(avatar_url: str, team_id: str, user: str) -> str:
    key = f"{team_id}-{user}"

    try:
        img_id = avatar_cache[key]
    except KeyError:
        img_bytes = overlay_disguise(avatar_url)
        img_id = str(uuid.uuid4())
        avatar_cache[key] = img_id
        avatar_cache[img_id] = img_bytes

    return f"{_asgi_app.web_url}/avatar/{img_id}.png"


def handle_train(team_id: str, user: str, client, respond):
    from .db import TooManyUsers, UserAlreadyExists, insert_user

    if MULTI_WORKSPACE_SLACK_APP:
        try:
            insert_user(team_id, user)
        except UserAlreadyExists:
            return respond(text=f"Team {team_id} already has {user} registered.")
        except TooManyUsers:
            return respond(text=f"Team {team_id} has too many users registered.")

    user_pipeline.spawn(team_id, client.token, user, respond)


def handle_list(team_id: str, users: list[str], respond):
    from .db import list_users

    active_user = get_active_user_for_team_id(team_id, users)
    if MULTI_WORKSPACE_SLACK_APP:
        users = list_users(team_id)
    else:
        path = VOL_MOUNT_PATH / team_id
        path.mkdir(parents=True, exist_ok=True)

        users = []

        for p in path.iterdir():
            adapter_path = get_user_checkpoint_path(p.name, team_id) / "adapter_config.json"
            if adapter_path.exists():
                users.append((p.name, "trained"))

    msg = "\n".join(
        f"{i}. {user} ({state}{', active' if user == active_user else ''})" for i, (user, state) in enumerate(users, 1)
    )

    return respond(text=msg if msg else "No users registered. Run /doppel train <user> first.")


@app.function(
    image=slack_image,
    secrets=[
        modal.Secret.from_name("slack-finetune-secret"),
        # TODO: Modal should support optional secrets.
        *([modal.Secret.from_name("neon-secret")] if MULTI_WORKSPACE_SLACK_APP else []),
    ],
    # Has to outlive both scrape and finetune.
    timeout=60 * 60 * 4,
    volumes={VOL_MOUNT_PATH: output_vol},
    min_containers=1,
)
@modal.asgi_app(label="doppel")
def _asgi_app():
    from slack_bolt import App
    from slack_bolt.adapter.fastapi import SlackRequestHandler

    if MULTI_WORKSPACE_SLACK_APP:
        slack_app = App(oauth_settings=get_oauth_settings())
    else:
        # If we don't want to use multi-workspace auth, the client ID & secret interfere
        # with regular bot token auth.
        os.environ.pop("SLACK_CLIENT_ID", None)
        os.environ.pop("SLACK_CLIENT_SECRET", None)
        slack_app = App(
            signing_secret=os.environ["SLACK_SIGNING_SECRET"],
            token=os.environ["SLACK_BOT_TOKEN"],
        )

    fastapi_app = FastAPI()
    handler = SlackRequestHandler(slack_app)

    @slack_app.event("url_verification")
    def handle_url_verification(body, logger):
        challenge = body.get("challenge")
        return {"challenge": challenge}

    @slack_app.event("app_mention")
    def handle_app_mentions(body, say, client):
        team_id = body["team_id"]
        channel_id = body["event"]["channel"]
        ts = body["event"].get("thread_ts", body["event"]["ts"])

        users = get_users(team_id, client)
        identity = get_identity(team_id, client)
        bot_id = identity["bot_id"]

        messages = client.conversations_replies(channel=channel_id, ts=ts, limit=1000)["messages"]
        messages.sort(key=lambda m: m["ts"])

        user = get_active_user_for_team_id(team_id, users.keys())
        if user is None:
            say(text="No users trained yet. Run /doppel <user> first.", thread_ts=ts)
            return

        _, avatar_url = users[user]
        username = f"{user}-bot"

        def is_assistant(m):
            return m.get("bot_id") == bot_id and m.get("username") == username

        input = get_messages_for_slack_thread(messages, identity, user, is_assistant)

        print("Input: ", input)

        model = Inference()

        try:
            avatar_url = get_or_create_avatar_url(avatar_url, team_id, user)
            print("Created avatar URL: ", avatar_url)
        except Exception as e:
            print("Error creating disguised avatar URL: ", e)
            avatar_url = avatar_url

        current_message = ""

        for chunk in model.generate.remote_gen(input, user=user, team_id=team_id):
            current_message += chunk
            messages = current_message.split("BOT: ")
            if len(messages) > 1:
                # Send all complete messages except the last partial one
                for message in messages[:-1]:
                    post_to_slack(message, client, channel_id, ts, avatar_url, username)

                current_message = messages[-1]

        # Send any remaining message
        if current_message:
            post_to_slack(current_message, client, channel_id, ts, avatar_url, username)

    @slack_app.command("/doppel")
    def handle_doppel(ack, respond, command, client):
        ack()
        team_id = command["team_id"]
        users = get_users(team_id, client)

        cmds = command["text"].split(" ", maxsplit=1)

        output_vol.reload()

        if cmds[0] == "train":
            if len(cmds) < 2:
                return respond(text="Usage: /doppel train <user>")
            user = cmds[1]
            if user not in users:
                return respond(text=f"User {user} not found.")
            handle_train(team_id, user, client, respond)
        elif cmds[0] == "list":
            return handle_list(team_id, users.keys(), respond)
        elif cmds[0] == "switch":
            if len(cmds) < 2:
                return respond(text="Usage: /doppel switch <user>")
            user = cmds[1]
            if user not in users:
                return respond(text=f"User {user} not found. Try /doppel list to see all trained users.")
            if not update_active_user(team_id, user):
                return respond(text=f"User {user} not trained yet. Run /doppel train <user> first.")
            return respond(text=f"Switched to {user}.")
        else:
            return respond(
                text="Usage:\n"
                "/doppel train <user> - Train a bot to imitate a user\n"
                "/doppel list - List all trained bots\n"
                "/doppel switch <user> - Switch to chatting as a different bot"
            )

    @fastapi_app.post("/")
    async def root(request: Request):
        return await handler.handle(request)

    @fastapi_app.get("/slack/install")
    async def oauth_start(request: Request):
        return await handler.handle(request)

    @fastapi_app.get("/slack/oauth_redirect")
    async def oauth_callback(request: Request):
        return await handler.handle(request)

    @fastapi_app.get("/avatar/{img_id}.png")
    async def avatar(img_id: str):
        return Response(content=avatar_cache[img_id], media_type="image/png")

    return fastapi_app


@app.function(
    image=slack_image,
    # TODO: Modal should support optional secrets.
    secrets=([modal.Secret.from_name("neon-secret")] if MULTI_WORKSPACE_SLACK_APP else []),
    # Has to outlive both scrape and finetune.
    timeout=60 * 60 * 4,
)
def user_pipeline(team_id: str, token: str, user: str, respond):
    from .db import delete_user, update_state

    try:
        respond(text=f"Began scraping {user}. Note that this may take up to a few hours due to Slack rate limits.")
        samples = scrape.remote(user, team_id, bot_token=token)
        respond(text=f"Finished scraping {user} (found {samples} samples), starting training.")

        if MULTI_WORKSPACE_SLACK_APP:
            update_state(team_id, user, "training")

        t0 = time.time()

        finetune.remote(user, team_id)

        respond(text=f"Finished training {user} after {time.time() - t0:.2f} seconds.")

        if MULTI_WORKSPACE_SLACK_APP:
            update_state(team_id, user, "trained")
    except Exception as e:
        respond(text=f"Failed to train {user} ({e}). Try again in a bit, or reach out to support@modal.com!")
        if MULTI_WORKSPACE_SLACK_APP:
            delete_user(team_id, user)
        raise e
