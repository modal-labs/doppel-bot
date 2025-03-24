"""
This module is only used when MULTI_WORKSPACE_SLACK_APP=True.
"""

import modal

from .common import app, slack_image

with slack_image.imports():
    import psycopg2

MAX_USERS_PER_TEAM = 10


class UserAlreadyExists(Exception):
    pass


class TooManyUsers(Exception):
    pass


def insert_user(team_id: str, user: str):
    """Inserts a user into the database, if it doesn't already exist, and if the team isn't full."""

    with psycopg2.connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM users WHERE team_id = %s", [team_id])
        count = cur.fetchone()[0]

        if count >= MAX_USERS_PER_TEAM:
            raise TooManyUsers()

        # Race possible here.
        cur.execute(
            "INSERT INTO users (team_id, handle, state) VALUES (%s, %s, 'scraping') ON CONFLICT DO NOTHING",
            [team_id, user],
        )

        if cur.rowcount == 0:
            raise UserAlreadyExists()


def list_users(team_id: str) -> list[tuple[str, str]]:
    with psycopg2.connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT handle, state FROM users WHERE team_id = %s", [team_id])
        return [(handle, state) for handle, state in cur.fetchall()]


def update_state(team_id: str, user: str, state: str):
    with psycopg2.connect() as conn:
        cur = conn.cursor()

        if state == "training":
            query = "UPDATE users SET state = %s, scraped_at = CURRENT_TIMESTAMP WHERE team_id = %s AND handle = %s"
        elif state == "trained":
            query = "UPDATE users SET state = %s, trained_at = CURRENT_TIMESTAMP WHERE team_id = %s AND handle = %s"
        else:
            query = "UPDATE users SET state = %s WHERE team_id = %s AND handle = %s"

        params = [state, team_id, user]
        cur.execute(query, params)
        conn.commit()


def delete_user(team_id: str, user: str):
    with psycopg2.connect() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE team_id = %s AND handle = %s", [team_id, user])
        conn.commit()


@app.function(
    image=slack_image,
    secrets=[modal.Secret.from_name("neon-secret")],
)
def create_tables():
    with psycopg2.connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TYPE state AS ENUM ('training', 'trained', 'scraping');
            CREATE TABLE users(
                id SERIAL PRIMARY KEY,
                team_id TEXT NOT NULL,
                handle TEXT NOT NULL,
                state state,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                scraped_at TIMESTAMP WITH TIME ZONE,
                trained_at TIMESTAMP WITH TIME ZONE
            );
            CREATE UNIQUE INDEX ix_users_team_id_handle ON users (team_id, handle);
            """
        )
        conn.commit()
