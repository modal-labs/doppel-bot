"""
This modules is only used when MULTI_WORKSPACE_SLACK_APP=True.
"""
from typing import Optional
from modal import Secret
from .common import stub

my_image = stub.slack_image.debian_slim().pip_install("psycopg2")
with my_image.run_inside():
    import psycopg2

# if stub.is_inside(stub.slack_image):
#     import psycopg2


def insert_user(team_id: str, user: str) -> tuple[Optional[str], Optional[str]]:
    """Inserts a team into the database, if it doesn't already exist. If it does, returns the state and handle
    for the existing team."""

    with psycopg2.connect() as conn:
        cur = conn.cursor()
        cur.execute(
            f"INSERT INTO users (team_id, handle, state) VALUES ('{team_id}', '{user}', 'scraping') ON CONFLICT DO NOTHING",
        )
        if cur.rowcount == 0:
            cur = conn.cursor()
            cur.execute(f"SELECT state, handle FROM users WHERE team_id = '{team_id}'")
            state, handle = cur.fetchone()
            return state, handle
        conn.commit()
        return None, None


def update_state(team_id: str, user: str, state: str):
    with psycopg2.connect() as conn:
        cur = conn.cursor()
        cur.execute(f"UPDATE users SET state = '{state}' WHERE team_id = '{team_id}' AND handle = '{user}'")
        conn.commit()


def delete_user(team_id: str, user: str):
    with psycopg2.connect() as conn:
        cur = conn.cursor()
        cur.execute(f"DELETE FROM users WHERE team_id = '{team_id}' AND handle = '{user}'")
        conn.commit()


@stub.function(
    image=stub.slack_image,
    secrets=[Secret.from_name("neon-secret")],
)
def create_tables():
    with psycopg2.connect() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TYPE state AS ENUM ('training', 'success', 'failure', 'scraping');
            CREATE TABLE users(id SERIAL PRIMARY KEY, team_id TEXT NOT NULL, handle TEXT NOT NULL, state state);
            CREATE UNIQUE INDEX ix_users_team_id ON users (team_id);
            """
        )
        conn.commit()
