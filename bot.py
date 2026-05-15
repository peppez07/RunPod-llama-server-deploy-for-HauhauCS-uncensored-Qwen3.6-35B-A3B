#!/usr/bin/env python3
"""Discord bot that proxies approved users to a llama-server backend.

Reads config from environment (or .env via python-dotenv if installed):

    DISCORD_TOKEN          Bot token from discord.com/developers
    ALLOWED_USER_IDS       Comma-separated Discord user IDs allowed to chat
    ALLOWED_CHANNEL_IDS    Comma-separated channel IDs (optional; empty = all)
    LLAMA_SERVER_URL       e.g. https://<podId>-8000.proxy.runpod.net/v1
    MODEL_NAME             Served-model alias (default: hauhau-aggressive)
    SYSTEM_PROMPT          System prompt prepended to every conversation
    DB_PATH                SQLite history file (default: ./history.db)
    HISTORY_LIMIT          Max history messages to send per turn (default: 30)
    MAX_TOKENS             Max completion tokens (default: 2048)
"""

import asyncio
import os
import sqlite3
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import discord
from discord import app_commands
from openai import AsyncOpenAI


def _int_set(name: str) -> set[int]:
    return {int(x) for x in os.environ.get(name, "").split(",") if x.strip()}


TOKEN = os.environ["DISCORD_TOKEN"]
ALLOWED_USERS = _int_set("ALLOWED_USER_IDS")
ALLOWED_CHANNELS = _int_set("ALLOWED_CHANNEL_IDS")
LLAMA_URL = os.environ.get("LLAMA_SERVER_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "hauhau-aggressive")
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", "You are a helpful assistant.")
DB_PATH = Path(os.environ.get("DB_PATH", "history.db"))
HISTORY_LIMIT = int(os.environ.get("HISTORY_LIMIT", "30"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2048"))

OPEN_MODE = not ALLOWED_USERS  # empty allowlist = anyone in a server with the bot
if OPEN_MODE:
    print("ALLOWED_USER_IDS empty — running in OPEN mode (any server member can chat).")

oai = AsyncOpenAI(base_url=LLAMA_URL, api_key="not-needed")


def db_init() -> sqlite3.Connection:
    db = sqlite3.connect(DB_PATH)
    db.execute("""CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        ts INTEGER NOT NULL
    )""")
    db.execute("CREATE INDEX IF NOT EXISTS idx_user_ts ON messages(user_id, ts)")
    db.commit()
    return db


def load_history(db: sqlite3.Connection, user_id: int) -> list[dict]:
    rows = db.execute(
        "SELECT role, content FROM messages WHERE user_id=? ORDER BY id DESC LIMIT ?",
        (user_id, HISTORY_LIMIT),
    ).fetchall()
    return [{"role": r, "content": c} for r, c in reversed(rows)]


def append_message(db: sqlite3.Connection, user_id: int, role: str, content: str) -> None:
    db.execute(
        "INSERT INTO messages(user_id, role, content, ts) VALUES (?, ?, ?, ?)",
        (user_id, role, content, int(time.time())),
    )
    db.commit()


def reset_history(db: sqlite3.Connection, user_id: int) -> None:
    db.execute("DELETE FROM messages WHERE user_id=?", (user_id,))
    db.commit()


def stats_for(db: sqlite3.Connection, user_id: int) -> tuple[int, int]:
    total = db.execute(
        "SELECT COUNT(*) FROM messages WHERE user_id=?", (user_id,)
    ).fetchone()[0]
    chars = db.execute(
        "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM messages WHERE user_id=?",
        (user_id,),
    ).fetchone()[0]
    return total, chars


DB = db_init()

intents = discord.Intents.default()
intents.message_content = True
intents.dm_messages = True
bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)


def chunk_for_discord(s: str, limit: int = 1990) -> list[str]:
    if len(s) <= limit:
        return [s or "(empty response)"]
    out, buf = [], ""
    for line in s.splitlines(keepends=True):
        while len(line) > limit:
            if buf:
                out.append(buf)
                buf = ""
            out.append(line[:limit])
            line = line[limit:]
        if len(buf) + len(line) > limit:
            out.append(buf)
            buf = ""
        buf += line
    if buf:
        out.append(buf)
    return out


@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} ({bot.user.id})")
    try:
        synced = await tree.sync()
        print(f"Synced {len(synced)} slash commands")
    except Exception as e:
        print(f"Slash sync failed: {e}")


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if not OPEN_MODE and message.author.id not in ALLOWED_USERS:
        return

    is_dm = message.guild is None
    is_mention = bot.user is not None and bot.user.mentioned_in(message)
    channel_ok = not ALLOWED_CHANNELS or message.channel.id in ALLOWED_CHANNELS

    if not is_dm and not (channel_ok or is_mention):
        return

    content = message.content
    if bot.user:
        for tag in (f"<@{bot.user.id}>", f"<@!{bot.user.id}>"):
            content = content.replace(tag, "")
    content = content.strip()
    if not content:
        return

    history = load_history(DB, message.author.id)
    payload = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history
        + [{"role": "user", "content": content}]
    )

    reply_msg = await message.reply("…")
    full = ""
    last_edit = 0.0

    try:
        async with message.channel.typing():
            stream = await oai.chat.completions.create(
                model=MODEL_NAME,
                messages=payload,
                stream=True,
                max_tokens=MAX_TOKENS,
            )
            async for chunk in stream:
                delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                if not delta:
                    continue
                full += delta
                now = time.time()
                if now - last_edit >= 1.5:
                    preview = full[-1900:] + "▌"
                    try:
                        await reply_msg.edit(content=preview)
                    except discord.HTTPException:
                        pass
                    last_edit = now
    except Exception as e:
        await reply_msg.edit(content=f"⚠️ backend error: `{type(e).__name__}: {e}`")
        return

    chunks = chunk_for_discord(full)
    await reply_msg.edit(content=chunks[0])
    for extra in chunks[1:]:
        await message.channel.send(extra)

    append_message(DB, message.author.id, "user", content)
    append_message(DB, message.author.id, "assistant", full)


def _slash_authorized(user_id: int) -> bool:
    return OPEN_MODE or user_id in ALLOWED_USERS


@tree.command(name="reset", description="Clear your conversation history")
async def reset_cmd(interaction: discord.Interaction):
    if not _slash_authorized(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    reset_history(DB, interaction.user.id)
    await interaction.response.send_message("History cleared.", ephemeral=True)


@tree.command(name="stats", description="Show your conversation stats")
async def stats_cmd(interaction: discord.Interaction):
    if not _slash_authorized(interaction.user.id):
        await interaction.response.send_message("Not authorized.", ephemeral=True)
        return
    total, chars = stats_for(DB, interaction.user.id)
    mode = "open" if OPEN_MODE else "restricted"
    await interaction.response.send_message(
        f"Messages stored: **{total}**\nTotal characters: **{chars}**\n"
        f"Mode: `{mode}`  Backend: `{LLAMA_URL}`",
        ephemeral=True,
    )


if __name__ == "__main__":
    bot.run(TOKEN)
