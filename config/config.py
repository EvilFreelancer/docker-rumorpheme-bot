import os
from typing import List
from dataclasses import dataclass, field
from dotenv import load_dotenv
from .base import getenv
from environs import Env

env = Env()
env.read_env()


@dataclass
class Config:
    TOKEN: str = getenv("BOT_TOKEN", str)
    ADMINS_IDS: List[str] = field(default_factory=lambda: getenv("ADMINS_ID", list))
    DATABASE_URL: str = getenv("DATABASE_URL", str, 'sqlite+aiosqlite:///data/rumorpheme.db')
    MODEL_PATH: str = getenv("MODEL_PATH", str, 'evilfreelancer/ruMorpheme-v0.1')
    WORK_PATH = os.getcwd()


def load_config() -> Config:
    load_dotenv()
    return Config()
