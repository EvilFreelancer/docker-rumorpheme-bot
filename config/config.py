import os
from typing import List
from dataclasses import dataclass, field
from environs import Env

from .base import getenv


@dataclass
class Config:
    TOKEN: str = getenv("BOT_TOKEN", str)
    ADMINS_IDS: List[str] = field(default_factory=lambda: getenv("ADMINS_ID", list))
    DATABASE_URL: str = getenv("DATABASE_URL", str, 'sqlite+aiosqlite:///data/rumorpheme.db')
    MODEL_PATH: str = getenv("MODEL_PATH", str, 'evilfreelancer/ruMorpheme-v0.1')
    WORK_PATH = os.getcwd()


def load_config() -> Config:
    env = Env()
    env.read_env()
    return Config()
