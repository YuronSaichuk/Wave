from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_PACKAGE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_PACKAGE_DIR / ".env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="WAVE_",
    )

    db: str = "postgresql://postgres:wave@localhost:5435/wave"
    library: Path = Path("./data/library")
    cache: Path = Path("./data/cache")
    mixes: Path = Path("./data/mixes")

    @property
    def database_url(self) -> str:
        return self.db


settings = Settings()
