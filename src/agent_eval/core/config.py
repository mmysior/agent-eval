from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


def _get_project_root() -> Path:
    current = Path(__file__).parent
    while (
        not (current / "uv.lock").exists() and not (current / "pyproject.toml").exists()
    ):
        if current.parent == current:
            return Path.cwd()
        current = current.parent
    return current


class Config(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True,
        extra="ignore",
    )

    # ==========================================
    # Agent Configuration
    # ==========================================
    DEFAULT_PROVIDER: str = ""
    DEFAULT_MODEL: str = "wcss-gpt-oss-20b"
    TOOL_ITERATION_LIMIT: int = 10

    # ==========================================
    # API Keys & Secrets
    # ==========================================
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    TOGETHER_API_KEY: str = ""
    MISTRAL_API_KEY: str = ""

    # ==========================================
    # Observability (Langfuse)
    # ==========================================
    LANGFUSE_TRACING_ENABLED: bool = False
    LANGFUSE_PUBLIC_KEY: str = ""
    LANGFUSE_SECRET_KEY: str = ""
    LANGFUSE_BASE_URL: str = ""

    # ==========================================
    # File Paths
    # ==========================================
    PROJECT_ROOT: Path = _get_project_root()
    PROMPTS_DIR: str = "prompts"
    CONFIG_DIR: str = "config"
    DATA_DIR: str = "data"
    TASKS_DIR: str = "tasks"

    @property
    def prompts_path(self) -> Path:
        """Get absolute path to prompts directory."""
        path = Path(self.PROMPTS_DIR)
        return path if path.is_absolute() else self.PROJECT_ROOT / path

    @property
    def config_path(self) -> Path:
        """Get absolute path to config directory."""
        path = Path(self.CONFIG_DIR)
        return path if path.is_absolute() else self.PROJECT_ROOT / path

    @property
    def data_path(self) -> Path:
        path = Path(self.DATA_DIR)
        return path if path.is_absolute() else self.PROJECT_ROOT / path

    @property
    def tasks_path(self) -> Path:
        path = Path(self.TASKS_DIR)
        return path if path.is_absolute() else self.PROJECT_ROOT / path

    @property
    def input_path(self) -> Path:
        return self.data_path / "input"

    @property
    def output_path(self) -> Path:
        return self.data_path / "output"


config = Config()
