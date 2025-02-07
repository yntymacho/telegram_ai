from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
import os
from loguru import logger

# Load environment variables
env_path = Path(__file__).parent / '.env'
logger.info(f"Loading environment variables from: {env_path}")
load_dotenv(env_path, override=True)  # Force override existing env vars

# Debug: Print environment variables
logger.debug(f"GOOGLE_CREDENTIALS_JSON: {os.getenv('GOOGLE_CREDENTIALS_JSON')}")
logger.debug(f"GOOGLE_SHEETS_ID: {os.getenv('GOOGLE_SHEETS_ID')}")
logger.debug(f"TELEGRAM_BOT_TOKEN: {os.getenv('TELEGRAM_BOT_TOKEN')}")

def get_env_var(key: str, default: str = "") -> str:
    """Get environment variable with error handling"""
    value = os.getenv(key, default)
    logger.debug(f"Loading env var {key}: {value}")
    return value

class Config(BaseModel):
    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent
    CHROMA_DB_PATH: Path = PROJECT_ROOT / "chroma_db"
    
    # API Keys
    TELEGRAM_BOT_TOKEN: str = Field(default_factory=lambda: get_env_var('TELEGRAM_BOT_TOKEN'))
    MISTRAL_API_KEY: str = Field(default_factory=lambda: get_env_var('MISTRAL_API_KEY'))
    
    # Google Sheets Config
    GOOGLE_CREDENTIALS_PATH: Path = Field(
        default_factory=lambda: Path(get_env_var('GOOGLE_CREDENTIALS_JSON'))
    )
    GOOGLE_SHEET_ID: str = Field(default_factory=lambda: get_env_var('GOOGLE_SHEETS_ID'))
    
    # Model Config
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    MISTRAL_MODEL: str = "mistral-large-latest"
    MAX_CONTEXT_LENGTH: int = 4096
    
    # RAG Config
    TOP_K_RESULTS: int = 3
    TEMPERATURE: float = 0.3
    
    def validate_paths(self) -> None:
        """Validate all required paths exist"""
        logger.debug(f"Validating Google credentials path: {self.GOOGLE_CREDENTIALS_PATH}")
        if not self.GOOGLE_CREDENTIALS_PATH.exists():
            raise FileNotFoundError(
                f"Google credentials not found at: {self.GOOGLE_CREDENTIALS_PATH}"
            )
        
        # Create ChromaDB directory if it doesn't exist
        self.CHROMA_DB_PATH.mkdir(parents=True, exist_ok=True)
    
    def validate_credentials(self) -> None:
        """Validate all required credentials are set"""
        if not self.TELEGRAM_BOT_TOKEN:
            raise ValueError("TELEGRAM_BOT_TOKEN not set in environment")
        if not self.MISTRAL_API_KEY:
            raise ValueError("MISTRAL_API_KEY not set in environment")
        if not self.GOOGLE_SHEET_ID:
            raise ValueError("GOOGLE_SHEETS_ID not set in environment")

# Create global config instance
config = Config()
