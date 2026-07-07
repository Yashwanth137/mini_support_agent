import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # API Keys
    google_api_key: str = ""
    gemini_api_key: str = ""
    groq_api_key: str = ""
    
    # Model configs
    embedding_model_name: str = "all-MiniLM-L6-v2"
    llm_model_name: str = "gemini-3.1-flash-lite"
    
    # Paths
    base_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = base_dir / "data"
    docs_dir: Path = data_dir / "docs"
    orders_csv: Path = data_dir / "orders.csv"
    faiss_index_path: Path = data_dir / "index.faiss"
    faiss_metadata_path: Path = data_dir / "metadata.pkl"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

# Make sure data and docs directories exist
settings.data_dir.mkdir(exist_ok=True)
settings.docs_dir.mkdir(exist_ok=True)
