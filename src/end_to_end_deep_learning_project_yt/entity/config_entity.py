from pathlib import Path

from pydantic import BaseModel, ConfigDict


class DataIngestionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifacts_root: Path
    data_ingestion: DataIngestionConfig
