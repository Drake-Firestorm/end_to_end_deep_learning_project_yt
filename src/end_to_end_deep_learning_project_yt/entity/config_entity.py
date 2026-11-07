from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DataIngestionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path


class PrepareBaseModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path


class Params(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    image_size: list[int] = Field(..., alias="IMAGE_SIZE")
    learning_rate: float = Field(..., alias="LEARNING_RATE")
    include_top: bool = Field(..., alias="INCLUDE_TOP")
    weights: str = Field(..., alias="WEIGHTS")
    classes: int = Field(..., alias="CLASSES")


class ParamsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prepare_base_model: PrepareBaseModelConfig
    params: Params


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifacts_root: Path
    data_ingestion: DataIngestionConfig
    prepare_base_model: PrepareBaseModelConfig
