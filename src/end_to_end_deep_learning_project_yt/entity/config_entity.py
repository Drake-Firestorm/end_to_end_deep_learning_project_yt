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
    epochs: int = Field(..., alias="EPOCHS")
    batch_size: int = Field(..., alias="BATCH_SIZE")
    is_augmentation: bool = Field(..., alias="AUGMENTATION")


class ParamsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prepare_base_model: PrepareBaseModelConfig
    params: Params


class PrepareCallbacksConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_dir: Path
    tensorboard_root_log_dir: Path
    checkpoint_model_filepath: Path


class TrainingBaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_dir: Path
    trained_model_path: Path


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    artifacts_root: Path
    data_ingestion: DataIngestionConfig
    prepare_base_model: PrepareBaseModelConfig
    prepare_callbacks: PrepareCallbacksConfig
    training: TrainingBaseConfig


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_config: TrainingBaseConfig
    updated_base_model_path: Path
    training_data: Path
    params: Params
