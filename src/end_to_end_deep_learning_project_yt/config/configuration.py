import os
from pathlib import Path

from end_to_end_deep_learning_project_yt.constants import constants
from end_to_end_deep_learning_project_yt.entity.config_entity import (
    Config,
    DataIngestionConfig,
    Params,
    ParamsConfig,
    PrepareCallbacksConfig,
)
from end_to_end_deep_learning_project_yt.utils.common import (
    create_directories,
    read_yaml,
)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath=constants.CONFIG_FILE_PATH,
        params_filepath=constants.PARAMS_FILE_PATH,
    ) -> None:
        self.config = read_yaml(config_filepath, Config)
        self.params = read_yaml(params_filepath, Params)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        return config

    def get_params_config(self) -> ParamsConfig:
        config = self.config.prepare_base_model
        params = self.params

        create_directories([config.root_dir])

        params_config = ParamsConfig(prepare_base_model=config, params=params)

        return params_config

    def get_prepare_callback_config(self) -> PrepareCallbacksConfig:
        prepare_callback_config = self.config.prepare_callbacks

        model_ckpt_dir = os.path.dirname(
            prepare_callback_config.checkpoint_model_filepath
        )

        create_directories(
            [
                Path(model_ckpt_dir),
                Path(prepare_callback_config.tensorboard_root_log_dir),
            ]
        )

        return prepare_callback_config
