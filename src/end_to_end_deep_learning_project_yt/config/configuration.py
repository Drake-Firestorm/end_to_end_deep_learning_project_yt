from end_to_end_deep_learning_project_yt.constants import constants
from end_to_end_deep_learning_project_yt.entity.config_entity import (
    Config,
    DataIngestionConfig,
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
        # self.params = read_yaml(params_filepath, DataIngestionConfig)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        return config
