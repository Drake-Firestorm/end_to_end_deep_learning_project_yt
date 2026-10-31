import zipfile
from urllib import request

from pathlib import Path

from end_to_end_deep_learning_project_yt.entity.config_entity import DataIngestionConfig
from end_to_end_deep_learning_project_yt.utils.common import (
    create_directories,
    get_size,
)
from end_to_end_deep_learning_project_yt.utils.custom_logger import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def download_file(self) -> None:
        target = Path(self.config.local_data_file)
        if target.exists():
            logger.info(
                f"File already exists of size: {get_size(Path(self.config.local_data_file))}"
            )
        else:
            create_directories([target.parent])
            logger.info(f"created directory at: {target.parent}")

            filename, headers = request.urlretrieve(
                url=self.config.source_url, filename=target
            )
            logger.info(f"{filename} downloaded with following info: \n{headers}")

    def extract_zip_file(self) -> None:
        """
        Extracts the zip file into data directory

        args:
            zip_file_path: str

        returns:
            None
        """
        unzip_path = Path(self.config.unzip_dir)

        create_directories([unzip_path])
        logger.info(f"created directory at: {unzip_path}")

        with zipfile.ZipFile(self.config.local_data_file) as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"zip extracted at: {unzip_path}")
