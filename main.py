from src.end_to_end_deep_learning_project_yt.utils.custom_logger import logger
from src.end_to_end_deep_learning_project_yt.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    logger.info("x ========== x")
except Exception as e:
    logger.exception(e)
    raise
