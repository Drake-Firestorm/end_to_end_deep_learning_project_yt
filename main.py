import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from src.end_to_end_deep_learning_project_yt.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from src.end_to_end_deep_learning_project_yt.pipeline.stage_02_prepare_base_model import (
    PrepareBaseModelTrainingPipeline,
)
from src.end_to_end_deep_learning_project_yt.pipeline.stage_03_training import (
    ModelTrainingPipeline,
)
from src.end_to_end_deep_learning_project_yt.pipeline.stage_04_evaluation import (
    EvaluationPipeline,
)
from src.end_to_end_deep_learning_project_yt.utils.custom_logger import logger

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


STAGE_NAME = "Prepare base model"

try:
    logger.info("*************************")
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    prepare_base_model = PrepareBaseModelTrainingPipeline()
    prepare_base_model.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    logger.info("x ========== x")
except Exception as e:
    logger.exception(e)
    raise


STAGE_NAME = "Training"

try:
    logger.info("*************************")
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    logger.info("x ========== x")
except Exception as e:
    logger.exception(e)
    raise


STAGE_NAME = "Evaluation stage"
try:
    logger.info("*************************")
    logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
    model_trainer = EvaluationPipeline()
    model_trainer.main()
    logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
    logger.info("x ========== x")
except Exception as e:
    logger.exception(e)
    raise
