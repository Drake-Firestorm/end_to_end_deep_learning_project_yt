import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from end_to_end_deep_learning_project_yt.components.evaluation import Evaluation
from end_to_end_deep_learning_project_yt.config.configuration import (
    ConfigurationManager,
)
from end_to_end_deep_learning_project_yt.utils.custom_logger import logger

STAGE_NAME = "Evaluation stage"


class EvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        val_config = config.get_validation_config()
        evaluation = Evaluation(val_config)
        evaluation.evaluation()
        evaluation.save_score()


if __name__ == "__main__":
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
