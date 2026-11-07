from end_to_end_deep_learning_project_yt.components.prepare_base_model import (
    PrepareBaseModel,
)
from end_to_end_deep_learning_project_yt.config.configuration import (
    ConfigurationManager,
)
from end_to_end_deep_learning_project_yt.utils.custom_logger import logger

STAGE_NAME = "Prepare base model"


class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_params_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        prepare_base_model.update_base_model()


if __name__ == "__main__":
    try:
        logger.info("*************************")
        logger.info(f">>>>> stage {STAGE_NAME} started <<<<<")
        obj = PrepareBaseModelTrainingPipeline()
        logger.info(f">>>>> stage {STAGE_NAME} completed <<<<<")
        logger.info("x ========== x")
    except Exception as e:
        logger.exception(e)
        raise
