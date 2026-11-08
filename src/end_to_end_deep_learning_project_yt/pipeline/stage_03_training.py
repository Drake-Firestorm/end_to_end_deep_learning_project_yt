import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from end_to_end_deep_learning_project_yt.components.prepare_callbacks import (
    PrepareCallback,
)
from end_to_end_deep_learning_project_yt.components.training import (
    Training,
)
from end_to_end_deep_learning_project_yt.config.configuration import (
    ConfigurationManager,
)
from end_to_end_deep_learning_project_yt.utils.custom_logger import logger

STAGE_NAME = "Training"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_callbacks_config = config.get_prepare_callback_config()
        prepare_callbacks = PrepareCallback(config=prepare_callbacks_config)
        callback_list = prepare_callbacks.get_tb_ckpt_callbacks()

        training_config = config.get_training_config()
        training = Training(config=training_config)
        training.get_base_model()
        training.train_valid_generator()
        training.train(callback_list=callback_list)


if __name__ == "__main__":
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
