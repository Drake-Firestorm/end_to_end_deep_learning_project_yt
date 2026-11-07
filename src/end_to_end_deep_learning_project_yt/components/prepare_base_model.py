from pathlib import Path

import tensorflow as tf

from end_to_end_deep_learning_project_yt.entity.config_entity import ParamsConfig


class PrepareBaseModel:
    def __init__(self, config: ParamsConfig):
        self.config = config

    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params.image_size,
            weights=self.config.params.weights,
            include_top=self.config.params.include_top,
        )

        self.save_model(
            path=self.config.prepare_base_model.base_model_path, model=self.model
        )

    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(units=classes, activation="softmax")(
            flatten_in
        )

        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"],
        )

        full_model.summary()
        return full_model

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params.classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params.learning_rate,
        )

        self.save_model(
            path=self.config.prepare_base_model.updated_base_model_path,
            model=self.full_model,
        )
