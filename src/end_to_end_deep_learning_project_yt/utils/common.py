import json
import os
from pathlib import Path

import yaml
from pydantic import BaseModel, FilePath, validate_call

from .custom_logger import logger


@validate_call
def read_yaml(path_to_yaml: FilePath, schema: type[BaseModel]) -> BaseModel:
    """
    Reads yaml file and validates it against the given Pydantic model

    args:
        path_to_yaml: path like input
        schema: the BaseModel subclass describing the expected structure

    raises:
        ValueError: if yaml file is empty
        ValidationError: if the content does not match the schema

    returns:
        BaseModel: instance of the provided schema type
    """
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)

    if content is None:
        raise ValueError(f"yaml file is empty: {path_to_yaml}")
    if not isinstance(content, dict):
        raise TypeError(
            f"yaml file must contain a mapping at top level: {path_to_yaml}"
        )

    logger.info(f"yaml file: {path_to_yaml} loaded successfully")

    return schema.model_validate(content)


@validate_call(validate_return=True)
def create_directories(path_to_directories: list, verbose=True):
    """
    Create list of directories

    Args:
        path_to_directories (list): list of path to directories
        verbose (bool, optional): ignore if multiple dirs is to be created. Defaults to False
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@validate_call(validate_return=True)
def save_json(path: Path, data: dict):
    """
    Save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")


# @validate_call(validate_return=True)
# def load_json(path: Path) -> ConfigBox:
#     """
#     Load json files data
#
#     Args:
#         path (Path): path to json file
#
#     Returns:
#         ConfigBox: data as class attributes instead of dict
#     """
#     with open(path) as f:
#         content = json.load(f)
#
#     logger.info(f"json file loaded from: {path}")
#
#     return ConfigBox(content)
#
#
# @validate_call(validate_return=True)
# def save_bin(data: Any, path: Path):
#     """
#     Save binary file
#
#     Args:
#         data (Any): data to be saved as binary
#         path (Path): path to binary file
#     """
#     joblib.dump(value=data, filename=path)
#     logger.info(f"binary file saved at: {path}")
#
#
# @validate_call(validate_return=True)
# def load_bin(path: Path) -> Any:
#     """
#     Load binary file
#
#     Args:
#         path (Path): path to binary file
#
#     Returns:
#         Any: object stored in the file
#     """
#     data = joblib.load(path)
#     logger.info(f"binary file loaded from: {path}")
#
#     return data


@validate_call(validate_return=True)
def get_size(path: Path) -> str:
    """
    Get size in KB

    Args:
        path (Path): path of file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path) / 1024)

    return f"~ {size_in_kb} KB"


# def decodeImage(imgstring, filename):
#     imgdata = base64.b64decode(imgstring)
#     with open(filename, "wb") as f:
#         f.write(imgdata)
#         f.close()
#
#
# def encodeImageIntoBase64(croppedImagePath):
#     with open(croppedImagePath, "rb") as f:
#         return base64.b64encode(f.read())
