import logging
import os
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s]: %(name)s - %(levelname)s - %(message)s:"
)
logger = logging.getLogger(__name__)

project_name = "end_to_end_deep_learning_project_yt"
list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "requirements.txt",
    "research/trials.ipynb",
    "templates/index.html",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    logger.info(f"Creating directory: {filedir} for the file: {filename}")

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            logger.info(f"Creating empty file: {filepath}")
    else:
        logger.info(f"{filename} already exists")
