from .custom_classify_nodes import *
from .utility_nodes import *
from .aesthetic_score_nodes import *
from .nudge_conditioning_node import *

NODE_CLASS_MAPPINGS = { 
    "Single Image Classifier" : ImageClassification,
    "Image Category Scorer": ImageCategoryScorer,
    "Image Aesthetic Scorer": ImageScorer,
    "Save If": SaveIf,
    "Running Average": RunningAverage,
    "Score Operations": ScoreOperations,
    "Show Scores": ShowScores,
    "Nudge Conditioning" : NudgeConditioning,
    }

__all__ = ['NODE_CLASS_MAPPINGS']

import shutil
import folder_paths

application_root_directory = os.path.dirname(folder_paths.__file__)
application_web_extensions_directory = os.path.join(application_root_directory, "web", "extensions", "cg_image_classify")
module_root_directory = os.path.dirname(os.path.realpath(__file__))
module_js_directory = os.path.join(module_root_directory, "js")
shutil.copytree(module_js_directory, application_web_extensions_directory, dirs_exist_ok=True)

folder_paths.folder_names_and_paths["customclassifier"] = ([os.path.join(folder_paths.models_dir, "customclassifier")], ["folder"])
folder_paths.folder_names_and_paths["customaesthetic"] = ([os.path.join(folder_paths.models_dir, "customaesthetic")], [".safetensors"])