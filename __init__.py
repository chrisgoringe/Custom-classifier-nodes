
#from .custom_classify_nodes import *
from .utility_nodes import *
from .aesthetic_score_nodes import *
#from .nudge_conditioning_node import *

import folder_paths
folder_paths.folder_names_and_paths["customclassifier"] = ([os.path.join(folder_paths.models_dir, "customclassifier")], ["folder"])
folder_paths.folder_names_and_paths["customaesthetic"] = ([os.path.join(folder_paths.models_dir, "customaesthetic")], [".safetensors"])

NODE_CLASS_MAPPINGS = { 
    #"Single Image Classifier" : ImageClassification,
    #"Image Category Scorer": ImageCategoryScorer,
    "Image Aesthetic Scorer": ImageScorer,
    "Conditioning Scorer" : ConditioningScorer,
    "Save If": SaveIf,
    "Running Average": RunningAverage,
    "Score Operations": ScoreOperations,
    "Show Scores": ShowScores,
    "Sort by Scores": SortByScores,
    #"Nudge Conditioning" : NudgeConditioning,
    }

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', "WEB_DIRECTORY"]
