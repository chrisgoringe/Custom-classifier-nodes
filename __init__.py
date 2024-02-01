try:
    from custom_nodes.cg_custom_core import CC_VERSION
    if CC_VERSION < 2.3: raise Exception()
except:
    import os, git
    import folder_paths
    print("Installing cg_custom_core : you may need to restart ComfyUI")
    repo_path = os.path.join(os.path.dirname(folder_paths.__file__), 'custom_nodes', 'cg_custom_core')  
    repo = git.Repo.clone_from('https://github.com/chrisgoringe/cg-custom-core.git/', repo_path)
    repo.git.clear_cache()
    repo.close()

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
    "Save If": SaveIf,
    "Running Average": RunningAverage,
    "Score Operations": ScoreOperations,
    "Show Scores": ShowScores,
    "Sort by Scores": SortByScores,
    #"Nudge Conditioning" : NudgeConditioning,
    }

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', "WEB_DIRECTORY"]

# remove any old js installations
old_code_location = os.path.join(os.path.dirname(folder_paths.__file__), "web", "extensions", "cg_image_classify")
if os.path.exists(old_code_location):
    os.remove(old_code_location)
# end remove old