import folder_paths
import json, os
from .aesthetic_predictor import AestheticPredictor
from .clip import CLIP
from PIL import Image
import numpy as np

class BaseClassifier:
    CATEGORY = "CustomAestheticScorer"
    FUNCTION = "func"

    def __init__(self):
        self.model = None
        self.model_path = None
        self.model_metadata = None

    def load_model(self, path):
        if self.model_path and self.model_path==path: return
       
        with open(path, "rb") as f:
            data = f.read()
            n_header = data[:8]
        n = int.from_bytes(n_header, "little")
        metadata_bytes = data[8 : 8 + n]
        header = json.loads(metadata_bytes)
        self.model_metadata = header.get("__metadata__", {})
        self.model = AestheticPredictor(clipper=CLIP(pretrained=self.model_metadata.get('clip_model','ViT-L/14')), pretrained=path, dropouts=[0,0,0])
        self.model.eval()
        mean = float(self.model_metadata['mean_predicted_score'])
        std = float(self.model_metadata['stdev_predicted_score'])
        self.scale = lambda a : float((a-mean)/std)

        self.model_path = path

class ImageScorer(BaseClassifier):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"custom_model": (folder_paths.get_filename_list("customaesthetic"), ),
                             "images": ("IMAGE", {}),
        } }
    
    RETURN_TYPES = ("STRING", "IMAGE", "FLOATLIST", )
    RETURN_NAMES = ("scores_str", "images", "scores", )

    def func(self, custom_model, images):
        model_path = os.path.join(folder_paths.folder_names_and_paths["customaesthetic"][0][0], custom_model)
        self.load_model(model_path)
        scores = []
        for im in images:
            i = 255. * im.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            scores.append(self.scale(self.model.evaluate_image(img)))
        return ( ",".join(str(x) for x in scores), images, scores )
