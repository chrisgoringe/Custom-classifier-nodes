import folder_paths
import json, os
from .aesthetic_predictor import AestheticPredictor
from PIL import Image
import numpy as np
import torch
from .ui_decorator import ui_signal
from comfy.model_management import get_torch_device, free_memory, unet_offload_device, soft_empty_cache

class BaseClassifier:
    CATEGORY = "CustomAestheticScorer"
    FUNCTION = "func"

    def __init__(self):
        self.model = None
        self.model_path = None
        self.model_metadata = None

    def load_model(self, path, device):
        if not (self.model_path and self.model_path==path): 
            with open(path, "rb") as f:
                data = f.read()
                n_header = data[:8]
            n = int.from_bytes(n_header, "little")
            metadata_bytes = data[8 : 8 + n]
            header = json.loads(metadata_bytes)
            self.model_metadata = header.get("__metadata__", {})
            self.model = AestheticPredictor.from_pretrained(path, use_cache=False, base_directory=os.path.dirname(os.path.realpath(__file__)))
        self.model.to(device)
        #self.model.to(torch.half)
        self.model_path = path

@ui_signal(['display_text'])
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
        free_memory(2*1024*1024*1024, get_torch_device())
        self.load_model(model_path, get_torch_device())
        scores = []
        for im in images:
            i = 255. * im.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            with torch.no_grad():
                scores.append(self.model.scale(self.model.evaluate_image(img)))
        score_string = ",".join(str(x) for x in scores)
        self.model.to(unet_offload_device())
        soft_empty_cache()
        return ( score_string, images, scores, score_string )

class SortByScores:
    CATEGORY = "CustomAestheticScorer"
    FUNCTION = "func"    
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE", {}), "scores": ("FLOATLIST",), "order": (["descending", "ascending"], ) } }
    
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images", )

    def func(self, images:torch.Tensor, scores, order):
        score_image = list( (score, images[i]) for i, score in enumerate(scores) )
        score_image.sort(reverse=(order=='descending'))
        sorted_images = list( si[1] for si in score_image )
        result = torch.stack( sorted_images )
        return (result, )