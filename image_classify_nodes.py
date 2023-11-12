import torch
from transformers import ViTImageProcessor, AutoModelForImageClassification, EfficientNetImageProcessor
from PIL import Image
import os, json
import numpy as np
from server import PromptServer
import folder_paths

def create_probability_calculator(model_folder, labels=[]):
    feature_extractor = ViTImageProcessor.from_pretrained(model_folder)
    device = "cuda"
    model = AutoModelForImageClassification.from_pretrained(model_folder, output_hidden_states=True).to(device)

    def calculate_probabilities(image):
        if image.mode != "RGB": image = image.convert("RGB")
        inputs = feature_extractor(images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
            if len(probs)==len(labels):
                return { labels[i]:float(probs[i]) for i in range(len(labels)) }
            else:
                return { str(i):float(probs[i]) for i in range(len(probs)) }
            
    return calculate_probabilities

def from_directory(dir):
    for image_filename in os.listdir(dir):
        image_filepath = os.path.join(dir, image_filename)
        yield image_filename, Image.open(image_filepath)

def most_likely(dict):
    prob = 0
    category = None
    idx = -1
    for i, c in enumerate(dict):
        if dict[c] > prob:
            prob = dict[c]
            category = c
            idx = i
    return idx, category, prob


class BaseClassifier:
    CATEGORY = "CustomClassifier"
    FUNCTION = "func"
    _pc = None
    _md = None
    _c = None

    @classmethod
    def probability_calculator(cls, model_directory, categories):
        if cls._md != model_directory or cls._c != categories:
            cls._pc = None
        if cls._pc is None:
            cls._md = model_directory
            cls._c = categories
            cls._pc = create_probability_calculator(model_directory, categories)
        return cls._pc
    
    @classmethod
    def get_probs(cls, model_directory, image):    
        try:
            with open(os.path.join(model_directory,'categories.json')) as f:
                categories = json.load(f)['categories']
        except:
            categories = []

        i = 255. * image[0].cpu().numpy()
        image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        return cls.probability_calculator(model_directory, categories)(image)


class ImageClassification(BaseClassifier):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "classifier": (folder_paths.get_filename_list("customclassifier"), ),
            "image": ("IMAGE", {}),
        } }
    
    RETURN_TYPES = ("STRING", "FLOAT",  "STRING")
    RETURN_NAMES = ("Category", "Probability", "Details")
    

    def func(self, classifier, image):
        probabilities = self.get_probs(classifier, image)
        idx, category, prob = most_likely(probabilities)

        return (
            category, 
            prob, 
            "\n".join(["{:>6.2f}% {:<20}".format(100*probabilities[l],l) for l in probabilities]),
        )

class ImageCategoryScorer(BaseClassifier):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
            "classifier": (folder_paths.get_filename_list("customclassifier"), ),
                "image": ("IMAGE", {}),
                "category": ("STRING",{"default":""}),
            },
            "hidden": { "node_id": "UNIQUE_ID" }, 
        }
    
    RETURN_TYPES = ("FLOAT",  "STRING")
    RETURN_NAMES = ("Probability", "ProbString")

    def func(self, classifier, image, category, node_id):
        probabilities = self.get_probs(classifier, image)
        prob = probabilities.get(category,0)
        text = "{:>6.2f}".format(100*prob)
        PromptServer.instance.send_sync("cg.image_classify.textmessage", {"id": node_id, "message":text})
        return ( prob, "{:>6.2f}".format(100*prob) )