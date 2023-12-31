import torch
from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import os, json
import numpy as np
from server import PromptServer
import folder_paths
import comfy.model_management

def create_probability_calculator(model_directory, labels=[]):
    model = AutoModelForImageClassification.from_pretrained(model_directory, output_hidden_states=True)
    feature_extractor = AutoImageProcessor.from_pretrained(model_directory)

    device = comfy.model_management.vae_device()
    offload_device = comfy.model_management.vae_offload_device()

    def calculate_probabilities(image):
        if image.mode != "RGB": image = image.convert("RGB")
        inputs = feature_extractor(images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            model.to(device)
            outputs = model(**inputs)
            model.to(offload_device)
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
    def get_probs(cls, classifier, image):    

        for folder in folder_paths.folder_names_and_paths["customclassifier"][0]:
            if os.path.exists(os.path.join(folder,classifier)):
                with open(os.path.join(folder,classifier,'categories.json')) as f:
                    categories = json.load(f)['categories']

        i = 255. * image.cpu().numpy()
        image = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        return cls.probability_calculator(os.path.join(folder,classifier), categories)(image)
    
    @classmethod
    def get_model_directories(cls):
        model_directories = []
        for folder in folder_paths.folder_names_and_paths["customclassifier"][0]:
            for subfolder in os.listdir(folder):
                if os.path.exists(os.path.join(folder,subfolder,"categories.json")):
                    model_directories.append(subfolder)
        return model_directories

class ImageClassification(BaseClassifier):
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { 
            "classifier": (cls.get_model_directories(), {}),
            "image": ("IMAGE", {}),
        } }
    
    RETURN_TYPES = ("STRING", "FLOAT",  "STRING")
    RETURN_NAMES = ("Category", "Probability", "Details")
    
    def func(self, classifier, image):
        probabilities = self.get_probs(classifier, image[0])
        idx, category, prob = most_likely(probabilities)
        warn = "Only considered first image\n" if len(image)>1 else ""
        return (
            category, 
            prob, 
            warn+"\n".join(["{:>6.2f}% {:<20}".format(100*probabilities[l],l) for l in probabilities]),
        )

class ImageCategoryScorer(BaseClassifier):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
            "classifier": (cls.get_model_directories(), {}),
                "images": ("IMAGE", {}),
                "category": ("STRING",{"default":""}),
            },
            "hidden": { "node_id": "UNIQUE_ID" }, 
        }
    
    RETURN_TYPES = ("FLOATLIST",  "STRING")
    RETURN_NAMES = ("scores", "ProbString")

    def func(self, classifier, images, category, node_id):
        probs = [self.get_probs(classifier, image).get(category,0) for image in images]
        text = ",".join(["{:>6.2f}".format(100*prob) for prob in probs])
        PromptServer.instance.send_sync("cg.image_classify.textmessage", {"id": node_id, "message":text})
        return ( probs, text )
    
