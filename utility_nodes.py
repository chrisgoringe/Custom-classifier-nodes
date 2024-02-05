from nodes import SaveImage
from server import PromptServer
import math

class SaveIf(SaveImage):
    @classmethod
    def INPUT_TYPES(s):
        it = super().INPUT_TYPES()
        it['required']['scores'] = ("FLOATLIST", {"default":0.0})
        it['required']['threshold'] = ("FLOAT", {"default":0.5, "step":0.001})
        it['optional'] = {
            "optional_scores" : ("FLOATLIST", {}),
            "optional_threshold" : ("FLOAT", {"default":0.5, "step":0.001})
        }
        return it
    FUNCTION = "func"
    CATEGORY = "CustomClassifier"

    def func(self, scores, threshold, images, optional_scores=None, optional_threshold=None, **kwargs):
        assert len(scores)==len(images)
        for i, score in enumerate(scores):
            if score>=threshold: 
                if optional_scores is None or optional_threshold is None or optional_scores[i]>optional_threshold:
                    return self.save_images(images[i].unsqueeze_(0), **kwargs)
        return ()
      
class ScoreOperations:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "x":("FLOATLIST", {}),
                "y":("FLOATLIST", {}),
                "z":("FLOAT", {"default":0.0}),
                "operation":(["max(x,y)", "min(x,y)", "x+y", "x-y", "x*y", "x/y", "x if y>z"],{})
                },
            "optional":{}
        }
    RETURN_TYPES = ("FLOATLIST",)
    RETURN_NAMES = ("result",)
    
    FUNCTION = "func"
    CATEGORY = "CustomClassifier"

    def func(self,x,y,z,operation):
        assert len(x)==len(y)
        r = []
        for i,a in enumerate(x):
            b = y[i]
            if operation=="max(x,y)": r.append(max(a,b))
            if operation=="min(x,y)": r.append(min(a,b))
            if operation=="x+y": r.append(a+b)
            if operation=="x-y": r.append(a-b)
            if operation=="x*y": r.append(a*b)
            if operation=="x/y": r.append(a/b)
            if operation=="x if y>z" : r.append(a if b>z else 0)
        return (r,)

class ShowScores:
    FUNCTION = "func"
    CATEGORY = "CustomClassifier"
    OUTPUT_NODE = True
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{"scores":("FLOATLIST", {}),},
            "hidden": { "node_id": "UNIQUE_ID" },
        }
    RETURN_TYPES = ()
    
    def func(self, scores, node_id):
        text = ",".join(["{:6.4f}".format(score) for score in scores])
        PromptServer.instance.send_sync("cg.image_classify.textmessage", {"id": node_id, "message":text})
        return ()
    
class RunningAverage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{"scores":("FLOATLIST", {}), "mode":(['value','percentage'],{}),},
            "hidden": { "node_id": "UNIQUE_ID" },
        }
                    
    FUNCTION = "func"
    CATEGORY = "CustomClassifier"
    RETURN_TYPES = ("FLOAT","STRING",)
    RETURN_NAMES = ("Average","Av_string",)
    OUTPUT_NODE = True

    def __init__(self):
        self.node_id = 0
        self.reset()

    def reset(self):
        self.count = 0
        self.mean = 0
        self.Q = 0
        if self.node_id: PromptServer.instance.send_sync("cg.image_classify.textmessage", {"id": self.node_id, "message":""})

    def func(self, scores, mode, node_id):
        Messages.register(node_id,self)
        self.node_id = node_id
        old_mean = self.mean
        
        self.mean = (self.mean*self.count + sum(scores))/(self.count+len(scores))
        self.Q += sum((x-old_mean)*(x-self.mean) for x in scores)
        self.count += len(scores)
        
        if mode=='percentage':
            text = "{:>6.2f} +/- {:>6.2f} % ({:>3})".format(100*self.mean, 100*math.sqrt(self.Q/self.count), self.count)
        else:
            text = "{:>6.3f} +/- {:>6.2f} ({:>3})".format(self.mean, math.sqrt(self.Q/self.count), self.count)
        PromptServer.instance.send_sync("cg.image_classify.textmessage", {"id": node_id, "message":text})
        return (self.mean,"{:>6.3f}".format(self.mean),)

class Messages:
    nodes = {}
    @classmethod
    def register(cls, node_id, node:RunningAverage):
        cls.nodes[node_id] = node
    @classmethod
    def reset(cls):
        for nid in cls.nodes: cls.nodes[nid].reset()

routes = PromptServer.instance.routes
@routes.post('/image_classify_reset')
async def image_classify_reset(request):
    post = await request.post()
    Messages.reset()

