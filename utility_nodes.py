from nodes import SaveImage
from server import PromptServer

class SaveIf(SaveImage):
    @classmethod
    def INPUT_TYPES(s):
        it = super().INPUT_TYPES()
        it['required']['scores'] = ("FLOATLIST", {"default":0.0})
        it['required']['threshold'] = ("FLOAT", {"default":0.5, "step":0.001})
        return it
    FUNCTION = "func"
    CATEGORY = "CustomClassifier"

    def func(self, scores, threshold, images, **kwargs):
        assert len(scores)==len(images)
        for i, score in enumerate(scores):
            if score>=threshold: return self.save_images(images[i].unsqueeze_(0), **kwargs)
        return ()
    
class ScoreOperations:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "x":("FLOATLIST", {}),
                "y":("FLOATLIST", {}),
                "operation":(["max(x,y)", "min(x,y)", "x+y", "x-y", "x*y", "x/y"],{})
                },
            "optional":{}
        }
    RETURN_TYPES = ("FLOATLIST",)
    RETURN_NAMES = ("result",)
    
    FUNCTION = "func"
    CATEGORY = "CustomClassifier"

    def func(self,x,y,operation):
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
            "required":{"scores":("FLOATLIST", {})},
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
        self.total = 0
        self.count = 0
        if self.node_id: PromptServer.instance.send_sync("cg.image_classify.textmessage", {"id": self.node_id, "message":""})

    def func(self, scores, node_id):
        Messages.register(node_id,self)
        self.node_id = node_id
        self.total += sum(scores)
        self.count += len(scores)
        text = "{:>6.2f}% ({:>3})".format(100*self.total/self.count, self.count)
        PromptServer.instance.send_sync("cg.image_classify.textmessage", {"id": node_id, "message":text})
        return (self.total/self.count,text,)

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

