from nodes import SaveImage
from server import PromptServer

class SaveIf(SaveImage):
    @classmethod
    def INPUT_TYPES(s):
        it = super().INPUT_TYPES()
        it['required']['score'] = ("FLOAT", {"default":0.0})
        it['required']['threshold'] = ("FLOAT", {"default":0.5})
        return it
    FUNCTION = "func"
    CATEGORY = "ImageClassify"

    def func(self, score, threshold, **kwargs):
        if score>=threshold: return self.save_images(**kwargs)
        return ()
    
class RunningAverage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{"score":("FLOAT", {"default":0.0})},
            "hidden": { "node_id": "UNIQUE_ID" },
        }
                    
    FUNCTION = "func"
    CATEGORY = "ImageClassify"
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

    def func(self, score, node_id):
        Messages.register(node_id,self)
        self.node_id = node_id
        self.total += score
        self.count += 1
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

