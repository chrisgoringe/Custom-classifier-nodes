from safetensors.torch import load_file
import torch

class BaseNudge:
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "func"
    CATEGORY = "conditioning"

    REQUIRED = {}

    STD = {  "conditioning": ("CONDITIONING", ), 
                    "token_strength": ("FLOAT", {"default": 1.0, "step": 0.01}),
                    "replace_token": ("INT",{"default":76, "min":-1, "max":76}),
                    "pool_strength": ("FLOAT", {"default": 1.0, "step": 0.01}) }

    @classmethod
    def ADD_STANDARD_INPUT_TYPES(cls, it):
        for k in cls.STD:
            it['required'][k] = cls.STD[k]
        for k in cls.REQUIRED: it['required'][k] = cls.REQUIRED[k]
        return it

    def _func(self, conditioning_in:list, conditioning_nudge:torch.Tensor, token_strength:float, replace_token:int, pool_strength:float):
        if len(conditioning_in)>1: print("Warning - only the first conditioning is nudged")
        c0:torch.Tensor = conditioning_in[0][0].clone()
        dic = conditioning_in[0][1].copy()
        assert conditioning_nudge.shape[0]==c0.shape[2], "Nudge incompatible with conditioning"

        if token_strength:
            for rt in range(77) if replace_token<0 else [replace_token,]:
                replaced = c0[:,rt,:]
                c0[:,rt,:] = torch.mul(conditioning_nudge,token_strength) + torch.mul(replaced,1-token_strength)

        if pool_strength:
            pool:torch.Tensor = conditioning_in[0][1]['pooled_output'].clone().squeeze()
            pool = torch.mul(conditioning_nudge[-len(pool):], pool_strength) + torch.mul(pool, 1-pool_strength)
            dic['pooled_output'] = pool.unsqueeze(0)

        ret = [[c0,dic],]
        for more in conditioning_in[1:]: ret.append(more)
        return( ret, )

class NudgeConditioning(BaseNudge):
    @classmethod
    def INPUT_TYPES(cls):
        it = {"required": { "nudge_file": ("STRING", {"default":""}) } }
        return cls.ADD_STANDARD_INPUT_TYPES(it)

    def func(self, conditioning, nudge_file:str, **kwargs):
        c1:torch.Tensor = load_file(nudge_file)['nudge']   
        return self._func(conditioning_in=conditioning, conditioning_nudge=c1, **kwargs)

'''
class ImageBasedNudgeConditioning(BaseNudge):
    @classmethod
    def INPUT_TYPES(cls):
        it = {"required": { "clip": ("CLIP", ), "images": ("IMAGE", ) } }
        return cls.ADD_STANDARD_INPUT_TYPES(it)

    def func(self, conditioning, clip, images, **kwargs):
        for image in images:
            c1 = clipify(image)
            conditioning = self._func(conditioning_in=conditioning, conditioning_nudge=c1, **kwargs)[0]
        return (conditioning,)
'''