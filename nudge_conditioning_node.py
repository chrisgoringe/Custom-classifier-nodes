from safetensors.torch import load_file
import torch

class NudgeConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {   "conditioning": ("CONDITIONING", ), 
                                "nudge_file": ("STRING", {"default":""}),
                                "nudge_strength": ("FLOAT", {"default": 1.0, "step": 0.01}),
                                "replace_token": ("INT",{"default":76, "min":0, "max":76}),
                             }}
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "func"

    CATEGORY = "conditioning"

    def func(self, conditioning, nudge_file:str, nudge_strength:float, replace_token:int):
        if len(conditioning)>1: print("Warning - conditionings after the first are being lost")
        
        c0:torch.Tensor = conditioning[0][0]                            
        L = c0.shape[2]
        c1:torch.Tensor = load_file(nudge_file)['nudge']
        F = c1.shape[0]
        if (L<F): c1 = torch.cat( [c1] + [torch.zeros((L-F)) ])
        replaced = c0[:,replace_token,:]
 
        c0[:,replace_token,:] = torch.mul(c1,nudge_strength) + torch.mul(replaced,1-nudge_strength)
        return( [[c0,conditioning[0][1].copy()],], )


