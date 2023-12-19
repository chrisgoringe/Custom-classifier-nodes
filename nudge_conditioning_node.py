from safetensors.torch import load_file
import torch

class NudgeConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {   "conditioning": ("CONDITIONING", ), 
                                "nudge_file": ("STRING", {"default":""}),
                                "nudge_strength": ("FLOAT", {"default": 0.1, "step": 0.01}),
                             }}
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "func"

    CATEGORY = "conditioning"

    def func(self, conditioning, nudge_file:str, nudge_strength:float):
        # T = tokens = 77
        # L = length = 2048
        # B = Batch
        # F = Features = 768
        method1=False
        c1:torch.Tensor = load_file(nudge_file)['nudge']
        F = c1.shape[0]
        c1 = c1.reshape((1,1,F)) 
        c1 = torch.mul(c1,nudge_strength)

        if len(conditioning)>1: print("Warning - conditionings after the first are being lost")
        
        c0:torch.Tensor = conditioning[0][0]                            
        B, T, L = c0.shape
        c1 = torch.cat( [c1.repeat((B,T,1))] + [torch.zeros( (B,T,(L-F)) )], dim=2)
        return( [[c1 + c0,conditioning[0][1].copy()],], )


