from .base import VitExtractor


class FaRLVitExtractor(VitExtractor):
    def __init__(self, model_name='farl_ViT-B/16', device='cuda',
                 pretrained='./checkpoints/FaRL-Base-Patch16-LAIONFace20M-ep64.pth'):
        model_name = model_name.split('_')[1]
        super().__init__(device=device, pretrained=pretrained, model_name=model_name)
