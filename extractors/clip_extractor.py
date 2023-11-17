from .base import VitExtractor


class CLIPVitExtractor(VitExtractor):
    def __init__(self, model_name='clip_ViT-B/16', device='cuda', pretrained=None):
        model_name = model_name.split('_')[1]
        super().__init__(model_name=model_name, device=device, pretrained=None)
