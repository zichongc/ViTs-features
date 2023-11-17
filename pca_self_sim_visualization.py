from argparse import ArgumentParser
import PIL
from PIL import Image
import tqdm
import os
import numpy as np
import torch
import torchvision.utils
from torchvision import transforms as T
from sklearn.decomposition import PCA
from extractors import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = ArgumentParser()
parser.add_argument("--image_path", type=str, default='./data/aligned/target8.png')
parser.add_argument("--layer", type=int, default=11,
                    help='Transformer layer from which to extract the feature, between 0-11')
parser.add_argument("--model_name", type=str, default='dino_vitb8',
                    help='options for DINO: dino_vitb8 | dino_vits8 | dino_vitb16 | dino_vits16, '
                         'options for FaRL: farl_ViT-B/16, '
                         'options for CLIP: clip_ViT-B/16 | clip_ViT-B/32 | clip_ViT-L/14 | clip_ViT-L/14@336px')
parser.add_argument('--pretrained', type=str, default='./checkpoints/FaRL-Base-Patch16-LAIONFace20M-ep64.pth',
                    help='if use FaRL, pretrained should be provided which is the checkpoints file of FaRL.')
parser.add_argument("--save_path", type=str, default='./outputs')
args = parser.parse_args()


def sformat(string: str):
    string = string.lower()
    model_name = string[:4]
    model_type = string.split('-')[1].replace('/', '') if '-' in string else string.split('vit')[1]
    return model_name+'_'+model_type


save_path = os.path.join(args.save_path, f'{sformat(args.model_name)}_ssim_'+os.path.basename(args.image_path))
preprocess = T.Compose([
    T.Resize(224),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
# define the extractor
extractor = {'dino': DINOVitExtractor, 'farl': FaRLVitExtractor, 'clip': CLIPVitExtractor}
vit_extractor = extractor[args.model_name[:4]](model_name=args.model_name, device=device, pretrained=args.pretrained)


def visualize(args, m, layer):
    assert m in ['k', 'q', 'v', 'f']
    # load the image
    input_img = Image.open(args.image_path).convert('RGB')
    input_img = T.Compose([
        T.Resize(224),
        T.ToTensor()
    ])(input_img).unsqueeze(0).to(device)

    # calculate self-sim
    ssim = {
        'k': vit_extractor.get_keys_self_sim_from_input,
        'q': vit_extractor.get_queries_self_sim_from_input,
        'v': vit_extractor.get_values_self_sim_from_input,
        'f': vit_extractor.get_features_self_sim_from_input
    }

    with torch.no_grad():
        try:
            input_img_ = preprocess(input_img)
            self_sim = ssim[m](input_img_, layer)
            cross_sim = vit_extractor.get_keys_cross_sim_from_input(input_img_, input_img_, layer)
        except:
            input_img_ = preprocess(input_img).half()
            self_sim = ssim[m](input_img_, layer)

    pca = PCA(n_components=3)
    pca.fit(self_sim[0].cpu().numpy())
    components = pca.transform(self_sim[0].cpu().numpy())

    patch_h_num = vit_extractor.get_height_patch_num(input_img.shape)
    patch_w_num = vit_extractor.get_width_patch_num(input_img.shape)
    components = components[1:, :]
    components = components.reshape(patch_h_num, patch_w_num, 3)
    comp = components
    comp_min = comp.min(axis=(0, 1))
    comp_max = comp.max(axis=(0, 1))
    comp_img = (comp - comp_min) / (comp_max - comp_min)
    pca_pil = Image.fromarray((comp_img * 255).astype(np.uint8))
    pca_pil = pca_pil.resize((224, 224), resample=PIL.Image.BILINEAR)
    pca_image = T.ToTensor()(pca_pil)

    return pca_image


if __name__ == '__main__':
    pca_images = []
    for mode in ['k', 'q', 'v', 'f']:
        for layer in tqdm.tqdm(range(12)):
            pca_images.append(visualize(args, mode, layer))
    pca_images = torch.stack(pca_images, dim=0)
    torchvision.utils.save_image(pca_images, save_path, nrow=12)
    print(save_path)
