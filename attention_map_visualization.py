from argparse import ArgumentParser
from PIL import Image
import tqdm
import os
import random
import colorsys
import skimage
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from matplotlib.patches import Polygon
import cv2.cv2 as cv2
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from extractors import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = ArgumentParser()
parser.add_argument("--image_path", type=str, default='./data/aligned/target10.png')
parser.add_argument("--layer", type=int, default=11,
                    help='Transformer layer from which to extract the feature, between 0-11')
parser.add_argument("--model_name", type=str, default='dino_vitb16',
                    help='options for DINO: dino_vitb8 | dino_vits8 | dino_vitb16 | dino_vits16, '
                         'options for FaRL: farl_ViT-B/16, '
                         'options for CLIP: clip_ViT-B/16 | clip_ViT-B/32 | clip_ViT-L/14 | clip_ViT-L/14@336px')
parser.add_argument('--pretrained', type=str, default='./checkpoints/FaRL-Base-Patch16-LAIONFace20M-ep64.pth',
                    help='if use FaRL, pretrained should be provided which is the checkpoints file of FaRL.')
parser.add_argument("--save_path", type=str, default='./outputs')
parser.add_argument("--threshold", type=float, default=None, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
parser.add_argument('--output_dir', type=str, default='./outputs/attentions')
args = parser.parse_args()


def sformat(string: str):
    string = string.lower()
    model_name = string[:4]
    model_type = string.split('-')[1].replace('/', '') if '-' in string else string.split('vit')[1]
    return model_name+'_'+model_type


save_path = os.path.join(args.save_path, f'{sformat(args.model_name)}_attention_'+os.path.basename(args.image_path))
cmap = plt.get_cmap('RdPu')
preprocess = transforms.Compose([
    transforms.Resize(224),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
# define the extractor
extractor = {'dino': DINOVitExtractor, 'farl': FaRLVitExtractor, 'clip': CLIPVitExtractor}
vit_extractor = extractor[args.model_name[:4]](model_name=args.model_name, device=device, pretrained=args.pretrained)


def visualize_mean_attention(args, layer):
    input_img = Image.open(args.image_path).convert('RGB')
    input_img = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])(input_img).unsqueeze(0).to(device)

    with torch.no_grad():
        try:
            input_img_ = preprocess(input_img)
            attentions, attention = vit_extractor.get_attentions_from_input(input_img_, layer)
        except:
            input_img_ = preprocess(input_img).half()
            attentions, attention = vit_extractor.get_attentions_from_input(input_img_, layer)

        patch_h_num = vit_extractor.get_height_patch_num(input_img.shape)
        patch_w_num = vit_extractor.get_width_patch_num(input_img.shape)
        patch_size = vit_extractor.get_patch_size()
        attention = attention[0, 0, 1:].reshape(-1)
        attention = attention.reshape(patch_h_num, patch_w_num)
        attention = F.interpolate(attention.unsqueeze(0).unsqueeze(0),
                                  scale_factor=patch_size, mode="bilinear")[0].cpu().numpy().squeeze()

        mask = torch.tensor(attention / attention.max()).unsqueeze(0).repeat(3, 1, 1)
        attention = (attention - attention.min()) / (attention.max() - attention.min())

        input_img = input_img.cpu().squeeze()
        input_img_mask = input_img * mask

        attention_heatmap = transforms.ToTensor()(Image.fromarray((cmap(attention)[:, :, :3]*255).astype(np.uint8)))

    return input_img_mask, attention_heatmap


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return


def visualize_attentions(args, layer):
    input_img = Image.open(args.image_path).convert('RGB')
    input_img = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()
    ])(input_img).unsqueeze(0).to(device)

    with torch.no_grad():
        try:
            input_img_ = preprocess(input_img)
            attentions, attention = vit_extractor.get_attentions_from_input(input_img_, layer)
        except:
            input_img_ = preprocess(input_img).half()
            attentions, attention = vit_extractor.get_attentions_from_input(input_img_, layer)

        nh = attentions.shape[1]  # number of head
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
        patch_h_num = vit_extractor.get_height_patch_num(input_img.shape)
        patch_w_num = vit_extractor.get_width_patch_num(input_img.shape)
        patch_size = vit_extractor.get_patch_size()
        if args.threshold is not None:
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - args.threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, patch_w_num, patch_h_num).float()
            # interpolate
            th_attn = F.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

        attentions = attentions.reshape(nh, patch_w_num, patch_h_num)
        attentions = F.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()

        # save attentions heatmaps
        os.makedirs(args.output_dir, exist_ok=True)
        torchvision.utils.save_image(torchvision.utils.make_grid(input_img, normalize=True, scale_each=True), os.path.join(args.output_dir, "img.png"))
        for j in range(nh):
            fname = os.path.join(args.output_dir, "attn-head" + str(j) + ".png")
            plt.imsave(fname=fname, arr=attentions[j], format='png')
            print(f"{fname} saved.")

        if args.threshold is not None:
            image = skimage.io.imread(os.path.join(args.output_dir, "img.png"))
            for j in range(nh):
                display_instances(image, th_attn[j], fname=os.path.join(args.output_dir, "mask_th" + str(args.threshold) + "_head" + str(j) +".png"), blur=False)


if __name__ == '__main__':
    images_with_mask = []
    attention_heatmaps = []
    for layer in tqdm.tqdm(range(12)):
        img_mask, heatmap = visualize_mean_attention(args, layer)
        images_with_mask.append(img_mask)
        attention_heatmaps.append(heatmap)
    results = torch.stack(images_with_mask+attention_heatmaps, dim=0)
    torchvision.utils.save_image(results, save_path, nrow=len(attention_heatmaps))
    print('result saved to', save_path)

    # for layer in range(12):
    #     visualize_attentions(args, layer)
