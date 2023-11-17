import os
import clip
import math
import torch
import torch.nn.functional as F


def attn_cosine_sim(x, eps=1e-08):
    x = x[0]  # TEMP: getting rid of redundant dimension, TBF
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix


def cross_cos_sim(x, y, eps=1e-08):
    x, y = x[0], y[0]
    norm_x = x.norm(dim=2, keepdim=True)
    norm_y = y.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm_x @ norm_y.permute(0, 2, 1), min=eps)
    cross_sim_matrix = (x @ y.permute(0, 2, 1)) / factor
    return cross_sim_matrix


class VitExtractor:
    BLOCK_KEY = 'block'
    ATTN = 'attn'
    KEY_LIST = [BLOCK_KEY, ATTN]

    def __init__(self, model_name='ViT-B/16', device='cuda', pretrained=None):
        model, preprocess = clip.load(model_name, device=device)
        model = model.to(device).eval()
        if pretrained is not None:
            assert os.path.exists(pretrained)
            ckpt = torch.load(pretrained)
            model.load_state_dict(ckpt['state_dict'], strict=False)
        self.model = model.visual
        self.num_heads = 12
        self.embed_dim = 768
        self.patch_size = int(model_name.split('/')[1])
        self.head_dim = self.embed_dim // self.num_heads
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self._init_hooks_data()

    def _init_hooks_data(self):
        self.layers_dict[VitExtractor.BLOCK_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.ATTN] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for key in VitExtractor.KEY_LIST:
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
        for block_idx, block in enumerate(self.model.transformer.resblocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:       # token
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            if block_idx in self.layers_dict[VitExtractor.ATTN]:        # attention (includes att)
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_attn_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _get_block_output(model, inp, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_model(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN].append([model, inp])

        return _get_attn_model

    def get_feature_from_input(self, input_img):  # List([B, N, D])
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()

        return feature

    def get_qkv_attn_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        features = []
        attention_module = self.outputs_dict[VitExtractor.ATTN]
        for attention in attention_module:
            tgt_len, bsz, C = attention[1][0].shape
            assert attention[1][0] is attention[1][1] and attention[1][1] is attention[1][2]
            weight, bias = attention[0].in_proj_weight, attention[0].in_proj_bias
            q, k, v = F.linear(attention[1][0], weight, bias).chunk(3, dim=-1)
            q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            k = k.contiguous().view(k.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
            v = v.contiguous().view(v.shape[0], bsz * self.num_heads, self.head_dim).transpose(0, 1)
            src_len = k.size(1)

            B, Nt, E = q.shape
            q_scaled = q / math.sqrt(E)
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
            attn_output_weights = F.softmax(attn_output_weights, dim=-1)
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            features.append({'q': q, 'k': k, 'v': v, 'attn': attn_output_weights})

        self._clear_hooks()
        self._init_hooks_data()

        return features

    def get_patch_size(self):
        return self.patch_size

    def get_width_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return w // patch_size

    def get_height_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return h // patch_size

    def get_patch_num(self, input_img_shape):
        patch_num = 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))
        return patch_num

    def get_head_num(self):
        return self.num_heads

    def get_embedding_dim(self):
        return self.embed_dim

    def get_keys_from_input(self, input_img, layer_num):
        qkv_attn = self.get_qkv_attn_feature_from_input(input_img)[layer_num]
        keys = qkv_attn['k']
        return keys

    def get_values_from_input(self, input_img, layer_num):
        qkv_attn = self.get_qkv_attn_feature_from_input(input_img)[layer_num]
        values = qkv_attn['v']
        return values

    def get_queries_from_input(self, input_img, layer_num):
        qkv_attn = self.get_qkv_attn_feature_from_input(input_img)[layer_num]
        queries = qkv_attn['q']
        return queries

    def get_attentions_from_input(self, input_img, layer_num, return_mean=True):
        qkv_attn = self.get_qkv_attn_feature_from_input(input_img)[layer_num]
        attn = qkv_attn['attn']
        attn_mean = attn.sum(dim=1) / self.num_heads
        return attn, attn_mean if return_mean else None

    def get_features_from_input(self, input_img, layer_num):
        features = self.get_feature_from_input(input_img)[layer_num]
        features_t = features.transpose(0, 1)
        return features_t

    def get_keys_self_sim_from_input(self, input_img, layer_num):
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_keys[None, None, ...])
        return ssim_map

    def get_values_self_sim_from_input(self, input_img, layer_num):
        values = self.get_values_from_input(input_img, layer_num=layer_num)
        h, t, d = values.shape
        concatenated_values = values.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_values[None, None, ...])
        return ssim_map

    def get_queries_self_sim_from_input(self, input_img, layer_num):
        queries = self.get_queries_from_input(input_img, layer_num=layer_num)
        h, t, d = queries.shape
        concatenated_values = queries.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_values[None, None, ...])
        return ssim_map

    def get_features_self_sim_from_input(self, input_img, layer_num):
        features = self.get_features_from_input(input_img, layer_num=layer_num)
        h, t, d = features.shape
        concatenated_values = features.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_values[None, None, ...])
        return ssim_map

    def get_keys_cross_sim_from_input(self, source_img, target_img, layer_num):
        src_keys = self.get_keys_from_input(source_img, layer_num=layer_num)
        tgt_keys = self.get_keys_from_input(target_img, layer_num=layer_num)
        assert src_keys.shape == tgt_keys.shape
        h, t, d = src_keys.shape
        concatenated_src_keys = src_keys.transpose(0, 1).reshape(t, h*d)
        concatenated_tgt_keys = tgt_keys.transpose(0, 1).reshape(t, h*d)
        cross_sim_map = cross_cos_sim(concatenated_src_keys[None, None, ...], concatenated_tgt_keys[None, None, ...])
        return cross_sim_map
