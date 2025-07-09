import torch
from pathlib import Path

import numpy as np
import math
import cv2

from .wegeft import WeGeFTWrapperForImageClassification, WeGeFTConfig
from .wegeft.modeling_wegeft import check_target_module_exists, _get_submodules, WeGeFTLinear, WeGeFTMergedLinear


def normalize(x: torch.Tensor, axis=None, thr=0.5):
    # x is the norm of of activations for each token
    keepdims = axis is not None
    xmin = x.amin(dim=axis, keepdim=keepdims)
    xmax = x.amax(dim=axis, keepdim=keepdims)
    x = (x - xmin) / (xmax - xmin)
    x = (x < thr) * 0 + (x > thr) * x
    return x


def save_image(image, path, rescale=False):
    # Rescale to 0-255 and convert to uint8 if needed
    if rescale:
        image = (image * 255.0).astype(np.uint8)
    # Switch from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(str(path), image)


class ClusterVisualizer:

    def __init__(self, model: WeGeFTWrapperForImageClassification, device, total_visualizations=30, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], wegeft_layer=-1):

        self.model = model
        self.config: WeGeFTConfig = model.config
        self.input_pre_hook_handle = None
        self.attn_post_hook_handles = []
        self.wegeft_hook_handles = []
        self.num_visualizations = 0
        self.total_visualizations = total_visualizations

        self.mean = torch.Tensor(mean).unsqueeze(-1).unsqueeze(-1).to(device)
        self.std = torch.Tensor(std).unsqueeze(-1).unsqueeze(-1).to(device)

        self.num_visualizations = 0
        self.reset()
        # self.register_hooks()

        self.wegeft_layer = wegeft_layer

    def reset(self):
        config = self.config
        self.remove_hooks()
        # print(f"Resetting visualization for {config.target_modules}")
        self.layer_counter = {wegeft_module: 0 for wegeft_module in config.target_modules}
        self.input = None
        self.cluster_outs = {wegeft_module: [None for l in range(self.model.num_layers)] for wegeft_module in config.target_modules}
        self.clustered_attns = {wegeft_module: [None for _ in range(self.model.num_layers)] for wegeft_module in config.target_modules}
        self.clustered_attns_pre_delta = {wegeft_module: [None for _ in range(self.model.num_layers)] for wegeft_module in config.target_modules}

    def register_hooks(self):
        config = self.config
        key_list = [key for key, _ in self.model.backbone.named_modules()]

        self.input_pre_hook_handle = self.model.backbone.register_forward_pre_hook(self.input_hook())
        
        for target_key in config.target_modules:
            _target_key = target_key.replace(":", ".")
            target_modules = [_get_submodules(self.model.backbone, key) for key in key_list if check_target_module_exists(config, _target_key, key)]
            for layer, (parent, target, target_name) in enumerate(target_modules):
                print(f"Registering visualization hook for {target_name} at layer {layer}")
                assert isinstance(target, WeGeFTLinear) or isinstance(target, WeGeFTMergedLinear), f"Target module {target} is not supported."
                self.attn_post_hook_handles.append(target.pre_delta_identity.register_forward_hook(self.attn_post_hook_pre_delta(target_key, layer)))
                self.attn_post_hook_handles.append(target.register_forward_hook(self.attn_post_hook(target_key, layer)))

        for target_key in config.target_modules:
            self.wegeft_hook_handles.append(self.model.in_projections[target_key].register_forward_hook(self.cluster_hook(target_key)))

    def remove_hooks(self):
        if self.input_pre_hook_handle is not None:
            self.input_pre_hook_handle.remove()
        self.input_pre_hook_handle = None

        for handle in self.attn_post_hook_handles:
            handle.remove()
        self.attn_post_hook_handles = []

        for handle in self.wegeft_hook_handles:
            handle.remove()
        self.wegeft_hook_handles = []

    def input_hook(self):

        def hook(module, input):
            _input = input[0][0]
            # Deprocess
            _input = _input * self.std + self.mean
            _input = torch.clip(_input.permute(1, 2, 0), 0., 1.)
            self.input = _input

        return hook
    
    def process_cluster(self, x, cluster, axis=(1, 2)):
        # x: N, d_out
        # cluster: [H],M, d_out
        attn = cluster @ x.T # M, N
        h, w = int(math.sqrt(attn.shape[-1])), int(math.sqrt(attn.shape[-1]))
        attn = attn.reshape(1, *attn.shape[:-1], h, w)
        # Interpolate to input size
        input_h, input_w = self.input.shape[0], self.input.shape[1]
        attn = torch.nn.functional.interpolate(attn, size=(input_h, input_w), mode="bilinear", align_corners=False)[0]
        return normalize(attn, axis=axis)
    
    def attn_post_hook_pre_delta(self, wegeft_module, layer):

        def hook(module, input, output):
            x = output[0][0, 1:, :] if isinstance(output, tuple) else output[0, 1:, :] # N, d_out
            cluster = self.cluster_outs[wegeft_module][layer] # M, d_out
            self.clustered_attns_pre_delta[wegeft_module][layer] = self.process_cluster(x.type(cluster.dtype), cluster)

        return hook

    def attn_post_hook(self, wegeft_module, layer):

        def hook(module, input, output):
            x = output[0][0, 1:, :] if isinstance(output, tuple) else output[0, 1:, :] # N, d_out
            cluster = self.cluster_outs[wegeft_module][layer] # M, d_out
            self.clustered_attns[wegeft_module][layer] = self.process_cluster(x.type(cluster.dtype), cluster)

        return hook
    
    def cluster_hook(self, wegeft_module):

        def hook(module, input, output):
            # print(f"Layer {wegeft_layer}")
            layer_counter = self.layer_counter[wegeft_module]
            self.cluster_outs[wegeft_module][layer_counter] = output.transpose(0, 1) # d_out x d (d acts as M)
            self.layer_counter[wegeft_module] += 1

        return hook
    
    def apply_heatmap(self, input, attn_map, alpha=0.6):
        heatmap = cv2.applyColorMap((attn_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap)[:, :, ::-1]
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
        cam = heatmap * alpha + np.float32(input) * (1 - alpha)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        return cam
    
    def save_visualizations(self, path, cls_name, alpha=0.6, suffix=""):
        config = self.config
        save_path  = Path(path) / f"visualization_{self.num_visualizations}_class_{cls_name}{suffix}"
        save_path.mkdir(parents=True, exist_ok=True)
        # Input
        input = self.input.cpu().numpy()
        save_image(input, save_path / "input.png", rescale=True)

        for wegeft_module in config.target_modules:
            # Attentions
            try:
                clustered_attns = [attn.detach().cpu().numpy() for attn in self.clustered_attns[wegeft_module]]
                clustered_attns_pre_delta = [attn.detach().cpu().numpy() for attn in self.clustered_attns_pre_delta[wegeft_module]]
            except AttributeError as e:
                print(f"Skipping {wegeft_module}")
                raise e
            for layer, (attn_maps, attn_maps_pre_delta) in enumerate(zip(clustered_attns, clustered_attns_pre_delta)):
                num_clusters, _, _ = attn_maps.shape
                for cluster in range(num_clusters):
                    attn_map = attn_maps[cluster]
                    cam = self.apply_heatmap(input, attn_map, alpha=alpha)
                    path = save_path / wegeft_module / "clusters" / f"layer_{layer}"
                    path.mkdir(parents=True, exist_ok=True)
                    save_image(cam, path / f"cluster_{cluster}.png", rescale=True)

                    # Pre delta
                    attn_map_pre_delta = attn_maps_pre_delta[cluster]
                    cam = self.apply_heatmap(input, attn_map_pre_delta, alpha=alpha)
                    path = save_path / wegeft_module / "clusters_pre_delta" / f"layer_{layer}"
                    path.mkdir(parents=True, exist_ok=True)
                    save_image(cam, path / f"cluster_{cluster}.png", rescale=True)
        
        self.num_visualizations += 1
        self.reset()

        if self.num_visualizations > self.total_visualizations:
            self.remove_hooks()
