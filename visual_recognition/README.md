# Visual Recognition Experiments

## Setup
The code in this repository has been tested with Python 3.9, but should be compatible with Python >= 3.8. We have tested the code with Pytorch 2.1.0. WeGeFT can be installed as a standalone package without the dependencies required for the experiments. To install WeGeFT, first install PyTorch
```sh
conda create -n wegeft-vit python=3.9.19 -y
conda activate wegeft-vit
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
# or, if you have CUDA 11.1
# conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
Once PyTorch is installed, install WeGeFT by running:
```sh
pip install -e .[common]
```
Note: The experiments with Caltech101 require a different tfds version. To install this, create a different environment with the commands above, and use the following command instead
```sh
pip install -e .[caltech]
```

## Applying WeGeFT to ViT backbone using [timm](https://github.com/huggingface/pytorch-image-models/tree/main) for image classification
```python
import timm
from wegeft.wegeft import WeGeFTWrapperForImageClassification, WeGeFTConfig

# Define the pretrained backbone
backbone = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=<NUM_CLASSES>)

# Define the WeGeFT configuration
wegeft_config = WeGeFTConfig(
    rank=16,
    dtype='float32',
    wegeft_paramters=dict(
        block_type='simple_block',
        act_layer="identity"
    )
    in_projection_bias=False,
    out_projection_bias=False,
    target_modules=["attn:proj"] # target modules for WeGeFT. By default, we use the final linear projection in the MHSA layer for ViTs
)
# The notation 'attn:proj' is chosen in order to disambiguate between any other modules named 'proj' in the model.
# The target modules should match the modules in the backbone model.
# Since WeGeFT stores it's modules in a ModuleDictionary, we use ':' to separate the module name from the submodule name instead
# # of the standard '.'. This is because keys in a ModuleDictionary cannot contain '.'. 

# Wrap the backbone with WeGeFT
model = WeGeFTWrapperForImageClassification(wegeft_config, backbone)

# Now, you can train the model as you would normally do
# ...
# ...
```

## Experiments
### FGVC
Please refer to [scripts/visual_classification/fgvc/README.md](scripts/fgvc/README.md) for details on data preparation and training.

### VTAB
Please refer to [scripts/visual_classification/vtab/README.md](scripts/vtab/README.md) for details on data preparation and training.
