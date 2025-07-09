# WeGeFT: Generative Interpretable Fine-Tuning 
[Chinmay Savadikar](https://savadikarc.github.io)<sup>1</sup>, Xi Song<sup>2</sup>, [Tianfu Wu](https://ece.ncsu.edu/people/twu19/)<sup>1</sup><br>
<sup>1</sup>North Carolina State University, <sup>2</sup>An Independent Researcher<br>
[[**Paper**](https://arxiv.org/abs/2312.00700)] | [[**Website**](https://savadikarc.github.io/wegeft)]

<p align="center">
<img src="acc-vs-params.jpg" width="70%" height="100%" class="center">
</p>
<p align="center">
<img src="clusters.jpg" width="70%" height="100%" class="center">
</p>

# Installation
The code in this repository has been tested with Python 3.9, but should be compatible with Python >= 3.8. We have tested the code with Pytorch 2.1.0. WeGeFT can be installed as a standalone package without the dependencies required for the experiments. To install WeGeFT, first install PyTorch
```sh
conda create -n wegeft python=3.9.19
conda activate wegeft
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# or, if you have CUDA 11.1
# conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
Once PyTorch is installed, install WeGeFT by running the following commands:
```sh
git clone https://github.com/savadikarc/wegeft.git
cd wegeft
pip install -e .
```

The directory [wegeft_experiment_utils/wegeft.visual_classification_utils](wegeft_experiment_utils/wegeft.visual_classification_utils) contains the utility functions required for the visual classification experiments. The scripts to run the our experiments do not require an installation of the utility code as a package (but does need the user to install additional dependencies). To replicate the exact environment used in our experiments, please see the instructions in [experiment_setup/SETUP.md](experiment_setup/SETUP.md). To run the experiments, please jump [here](#experiments-and-setup).

# Applying WeGeFT to any Transformer backbone

### Causal Language Modeling using [transformers](https://huggingface.co/transformers/) (example using Llama 3)
```python
from transformers import AutoModelForCausalLM
from wegeft.wegeft import WeGeFTConfig, WeGeFTWrapperForCausalLM

# Define the pretrained backbone
dtype = torch.bfloat16 # or torch.float32, if bfloat16 is not supported
backbone = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    torch_dtype=dtype,
    device_map=<DEVICE>
)

# Define the WeGeFT configuration
wegeft_config = WeGeFTConfig(
    rank=64, # Rank of WeGeFT
    dtype=dtype, # dtype for WeGeFT parameters and residual generation
    wegeft_paramters=dict( # WeGeFT schema. By default, WeGeFT uses two simple linear projections
        block_type='simple_block', # denoted by 'simple_block'
        act_layer="identity" # with no non-linerarity in between
    )
    in_projection_bias=False, # and no biases in the first projection (phi)
    out_projection_bias=False, # and no biases in the second projection (psi)
    target_modules=["q_proj", "v_proj"] # target modules for WeGeFT
)

# Wrap the backbone with WeGeFT
model = WeGeFTWrapperForCausalLM(wegeft_config, backbone)

# Now, you can train the model as you would normally do
# ...
# ...
```

### Vision Transformer (ViT) backbone using [timm](https://github.com/huggingface/pytorch-image-models/tree/main) for image classification
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

### RoBERTa backbone using [transformers](https://huggingface.co/transformers/) for Sequence Classification
```python
from transformers import AutoConfig, AutoModelForSequenceClassification
from wegeft.wegeft import WeGeFTConfig, WeGeFTWrapperForSeqClassification

# Define the pretrained backbone
config = AutoConfig.from_pretrained(
    "FacebookAI/roberta-base",
    num_labels=<NUM_LABELS>,
)
backbone = AutoModelForSequenceClassification.from_pretrained(
    "FacebookAI/roberta-base",
    config=config,
)

# Define the WeGeFT configuration
wegeft_config = WeGeFTConfig(
    rank=32,
    dtype='float32',
    wegeft_paramters=dict(
        block_type='simple_block',
        act_layer="identity"
    )
    in_projection_bias=False,
    out_projection_bias=False,
    target_modules=["query", "value"]
)

# Wrap the backbone with WeGeFT
model = WeGeFTWrapperForSeqClassification(wegeft_config, backbone)

# Now, you can train the model as you would normally do
# ...
# ...
```

# Environment Setup
Please follow the instructions in [experiment_setup/SETUP.md](experiment_setup/SETUP.md) to setup the environment.

# Experiments and Setup
## Language Modeling
Please refer to [scripts/language_modeling/README.md](scripts/language_modeling/README.md) for details on data preparation and training for commonsense reasoning, arithmetic reasoning and instruction tuning experiments.

## Visual Recognition
### FGVC
Please refer to [scripts/visual_classification/fgvc/README.md](scripts/visual_classification/fgvc/README.md) for details on data preparation and training.

### VTAB
Please refer to [scripts/visual_classification/vtab/README.md](scripts/visual_classification/vtab/README.md) for details on data preparation and training.

# Acknowledgements
This code is based on code from [timm](https://github.com/huggingface/pytorch-image-models/tree/main), [TOAST](https://github.com/bfshi/TOAST), and [pyreft](https://github.com/stanfordnlp/pyreft). We thank the authors for their amazing work.

# Citation
```bibtex
@misc{savadikar2024wegeft,
    title={WeGeFT: Generative Interpretable Fine-Tuning}, 
    author={Chinmay Savadikar and Xi Song and Tianfu Wu},
    year={2024},
    eprint={2312.00700},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
