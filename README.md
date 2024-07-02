# GIFT: Generative Interpretable Fine-Tuning 
[Chinmay Savadikar](https://savadikarc.github.io)<sup>1</sup>, Xi Song<sup>2</sup>, [Tianfu Wu](https://ece.ncsu.edu/people/twu19/)<sup>1</sup><br>
<sup>1</sup>North Carolina State University, <sup>2</sup>An Independent Researcher<br>
[[**Paper**](https://arxiv.org/abs/2312.00700)] | [[**Website**](https://savadikarc.github.io/gift)]

<p align="center">
<img src="teaser.jpg" width="80%" height="100%" class="center">
</p>

# Installation
The code in this repository has been tested with Python 3.9, but should be compatible with Python >= 3.8. We have tested the code with Pytorch 2.1.0. GIFT can be installed as a standalone package without the dependencies required for the experiments. To install GIFT, first install PyTorch
```sh
conda create -n gift python=3.9.19
conda activate gift
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# or, if you have CUDA 11.1
# conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```
Once PyTorch is installed, install GIFT by running the following commands:
```sh
git clone https://github.com/savadikarc/gift.git
cd gift
pip install -e .
```

The directory [gift_experiment_utils/visual_classification_utils](gift_experiment_utils/visual_classification_utils) contains the utility functions required for the visual classification experiments. The scripts to run the our experiments do not require an installation of the utility code as a package (but does need the user to install additional dependencies). To replicate the exact environment used in our experiments, please see the instructions in [experiment_setup/SETUP.md](experiment_setup/SETUP.md). To run the experiments, please jump [here](#experiments-and-setup).

# Applying GIFT to any Transformer backbone

### Causal Language Modeling using [transformers](https://huggingface.co/transformers/) (example using Llama 3)
```python
from transformers import AutoModelForCausalLM
from gift.gift import GIFTConfig, GIFTWrapperForCausalLM

# Define the pretrained backbone
dtype = torch.bfloat16 # or torch.float32, if bfloat16 is not supported
backbone = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    torch_dtype=dtype if dtype != "float8" else None,  # save memory
    load_in_8bit=True if dtype == "float8" else False,
    device_map=<DEVICE>
)

# Define the GIFT configuration
gift_config = GIFTConfig(
    rank=64, # Rank of GIFT
    dtype=dtype, # dtype for GIFT parameters and residual generation
    gift_paramters=dict( # GIFT schema. By default, GIFT uses two simple linear projections
        block_type='simple_block', # denoted by 'simple_block'
        act_layer="identity" # with no non-linerarity in between
    )
    in_projection_bias=False, # and no biases in the first projection (phi)
    out_projection_bias=False, # and no biases in the second projection (psi)
    target_modules=["q_proj", "v_proj"] # target modules for GIFT
)

# Wrap the backbone with GIFT
model = GIFTWrapperForCausalLM(gift_config, backbone)

# Now, you can train the model as you would normally do
# ...
# ...
```

### Vision Transformer (ViT) backbone using [timm](https://github.com/huggingface/pytorch-image-models/tree/main) for image classification
```python
import timm
from gift.gift import GIFTWrapperForImageClassification, GIFTConfig

# Define the pretrained backbone
backbone = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True, num_classes=<NUM_CLASSES>)

# Define the GIFT configuration
gift_config = GIFTConfig(
    rank=16,
    dtype='float32',
    gift_paramters=dict(
        block_type='simple_block',
        act_layer="identity"
    )
    in_projection_bias=False,
    out_projection_bias=False,
    target_modules=["attn:proj"] # target modules for GIFT. By default, we use the final linear projection in the MHSA layer for ViTs
)
# The notation 'attn:proj' is chosen in order to disambiguate between any other modules named 'proj' in the model.
# The target modules should match the modules in the backbone model.
# Since GIFT stores it's modules in a ModuleDictionary, we use ':' to separate the module name from the submodule name instead
# # of the standard '.'. This is because keys in a ModuleDictionary cannot contain '.'. 

# Wrap the backbone with GIFT
model = GIFTWrapperForImageClassification(gift_config, backbone)

# Now, you can train the model as you would normally do
# ...
# ...
```

### RoBERTa backbone using [transformers](https://huggingface.co/transformers/) for Sequence Classification
```python
from transformers import AutoConfig, AutoModelForSequenceClassification
from gift.gift import GIFTConfig, GIFTWrapperForSeqClassification

# Define the pretrained backbone
config = AutoConfig.from_pretrained(
    "FacebookAI/roberta-base",
    num_labels=<NUM_LABELS,
)
backbone = AutoModelForSequenceClassification.from_pretrained(
    "FacebookAI/roberta-base",
    config=config,
)

# Define the GIFT configuration
gift_config = GIFTConfig(
    rank=32,
    dtype='float32',
    gift_paramters=dict(
        block_type='simple_block',
        act_layer="identity"
    )
    in_projection_bias=False,
    out_projection_bias=False,
    target_modules=["query", "value"]
)

# Wrap the backbone with GIFT
model = GIFTWrapperForSeqClassification(gift_config, backbone)

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

## Natural Language Understanding using GLUE
Please refer to [scripts/language_understanding/README.md](scripts/language_understanding/README.md) for details on data preparation and training.

# Acknowledgements
This code is based on code from [timm](https://github.com/huggingface/pytorch-image-models/tree/main), [TOAST](https://github.com/bfshi/TOAST), and [pyreft](https://github.com/stanfordnlp/pyreft). We thank the authors for their amazing work.

# Citation
```bibtex
@misc{savadikar2024gift,
    title={GIFT: Generative Interpretable Fine-Tuning}, 
    author={Chinmay Savadikar and Xi Song and Tianfu Wu},
    year={2024},
    eprint={2312.00700},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
