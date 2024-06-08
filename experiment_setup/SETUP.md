# Environment Setup

We recommend using Anaconda. Run the following commands to create a new conda environment and install the required packages.

```sh
conda create -n gift python=3.9.19
conda activate gift
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
# or, if you have CUDA 11.1
# conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install GIFT
pip install -e .

cd experiment_setup
# Install the dependencies for the Visual Classification experiments
pip install -r requirements_vision.txt
# Install the dependencies for the language modeling experiments
pip install -r requirements_language.txt
```

For ```caltech101``` dataset in the VTAB benchmark, the stable version of Tensorflow Datasets should be used. For this, create a new conda environment replace ```requirements_vision.txt``` with ```requirements_vision_caltech101.txt```.
