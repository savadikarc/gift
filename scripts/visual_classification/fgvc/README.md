# FGVC

# Data Preparation
Please refer to [DATA.md](DATA.md) for details on data preparation.

# Hyperparameter tuning
- Run the hyperparameter search script to find the best hyperparameters:
```sh
# For GIFT
./tune_fgvc_gift.sh <DATASET> <GPU_ID> vit_base_patch16_224.augreg_in21k [--log-wandb --wandb-project <PROJECT_NAME>] [--artifact_dir <ARTIFACT_DIR>]
# For LoRA
./tune_fgvc_lora.sh <DATASET> <GPU_ID> vit_base_patch16_224.augreg_in21k [--log-wandb --wandb-project <PROJECT_NAME>] [--artifact_dir <ARTIFACT_DIR>]
# For BitFit
./tune_fgvc_bitfit.sh <DATASET> <GPU_ID> vit_base_patch16_224.augreg_in21k [--log-wandb --wandb-project <PROJECT_NAME>] [--artifact_dir <ARTIFACT_DIR>]
# For VPT
./tune_fgvc_vpt.sh <DATASET> <GPU_ID> vit_base_patch16_224.augreg_in21k [--log-wandb --wandb-project <PROJECT_NAME>] [--artifact_dir <ARTIFACT_DIR>]
```

# Training
```sh
# Note that the --evaluate flag is used to evaluate the model on the test set
./train_fgvc.sh <METHOD> <DATASET> <GPU_ID> vit_base_patch16_224.augreg_in21k --lr <LR> --weight-decay <WEIGHT_DECAY> --evaluate [--experiment <EXP_NAME>] [--log-wandb --wandb-project <PROJECT_NAME>] [--artifact_dir <ARTIFACT_DIR>]
```

# Notes
- ```<DATASET>``` can take values in ```CUB```, ```StanfordCars```, ```StanfordDogs```, ```OxfordFlowers```, or ```nabirds```
- ```<METHOD>``` can take values `gift`, `lora`, `bitfit` or `vpt`.
- ```--experiment <EXP_NAME>``` can be used to name the experiment. If not provided, a name will be generated automatically based on the method, evaluation flag, learning rate, weight decay, and downsample ratio/rank.
- `<GPU_ID>` is the numeric ID of the GPU to be used (e.g., for cuda:0, use `<GPU_ID>` as `0`)

## Default configurations
The default configurations are given in `configs/training`, and can be overridden by passing the desired configuration as a command line argument. For example, to override the default rank from 16 to 32 for GIFT, pass `--gift_rank 32` to the training script. For e.g.,
```sh
./train_fgvc.sh <METHOD> <DATASET> <GPU_ID> vit_base_patch16_224.augreg_in21k --lr <LR> --weight-decay <WEIGHT_DECAY> --evaluate [--experiment <EXP_NAME>] [--log-wandb --wandb-project <PROJECT_NAME>] [--artifact_dir <ARTIFACT_DIR>] --gift_rank 32
```
The same can be done for tuning scripts as well.

## Cluster Visualization
- To visualize the clusters on FGVC, run the following command:
  ```sh
  ./visualize.sh gift <DATASET> <EXP_NAME> <GPU_ID> vit_base_patch16_224.augreg_in21k --evaluate --class_to_visualize 0 --gift_block_block_type simple_block [--artifact_dir <ARTIFACT_DIR>]
  ```
  ```<DATASET>``` can take values in ```CUB```, ```StanfordCars```, ```StanfordDogs```, ```OxfordFlowers```, or ```nabirds```. ```<EXP_NAME>``` is the name of the experment generated after the training is complete. `<EXP>` is the name of the experiment (specified or generated) after the training is complete for the `<DATASET>`.
- For example, to visualize the clusters in the paper, run the following command:
  ```sh
  # Figure 2 (Check layers and clusters by subtracting 1 from those mentioned in the paper, since the figures consider
  # zero indexing.)
  ./visualize.sh gift nabirds 0 vit_base_patch16_224.augreg_in21k --evaluate --class_to_visualize 47
  # Figure 4 (Appendix)
  ./visualize.sh gift nabirds 0 vit_base_patch16_224.augreg_in21k --evaluate --class_to_visualize 333
  ./visualize.sh gift nabirds 0 vit_base_patch16_224.augreg_in21k --evaluate --class_to_visualize 49
  ```

## Validation
To validate the models on the test sets, run the following command:
```sh
./validate.sh <METHOD> <DATASET> <GPU_ID> <EXP_NAME> vit_base_patch16_224.augreg_in21k --evaluate [--artifact_dir <ARTIFACT_DIR>]
```
