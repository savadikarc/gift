# VTAB

# Data preparation
Please refer to [DATA.md](DATA.md) for details on data preparation.

# Hyperparameter tuning
- Run the hyperparameter search script to find the best hyperparameters:
```sh
# For GIFT
./tune_vtab_gift.sh <DATASET> <GPU_ID> vit_base_patch16_224.augreg_in21k [--log-wandb --wandb-project <PROJECT_NAME>] [--artifact_dir <ARTIFACT_DIR>]
# For LoRA
./tune_vtab_lora.sh <DATASET> <GPU_ID> vit_base_patch16_224.augreg_in21k [--log-wandb --wandb-project <PROJECT_NAME>] [--artifact_dir <ARTIFACT_DIR>]
# For BitFit
./tune_vtab_bitfit.sh <DATASET> <GPU_ID> vit_base_patch16_224.augreg_in21k [--log-wandb --wandb-project <PROJECT_NAME>] [--artifact_dir <ARTIFACT_DIR>]
# For VPT
./tune_vtab_vpt.sh <DATASET> <GPU_ID> vit_base_patch16_224.augreg_in21k [--log-wandb --wandb-project <PROJECT_NAME>] [--artifact_dir <ARTIFACT_DIR>]
```

# Training
```sh
# Note that the --evaluate flag is used to evaluate the model on the test set
./train_vtab.sh <METHOD> <DATASET> <GPU_ID> vit_base_patch16_224.augreg_in21k --lr <LR> --weight-decay <WEIGHT_DECAY> --evaluate [--experiment <EXP_NAME>] [--log-wandb --wandb-project <PROJECT_NAME>] [--artifact_dir <ARTIFACT_DIR>]
```

# Notes
- ```<DATASET>``` can take the following values (including quotation marks):
  - ```"vtab-cifar(num_classes=100)"```
  - ```"vtab-caltech101"```
  - ```"vtab-clevr(task=\"closest_object_distance\")"```
  - ```"vtab-clevr(task=\"count_all\")"```
  - ```"vtab-svhn"```
  - ```"vtab-patch_camelyon"```
  - ```"vtab-dmlab"```
  - ```"vtab-diabetic_retinopathy(config=\"btgraham-300\")"```
  - ```"vtab-smallnorb(predicted_attribute=\"label_elevation\")"```
  - ```"vtab-smallnorb(predicted_attribute=\"label_azimuth\")"```
  - ```"vtab-sun397"```
  - ```"vtab-dtd"```
  - ```"vtab-resisc45"```
  - ```"vtab-eurosat"```
  - ```"vtab-oxford_iiit_pet"```
  - ```"vtab-oxford_flowers102"```
  - ```"vtab-dsprites(predicted_attribute=\"label_orientation\",num_classes=16)"```
  - ```"vtab-dsprites(predicted_attribute=\"label_x_position\",num_classes=16)"```
  - ```"vtab-kitti(task=\"closest_vehicle_distance\")"```
- ```<METHOD>``` can take values `gift`, `lora`, `bitfit` or `vpt`.
- ```--experiment <EXP_NAME>``` can be used to name the experiment. If not provided, a name will be generated automatically based on the method, evaluation flag, learning rate, weight decay, and downsample ratio/rank.
- `<GPU_ID>` is the numeric ID of the GPU to be used (e.g., for cuda:0, use `<GPU_ID>` as `0`)

## Default configurations
The default configurations are given in `configs/training`, and can be overridden by passing the desired configuration as a command line argument. For example, to override the default rank from 16 to 32 for GIFT, pass `--gift_rank 32` to the training script. For e.g.,
```sh
./train_vtab.sh <METHOD> <DATASET> <GPU_ID> vit_base_patch16_224.augreg_in21k --lr <LR> --weight-decay <WEIGHT_DECAY> --evaluate [--experiment <EXP_NAME>] [--log-wandb --wandb-project <PROJECT_NAME>] [--artifact_dir <ARTIFACT_DIR>] --gift_rank 32
```
The same can be done for tuning scripts as well.
