# VTAB Preperation
This file has been taken and subsequently adapted from [https://github.com/bfshi/TOAST/blob/main/visual_classification/VTAB_SETUP.md](https://github.com/bfshi/TOAST/blob/main/visual_classification/VTAB_SETUP.md). We thank the authors of TOAST for their work.

The code to download and prepare the data has been provided in ```download_vtab.py```. Uncomment the relevant sections to download the datasets you need. The script will download the datasets to the ```data_dir``` specified in the script. We use the splits provided by tfds.

## Special Care
These datasets need special care:

### Kitti
```python
# kitti version is wrong from vtab repo, try 3.2.0 (https://github.com/google-research/task_adaptation/issues/18)
dataset_builder = tfds.builder("kitti:3.2.0", data_dir=data_dir)
dataset_builder.download_and_prepare()
```

# Diabetic Retinopathy
```python
"""
Download this dataset from Kaggle.
https://www.kaggle.com/c/diabetic-retinopathy-detection/data
After downloading, 
- unpack the test.zip file into <data_dir>/manual_dir/.
- unpack the sample.zip to sample/. 
- unpack the sampleSubmissions.csv and trainLabels.csv.

# ==== important! ====
# 1. make sure to check that there are 5 train.zip files instead of 4 (somehow if you chose to download all from kaggle, the train.zip.005 file is missing)
# 2. if unzip train.zip ran into issues, try to use jar xvf train.zip to handle huge zip file
cat test.zip.* > test.zip
cat train.zip.* > train.zip
"""

config_and_version = "btgraham-300" + ":3.*.*"
dataset_builder = tfds.builder("diabetic_retinopathy_detection/{}".format(config_and_version), data_dir=data_dir)
dataset_builder.download_and_prepare()
```

### Resisc45
```python
"""
download/extract dataset artifacts manually: 
Dataset can be downloaded from OneDrive: https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs
After downloading the rar file, please extract it to the manual_dir.
"""

dataset_builder = tfds.builder("resisc45:3.*.*", data_dir=data_dir)
dataset_builder.download_and_prepare()
```
### Caltech101
- ```tensorflow-datasets==4.9.3``` should be used instead of ```tfds-nightly==4.4.0.dev202201080107``` as used by the other datasets. Please follow the instructions in (environment_setup/SETUP.md)[environment_setup/SETUP.md] to create the environment.
- The download for `caltech101` might give an error for failed checksum. If this persists, we will upload of tfrecord files compatible with the environment to a public location and provide a link here.

## Notes

### TFDS version
Note that the experimental results may be different with different API and/or dataset generation code versions. See more from [tfds documentation](https://www.tensorflow.org/datasets/datasets_versioning). We use the following versions:

```bash

# Natural:
cifar100: 3.0.2
caltech101: 3.0.1
dtd: 3.0.1
oxford_flowers102: 2.1.1
oxford_iiit_pet: 3.2.0
svhn_cropped: 3.0.0
sun397: 4.0.0

# Specialized:
patch_camelyon: 2.0.0
eurosat: 2.0.0
resisc45: 3.0.0
diabetic_retinopathy_detection: 3.0.0


# Structured
clevr: 3.1.0
dmlab: 2.0.1
kitti: 3.2.0
dsprites: 2.0.0
smallnorb: 2.0.0
```
