# Natural Language Unerstanding on GLUE
This directory contains the scripts to finetune RoBERTa-Base and RoBERTa-Large on the GLUE benchmark: [General Language Understanding Evaluation](https://gluebenchmark.com/). Based on the script [`run_glue_no_trainer.py`](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue_no_trainer.py).

The following scripts will run the hyperparameter search (with seed 42) and final run with 5 seeds (42, 43, 44, 45, 46). GLUE dataset will downloaded automatically using HuggingFace datasets library.
```sh
# RoBERTa-Base
./run_glue_base.sh <TASK_NAME> <GPU_ID> <EPOCHS> --gift_rank 32
# RoBERTa-Large
./run_glue_large.sh <TASK_NAME> <GPU_ID> <EPOCHS> --gift_rank 32
```
- ```<TASK_NAME>``` can take values ```cola```, ```mrpc```, ```qnli```, ```rte```, ```sst2``` or ```stsb```
- The number of training epochs for each task is given in Table 9 in the Appendix.
- `<GPU_ID>` is the numeric ID of the GPU to be used (e.g., for cuda:0, use `<GPU_ID>` as `0`)
