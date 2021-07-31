# Sanger

This repository implements the proposed framework in the paper Sanger: A Co-Design Framework for Enabling Sparse Attention using Reconfigurable Architecture (MICRO'21)

## Overview

Sanger, a framework that harvests sparsity in the attention mechanism through synergistic hardware and software co-design. The software part prunes the attention matrix into a dynamic structured pattern, and the hardware part features a reconfigurable architecture that exploits such pattern.

## Getting Started

### Requirements

-  For software experiments
   -  CUDA SDK >= 10.1
   -  Python >= 3.7
   -  PyTorch >= 1.7.0
   -  :hugs: Transformers 4.7.0
-  For hardware experiments
   -  JDK 8 or 11
   -  Scala compiler `sbt`. You can download it from [here](https://www.scala-sbt.org/).

### Installation

1.  Clone or download this repository
2.  Download the CLOTH dataset from [here](https://www.cs.cmu.edu/~glai1/data/cloth/) to `data/cloth`
3.  Create a virtual environment (either [virtualenv](https://virtualenv.pypa.io/en/latest/) or [conda](https://docs.anaconda.com/anaconda/install/index.html)) with a Python version of at least 3.7.
4.  Install dependent Python packages: `pip install -r requirements.txt`
5.  Set relevant environment variables
    1.  `export PROJ_ROOT=<path-to-this-repo>`
    2.  `export WANDB_ENABLED=true` to enable [wandb](https://docs.wandb.ai/quickstart) logging (optional)

## Experiment Workflow

### Hardware experiments

1.  Run the tests. 
    1.  `cd` into the `hardware/` directory, run `sbt` and type `test` into the `sbt` console.
2.  Check the result. 
    -  The tests generate random data and the corresponding control signals for the three modules. 
    -  The output of the modules is compared with a directly computed ground truth. 
    -  The relative error should be below a threshold of 5%, which does not affect the final accuracy.

### Software experiments

1.  Evaluate Sanger performance

    1.  Train a model with Sanger sparse attention. 

        We provide scripts for training in the `scripts/` sub-directory. For example, to train a Sanger-pruned BERT-Base model on SQuAD, you can execute `scripts/train_sparse_on_squad.sh`. Note that you have to pass in an appropriate configuration file, which you can find in `configs/`. You can skip this step if you choose to load a fine-tuned checkpoint directly.

    2.  Evaluate the fine-tuned model. 

        We also provide scripts for evaluation in `scripts/`. For example, to evaluate the sparse model from the last step, you can execute `scripts/eval_sparse_on_squad.sh`. If you need to load a checkpoint from a non-standard location, be sure to change the path in the script. When the evaluation is complete, the script should print out the accuracy.

    3.  Measure sparsity and load balance. 

        Each evaluation script contains a flag that enables measuring the sparsity level of attention and calculating the load balance of the PE array. If you set this flag in the previous step, the script will log the results to a CSV file named `load_balance.csv` during evaluation.

    4.  Estimate the hardware performance of Sanger. 

        We implement a simple simulator in `bench_sanger.py` that estimates the latency of executing an attention layer on Sanger, given average sparsity and load balance. Executing this Python script will read the CSV file generated in the previous step, and print the average sparsity, load balance and estimated latency.

2.  Comparison with dense attention and static sparse attention.

    1.  Train a model with dense or static sparse attention. 

        We provide dedicated scripts for train models with dense attention (e.g. `scripts/train_dense_on_squad.sh`). To train a model with static sparse attention, you can use the same script as Sanger and pass in an appropriate configuration file (e.g. `bert_base_longformer.json`).

    2.  Evaluate the fine-tuned model. 

        The process is similar to evaluating Sanger models. Note that you also need to use different scripts when evaluating dense models.

3.  Comparison with CPU and GPU.

    You can measure the latency of dense attention on CPU and GPU by executing `bench_cpu_gpu.py`.

## Internals

-  `configs/`: This sub-directory contains configuration files for dense models, sparse models, and static sparsity (BigBird, Longformer, etc.).
-  `data/`: This sub-directory is intended for storing manually downloaded datasets. Only the CLOTH dataset needs to be stored here, because GLUE and SQuAD are downloaded and managed automatically by the :hugs: ​transformers library.
-  `hardware/`: This sub-directory holds code related to the hardware implementation of Sanger. For the sake of clarity, we will describe this part separately in the next section.
   -  `src/main/scala/pe_row`: This sub-directory contains the main source code of the three hardware modules:
      -  `pe_row.scala`: The reconfigurable sparse PE array for computing SDDMM and SpMM.
      -  `mask.scala`: The dense low-bit PE array which produce the attention mask.
      -  `pack.scala`: The pack module which convert the attention mask to the configuration of the sparse PE array.
   -  `src/test/scala/pe_row`: This sub-directory contains the unit tests for the hardware modules.
      -  `pe_row_test.scala`: Unit test for the sparse PE array.
      -  `mask_test.scala`: Unit test for the dense PE array.
      -  `pack_text.scala`: Unit test for the pack module.
-  `outputs/`: This sub-directory is intended for storing training and evaluation results.
-  `scripts/`: This sub-directory holds the shell scripts for running experiments.
-  `bench_cpu_gpu.py`: This script benchmarks dense attention on CPU and GPU.
-  `bench_sanger.py`: This script is used to simulate the hardware performance of Sanger.
-  `modeling_<​​​​model>​​​.py`: These files contain implementations of the BERT, GPT2 and BART models, supporting both dense and sparse attention.
-  `modeling_sanger_attn.py`: This file contains an implementation of the sparse attention algorithm of Sanger, and some helper functions for measuring sparsity and load balance.
-  `modeling_static_spattn.py`: This file implements some attention mechanisms with static sparsity.
-  `run_<task>​​​​​​​.py`: These files are intended for training or evaluating models on GLUE, SQuAD or CLOTH.
-  `quant_utils.py`: This file contains some helper functions related to quantization.



## Citation

Liqiang Lu, Yicheng Jin, Hangrui Bi, Zizhang Luo, Peng Li, Tao Wang, Yun Liang. Sanger: A Co-Design Framework for Enabling Sparse Attention using Reconfigurable Architecture. The 54th International Symposium on Microarchitecture (MICRO’21), 2021.