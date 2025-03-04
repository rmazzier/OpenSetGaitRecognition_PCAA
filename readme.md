# Open-Set Gait Recognition from Sparse mmWave Radar Point Clouds

This repository contains the implementation of the experimental results described in: `R. Mazzieri, J. Pegoraro and M. Rossi, "Open-Set Gait Recognition from Sparse mmWave Radar Point Clouds" in [JournalNameHere], doi: [DOI here].` The paper is available [here](link/to/paper).

All the experiments are implemented in **Python version 3.8.10** and are based on PyTorch.

If you find our work and implementation useful for your research, please, cite us using the following BibTeX:
```
@arxivrefhere

@datasetrefhere
```

## 1. Setup
After setting up a new Python environment, install the required packages by running:
```
pip install -r requirements.txt
```

## 2. How to train PCAA 

Our proposed architecture, PCAA (Point Cloud Adversarial Autoencoder), was trained and validated on `mmGait10`, a newly proposed dataset containing radar point cloud traces from 10 human subjects, each with three distinct walking modalities. The dataset can be downloaded [here](link/to/dataset).

Our results, reported in Section VI-B and VI-C of the paper, can be reproduced by training the model from scratch. To do so:

1. Extract the dataset in the folder `data/multi-scenario_dataset/`

2. Run the file `PCAA_ablation.py` to train all the variations of PCAA reported in Table 4 of the paper, where each variation is trained 5 times, each with different seen/unseen splits.
If needed the script can be easily modified to train just a subset of all the models reported in the paper, by modifying the variables `n_training_classes` and `n_tests`. These control the considered problem openness and number of split variations respectively. At the end of each training, the model weights and configuration file are saved in `models/<run-name>`.  

3. To check model performance on the open set classification problem, run `inference_PCAA.py`. Final model predictions and ground truth labels are stored in the model folder in the form of .npy files, and final metrics are stored in .json files. Additionally, confusion matrices for the open-set problem are produced and saved in in `figures/<run-name>`.

4. In Figure 4 and 5 of our paper we also explore additional factors that could affect the final performance, namely the number of available points in the input point clouds, and specific walking manners of the subjects. Those results can be reproduced by simply running `train_pointsubsampling.py` and `inferene_scenarios.py` respectively. In this case, training and inference are performed in a single file. Model weights and final metrics are stored in the same fashion as point (2) and (3) of this list.

## 3. How to train the baseline

In our study we compare PCAA with the closest existing architecture that solves the open-set gait classification problem from radar traces, namely “Open-set Recognition based on Contrastive constraint and Ensemble based out-of-distribution Detection” (OR-CED).

```
Yang, Yang, et al. "Multiscenario open-set gait recognition based on radar micro-Doppler signatures." IEEE Transactions on Instrumentation and Measurement 71 (2022): 1-13.
```

To reproduce our results for the baseline:

1. Run `train_ORCED.py` to train the model under all the considered levels of openness and considering 5 different seen/unseen splits each time. Similarly to the previous script, only a subset of runs can be selected by modifying the relative variables. Again, model weights and configuration files are saved in `models/<run-name>`.
2. To check model performance on the open set classification problem, run `inference_ORCED.py`. Final model predictions and ground truth labels are stored in the same fashion of the PCAA ones.

