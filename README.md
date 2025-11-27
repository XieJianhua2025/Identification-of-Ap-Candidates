# Identification-of-Ap-Candidates
Welcome! Here we provide the ECACNet code (PyTorch format) and the validation set list for the seven stellar classes.
# ECACNet.py
This repository contains the core network of ECACNet, which is implemented in PyTorch and designed for classifying Ap stars and six other stellar types. For training purposes, you may need to add specific parameters during the training process, such as the optimizer, learning rate, and others.
# ECACNet_model.pth
This file contains the trained ECACNet model, which can be directly loaded for inference or further fine-tuning. It includes the learned weights from training on Ap stars and the six other stellar classes.
# validation data.csv
This file provides information about the validation dataset, including details such as designation, obsid, and other related parameters.
# Table6 and Table7
Table 6 presents the results of our cross-matching with the known Ap star catalog, while Table 7 lists our newly discovered Ap candidate stars.
# If you use the code or dataset provided in this repository in your research, please cite the following [paper](https://doi.org/10.1093/mnras/staf1962):Thank you for citing our work.

