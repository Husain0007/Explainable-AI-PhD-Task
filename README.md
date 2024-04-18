# CORA Dataset Node Classification Task

<p>In this project I developed a Node Classifier for the CORA dataset consisting of 2708 scientific publications. To prepare the </p>

## Tasks
- [x] Load the data.
- [x] Split the data using 10-fold cross validation.
- [x] Develop a Machine Learning approach to learn and predict the subjects of papers.
- [x] Store predictions in a [TSV file](./inference_predictions.tsv).
- [x] Evaluate approach in terms of **accuracy**.

## System Requirements

* Ubuntu 20.04.5 LTS
* Conda == 22.11.1 ([Installion Link](https://anaconda.org/anaconda/conda/files?version=22.11.1&page=1)) **or** Mamba == 1.2.0 ([Installtion Link](https://github.com/mamba-org/mamba/releases/tag/2023.01.16)).


## Set Up & Model Verification

* To evaluate the performance of the`final_model.bin` using the `check.py` script first the required conda/mamba virtual environment should be created and activated.
* To create the virtual environment execute the below command.
```
    (base) conda env create -f environment.yaml
```
* Once the `stellar_env` is created it can be activated as follows:
```
    (base) source activate stellar_env
```
* To evaluate the `final_model.bin` models' performance execute the following command after activating the virtual environment.
```
    (stellar_env) python check.py
```
* To deactivate the virtual environment execute the following command.
```
    (stellar_env) conda deactivate
```

*  *  *



