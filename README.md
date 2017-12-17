# Applying Transfer Learning to Question Answering Systems
2017 Fall MIT 6.806 NLP project

**Authors:**
- Grace Lam
- Austin Wang

## Usage

To run the models, move into the `src` directory and run the following commands, which will train the models with the default parameters and output evaluation results. Read the files for all parameter options, and modify the control output variables to change the behavior of the program (e.g. debug mode, iterating through hyper-parameters, loading a model from file, etc.).

**In-Domain CNN Model**:

`python2 cnn_model.py`

**In-Domain LSTM Model**:

`python2 lstm_model.py`

**TF-IDF Baseline Model**:

`python2 tfidf.py`

**Direct Transfer CNN Model**:

`python2 direct_transfer.py`

**Domain Adaptation CNN-FFNN Model**:

`python2 domain_adaptation.py`

**Domain Adaptation CNN-LogReg Model**:

`python2 domain_adaptation_2.py`

## Datasets

Our project uses the [AskUbuntu](https://github.com/taolei87/askubuntu) and [Android](https://github.com/jiangfeng1124/Android) datasets from the associated Stack Exchange forums. To use these datasets, clone or download them and place the folders in the root of the project with the names `askubuntu-master` and `Android-master`. You will additionally need to download the [GloVe embeddings](https://nlp.stanford.edu/projects/glove/), unzipped and stored in the root of the project, and the more complete 200-Dimensional word2vec embeddings found in the askubuntu repository README (place this file in the `askubuntu-master/vector` directory).
