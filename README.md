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
