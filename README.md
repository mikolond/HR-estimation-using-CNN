# Hearth Rate estimation using Convolution Neural network
This is repositiry for doploma work focusing on implpementation of the visual hearth rate estimator according to [Visual Heart Rate Estimation with
Convolutional Neural Network](https://cmp.felk.cvut.cz/~spetlrad/ecg-fitness/visual-heart-rate.pdf)

## Contents
Repository contains a tool for creating synthetic data for training(mainly for debuging purposes), tools for creating dataset for training from synthetic data and from ecg-fitness dataset. Then it contains code for training of the Extractor and a tool for validating accuracy of the trained model.

### Synthetic data dataset
 The synthetic data can be created using /synthetic_data/create_synhtetic_dataset.py
It can be called with arguments see help (python create_synthetic_data.py -h)

### Real data dataset
A dataset from real data can be created using dataset_creator.py
Currently it supports only creating dataset from ECG-fitness dataset.
It can be called with argument, see help.

### Training 
The training is implemented in train.py. See help. The weights are  being saved each epoch of training in folder model_weights. The training uses tensorboard to log the progress of the training.

### Evaluation
If the model is trained it can be evaluated using evaluate.py. See help. It calculates the average deviation of the predicted hearth-rate and the real one. It also caluclate root mean square average and the signal to noise ration(since it is the loss function).
