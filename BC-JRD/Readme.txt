1. Download the dataset and place it in the "data" folder. 
    You can find the dataset at the following link: 
    https://ieee-dataport.org/documents/object-wise-just-recognizable-distortion-dataset

2. The partition information of the dataset is stored in the "train/val/test.json" files, located in the "jsonfiles" folder.

3. "train.py" contains the code for training the model.
    "PredictJRD.py" is used to predict JRD using the trained model. 
    "model.py" provides the model structure. 
    "my_dataset.py" provides the function for instantiating the dataset. 
    "my_utils.py" provides functions for training, validation, and testing.