## This repository contains the codes and data used in ICML'21 paper "Towards Open-World Recommendation: An Inductive Model-Based Collaborative Filtering Apparoach"

The trained model and preprocessed data can be downloaded by the Google drive

    https://drive.google.com/drive/folders/1rTfOKZJ-zYrNY9hDUtU9H-UG9fPPQOds?usp=sharing

To reproduce the results in our paper (i.e. Table 2, 3, 4), you need to first download the trained model and data to corresponding folders and run the test.py script in each folder. Take Movielens-1M dataset as an example. You need to first download data/ml-1m.pkl from the Google drive to ./data in your own computer and download model/ml-1m/ to ./code/ml-1m/ in your computer. Then you can run

    python ./code/ml-1m/IDCF-NN/test-1m.py

to reproduce the results of IDCF-NN model on few-shot query users on ML-1M. Also, you can run

    pytho ./code/ml-1m/IDCF-NN/test-1m.py --extra

to reproduce the results of IDCF-NN model on zero-shot new users on ML-1M.

More information will be updated.
