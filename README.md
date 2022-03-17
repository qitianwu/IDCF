## InDuctive Collaborative Filtering (IDCF)

The codes and data used in ICML'21 paper [Towards Open-World Recommendation: An Inductive Model-Based Collaborative Filtering Apparoach](https://arxiv.org/abs/2007.04833). This work proposes a new collaborative filtering approach for recsys. The new method could achieve inductive learning for new users in testing set and also help to address cold-start problem on user side.

### Dependency

Python 3.8, Pytorch 1.7, Pytorch Geometric 1.6

### Download trained model and data

The trained model and preprocessed data can be downloaded by the Google drive

    https://drive.google.com/drive/folders/1rTfOKZJ-zYrNY9hDUtU9H-UG9fPPQOds?usp=sharing

### Reproduce results

To reproduce the results in our paper (i.e. Table 2, 3, 4), you need to first download the trained model and data to corresponding folders and run the test.py script in each folder. Take Movielens-1M dataset as an example. You need to first download the folder ***data/ml-1m.pkl*** from the Google drive to ***./data*** in your own computer and download ***model/ml-1m/*** to ***./code/ml-1m/*** in your computer. Then you can run

    python ./code/ml-1m/IDCF-NN/test-1m.py

to reproduce the results of IDCF-NN model on few-shot query users on ML-1M. Also, you can run

    python ./code/ml-1m/IDCF-NN/test-1m.py --extra

to reproduce the results of IDCF-NN model on zero-shot new users on ML-1M.

### Run the code for training

Our model needs a two-stage training. To train the model from the beginning, you can run two scripts in order. 

First, you need to run
    
    python ./code/ml-1m/IDCF-NN/pretrain-1m.py

to pretrain the matrix factorization model. Alternatively, you can skip the pretrain stage by directly using our pretrained model, i.e., download the model file from the path ***model/ml-1m/IDCF-NN/pretrain-1m/model.pkl*** in the Google drive to ***./code/ml-1m/IDCF-NN/pretrain-1m/model.pkl*** in your computer. 

Second, you need to run the script ***train-1m.py***. 

    python ./code/ml-1m/IDCF-NN/train-1m.py

Also, in our paper, we consider two scenarios: inductive learning for interpolation (for few-shot query users) and inductive learning for extrapolation (for new test users). For running the former case, you need to set the variable EXTRA as False in ***train-1m.py***. For the latter case, you can set EXTRA=True in ***train-1m.py***.

For model details, please refer to our paper.

If you found the codes or datasets useful, please consider cite our paper:

    @inproceedings{wu2021idcf,
    title = {Towards Open-World Recommendation: An Inductive Model-Based Collaborative Filtering Apparoach},
    author = {Qitian Wu and Hengrui Zhang and Xiaofeng Gao and Junchi Yan and Hongyuan Zha},
    booktitle = {International Conference on Machine Learning (ICML)},
    year = {2021}
    }
    
