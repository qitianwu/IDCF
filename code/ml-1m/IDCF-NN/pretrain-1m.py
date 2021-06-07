import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import sys
import argparse
import yaml
from model import NNMFModel
from utils import generate_data

from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO, filename='record.log', format='%(message)s')

import torch

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
#fix_seed(1234)

parser = argparse.ArgumentParser(description='PMF')
parser.add_argument('--gpus', default='0', help='gpus')
args = parser.parse_args()

config = yaml.safe_load(open("./datainfo.yaml", 'r'))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LEARNING_RATE = 0.001 #default 0.001
DECAYING_FACTOR = 0.95 #default 0.95
LAMBDA_REG = 0.2 #default 0.05
BATCH_SIZE_TRAIN = 1024
BATCH_SIZE_TEST = 100000
n_epochs = 100


DATASET = 'ml-1m'
SPLIT_WAY = 'threshold'
THRESHOLD = 30
SUPP_RATIO = 0.8
TRAINING_RATIO = 1.0
datadir = '../../../data/'
n_user = config[DATASET]['n_user']
n_item = config[DATASET]['n_item']

train_set_supp, train_set_que, test_set_supp, test_set_que, user_supp_list, user_his_dic = \
				generate_data(datadir=datadir, 
							dataset=DATASET, 
							split_way=SPLIT_WAY,
							supp_ratio=SUPP_RATIO, 
							threshold=THRESHOLD,
							training_ratio=TRAINING_RATIO)

if SPLIT_WAY == 'all':
	train_set = torch.tensor(train_set_supp + train_set_que)
	test_set = torch.tensor(test_set_supp + test_set_que)
else:
	train_set = torch.tensor(train_set_supp)
	test_set = torch.tensor(test_set_supp)

train_set = train_set[torch.randperm(train_set.size(0))]
val_set = train_set[int(0.95*train_set.size(0)):]
train_set = train_set[:int(0.95*train_set.size(0))]

def train(model, optimizer, train_x, train_y):
	model.train()
	optimizer.zero_grad()
	pred_y = model(train_x)
	loss_r = torch.sum((train_y - pred_y) ** 2)
	loss_reg = model.regularization_loss()
	loss = loss_r + LAMBDA_REG * loss_reg
	loss.backward()
	optimizer.step()
	return loss_r.item(), loss_reg.item()

def test(model, test_x, test_y):
	model.eval()
	with torch.no_grad():
		pred_y = model(test_x)
		loss_r = torch.mean((test_y - pred_y) ** 2)
	y_hat, y = pred_y.cpu().numpy(), test_y.cpu().numpy()
	MAE = np.mean(np.abs(y_hat - y))
	RMSE = np.sqrt(np.mean(np.square(y_hat - y)))

	return loss_r.item(), MAE, RMSE

def save_model(model, path):
	torch.save(model.state_dict(), path+'model.pkl')

start_time = datetime.now()
train_size, test_size = train_set.size(0), test_set.size(0)
n_iter = n_epochs * train_size // BATCH_SIZE_TRAIN
bestMAE, bestRMSE = 10.0, 10.0
step = 0

model = NNMFModel(n_user, n_item).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=0.)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAYING_FACTOR)
val_set_supp_x = val_set[:, :2].long().to(device)
val_set_supp_y = val_set[:, 2].float().to(device)
test_set_supp_x = test_set[:, :2].long().to(device)
test_set_supp_y = test_set[:, 2].float().to(device)
for epoch in range(n_epochs):
	train_set = train_set[torch.randperm(train_size)]
	iter_num, loss_r_sum, loss_reg_sum = 0, 0., 0.
	for i in range(train_size // BATCH_SIZE_TRAIN + 1):
		train_set_supp_i = train_set[i*BATCH_SIZE_TRAIN : (i+1)*BATCH_SIZE_TRAIN]
		train_set_supp_i_x = train_set_supp_i[:, :2].long().to(device)
		train_set_supp_i_y = train_set_supp_i[:, 2].float().to(device)
		loss_r, loss_reg = train(model, optimizer, train_set_supp_i_x, train_set_supp_i_y)
		step += 1
		iter_num += 1
		loss_r_sum += loss_r
		loss_reg_sum += loss_reg
	loss_r_train = loss_r_sum / (iter_num * BATCH_SIZE_TRAIN)
	loss_reg_train = loss_reg_sum / (iter_num * BATCH_SIZE_TRAIN)
	print('Epoch {} Step {}: Train {:.4f} Reg: {:.4f}'.format(epoch, step, loss_r_train, loss_reg_train))
	iter_num, loss_r_sum, loss_reg_sum = 0, 0., 0.
	loss_r_test, MAE, RMSE = test(model, test_set_supp_x, test_set_supp_y)
	print('Test: {:.4f} MAE: {:.4f} RMSE: {:.4f}'.format(loss_r_test, MAE, RMSE))
	loss_r_val, MAE, RMSE = test(model, val_set_supp_x, val_set_supp_y)
	print('Val: {:.4f} MAE: {:.4f} RMSE: {:.4f}'.format(loss_r_val, MAE, RMSE))
	scheduler.step()

	if RMSE < bestRMSE:
		bestRMSE = RMSE
		save_model(model, path='./pretrain-1m/')




	


		