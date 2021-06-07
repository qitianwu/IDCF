import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import argparse
import yaml
from model import GCMCModel
from utils import generate_data
from datetime import datetime
import torch

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(1234)

parser = argparse.ArgumentParser(description='PMF')
parser.add_argument('--gpus', default='0', help='gpus')
args = parser.parse_args()

config = yaml.safe_load(open("./datainfo.yaml", 'r'))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda')

LEARNING_RATE = 0.001 #default 0.001
DECAYING_FACTOR = 0.99 #default 0.95
LAMBDA_REG = 0.02 #default 0.05
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 1000
n_epochs = 100


DATASET = 'douban'
SPLIT_WAY = 'threshold'
THRESHOLD = 30
SUPP_RATIO = 0.9
TRAINING_RATIO = 1
datadir = '../../../data/'
n_user = config[DATASET]['n_user']
n_item = config[DATASET]['n_item']
n_rating = config[DATASET]['n_rating']

train_set_supp, train_set_que, test_set_supp, test_set_que, user_his_dic, edge_UI = \
				generate_data(datadir=datadir, 
							dataset=DATASET, 
							split_way=SPLIT_WAY,
							supp_ratio=SUPP_RATIO, 
							threshold=THRESHOLD,
							training_ratio=TRAINING_RATIO)

train_set = torch.tensor(train_set_supp + train_set_que)
test_set = torch.tensor(test_set_supp + test_set_que)
edge_IU = []	
for n in range(n_rating):
	edge_UI[n] = torch.tensor(edge_UI[n])
	edge_IU_n = edge_UI[n].transpose(1, 0).contiguous()
	edge_IU.append(edge_IU_n)

def train(model, optimizer, i):
	model.train()
	optimizer.zero_grad()

	train_set_i = train_set[i*BATCH_SIZE_TRAIN : (i+1)*BATCH_SIZE_TRAIN]
	train_set_i_x = train_set_i[:, :2].long().to(device)
	train_set_i_y = train_set_i[:, 2].long().to(device)
	edge_UI_i = [edge_UI[n][train_set_i_x[:, 0]].to(device) for n in range(n_rating)]
	edge_IU_i = [edge_IU[n][train_set_i_x[:, 1]].to(device) for n in range(n_rating)]

	pred_y = model(train_set_i_x, edge_UI_i, edge_IU_i)
	loss_r = torch.sum((train_set_i_y - pred_y) ** 2)
	loss_reg = model.regularization_loss()
	loss = loss_r + LAMBDA_REG * loss_reg
	loss.backward()
	optimizer.step()
	return loss_r.item(), loss_reg.item()

def test(model):
	model.eval()
	loss_r_test_sum, l1_sum, l2_sum = 0., 0., 0.
	test_size = test_set.size(0)
	for i in range(test_size // BATCH_SIZE_TEST + 1):
		with torch.no_grad():
			test_set_i = test_set[i*BATCH_SIZE_TEST : (i+1)*BATCH_SIZE_TEST]
			test_set_i_x = test_set_i[:, :2].long().to(device)
			test_set_i_y = test_set_i[:, 2].long().to(device)
			edge_UI_i = [edge_UI[n][test_set_i_x[:, 0]].to(device) for n in range(n_rating)]
			edge_IU_i = [edge_IU[n][test_set_i_x[:, 1]].to(device) for n in range(n_rating)]

			pred_y = model(test_set_i_x, edge_UI_i, edge_IU_i)
			loss_r = torch.sum((test_set_i_y - pred_y) ** 2)
		y_hat, y = pred_y.cpu().numpy(), test_set_i_y.cpu().numpy()
		loss_r_test_sum += loss_r.item()
		l1_sum += np.sum( np.abs(y_hat - y) )
		l2_sum += np.sum( np.square(y_hat - y) )
	TestLoss = loss_r_test_sum / test_size
	MAE = l1_sum / test_size
	RMSE = np.sqrt( l2_sum / test_size )

	return TestLoss, MAE, RMSE

def save_model(model, path):
	torch.save(model.state_dict(), path+'model.pkl')

start_time = datetime.now()
train_size, test_size = train_set.size(0), test_set.size(0)
n_iter = n_epochs * train_size // BATCH_SIZE_TRAIN
bestRMSE = 10.0
step = 0

model = GCMCModel(n_user = n_user, 
				n_item = n_item, 
				n_rating = n_rating, 
				embedding_size=16, 
				hidden_size=16,
				device = device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=5e-2)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAYING_FACTOR)
for epoch in range(n_epochs):
	train_set = train_set[torch.randperm(train_size)]
	iter_num, loss_r_sum, loss_reg_sum = 0, 0., 0.
	for i in range(train_size // BATCH_SIZE_TRAIN + 1):
		
		loss_r, loss_reg = train(model, optimizer, i)
		step += 1
		iter_num += 1
		loss_r_sum += loss_r
		loss_reg_sum += loss_reg
	loss_r_train = loss_r_sum / train_size
	loss_reg_train = loss_reg_sum / train_size
	print('Epoch {} Step {}: Train {:.4f} Reg: {:.4f}'.format(epoch, step, loss_r_train, loss_reg_train))
	iter_num, loss_r_sum, loss_reg_sum = 0, 0., 0.
	loss_r_test, MAE, RMSE = test(model)
	print('Test: {:.4f} MAE: {:.4f} RMSE: {:.4f}'.format(loss_r_test, MAE, RMSE))
	scheduler.step()

	if RMSE < bestRMSE:
		bestRMSE = RMSE
		save_model(model, path='./train-douban/')