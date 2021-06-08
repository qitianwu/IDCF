import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import argparse
import yaml
from model import GCMCModel
from utils import *
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

LEARNING_RATE = 0.001
DECAYING_FACTOR = 1.
LAMBDA_REG = 0.05
BATCH_SIZE_TRAIN = 1024
BATCH_SIZE_TEST = 1024
HIS_MAXLEN = 100
HIS_SAMPLE_NUM = 20
n_epochs = 1 # 500

DATASET = 'ml-1m'
TRAINING_RATIO = 1
THRESHOLD = 30
datadir = '../../../data/'
n_user = config[DATASET]['n_user']
n_item = config[DATASET]['n_item']
n_rating = config[DATASET]['n_rating']

train_set_supp, train_set_que, test_set_supp, test_set_que, user_his_dic, edge_UI = \
				generate_data(datadir=datadir, 
							dataset=DATASET,
							threshold=THRESHOLD,
							training_ratio=TRAINING_RATIO,
							sample_graph=False)

train_set_que = torch.tensor(train_set_que)
train_set_supp = torch.tensor(train_set_supp)
test_set = torch.tensor(test_set_supp + test_set_que)
test_set_supp = torch.tensor(test_set_supp)
test_set_que = torch.tensor(test_set_que)
edge_IU = []	
for n in range(n_rating):
	edge_UI[n] = torch.tensor(edge_UI[n])
	edge_IU_n = edge_UI[n].transpose(1, 0).contiguous()
	edge_IU.append(edge_IU_n)

def train(model, optimizer, i):
	model.train()
	optimizer.zero_grad()

	train_set_i = train_set_que[i*BATCH_SIZE_TRAIN : (i+1)*BATCH_SIZE_TRAIN]
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

def test(model, test_set):
	model.eval()
	loss_r_test_sum, l1_sum, l2_sum, ndcg_sum, num = 0., 0., 0., 0., 0
	test_size = test_set.size(0)
	user_score_dict, user_label_dict = {}, {}
	for k in user_his_dic.keys():
		user_score_dict[k] = []
		user_label_dict[k] = []
	for i in range(test_size // BATCH_SIZE_TEST + 1):
		with torch.no_grad():
			test_set_i = test_set[i*BATCH_SIZE_TEST : (i+1)*BATCH_SIZE_TEST]
			test_set_i_x = test_set_i[:, :2].long().to(device)
			test_set_i_y = test_set_i[:, 2].float().to(device)
			edge_UI_i = [edge_UI[n][test_set_i_x[:, 0]].to(device) for n in range(n_rating)]
			edge_IU_i = [edge_IU[n][test_set_i_x[:, 1]].to(device) for n in range(n_rating)]

			pred_y = model(test_set_i_x, edge_UI_i, edge_IU_i)
			loss_r = torch.sum((test_set_i_y - pred_y) ** 2)
		y_hat, y = pred_y.cpu().numpy(), test_set_i_y.cpu().numpy()
		loss_r_test_sum += loss_r.item()
		l1_sum += np.sum( np.abs(y_hat - y) )
		l2_sum += np.sum( np.square(y_hat - y) )
		for k in range(test_set_i.size(0)):
			u, s, y = test_set_i_x[k, 0].item(), pred_y[k].item(), test_set_i_y[k].item()
			user_score_dict[u] += [s]
			user_label_dict[u] += [y]
	TestLoss = loss_r_test_sum / test_size
	MAE = l1_sum / test_size
	RMSE = np.sqrt( l2_sum / test_size )
	for k in user_score_dict.keys():
		if len(user_score_dict[k]) <= 1:
			continue
		ndcg_sum += ndcg_k(user_score_dict[k], user_label_dict[k], len(user_score_dict[k]))
		num += 1
	return TestLoss, MAE, RMSE, ndcg_sum, num

def load_model(model, path):
	model.load_model(path+'model.pkl')

model = GCMCModel(n_user = n_user, 
				n_item = n_item,
				n_rating = n_rating,
				device = device).to(device)
load_model(model, path='./train-1m/')

loss_r_test, MAE_s, RMSE_s, ndcg_sum_s, num_s = test(model, test_set_supp)
NDCG_s = ndcg_sum_s / num_s
log = 'Key Test Result: MAE: {:.4f} RMSE: {:.4f} NDCG {:.4f}'.format(MAE_s, RMSE_s, NDCG_s)
print(log)

loss_r_test, MAE_q, RMSE_q, ndcg_sum_q, num_q = test(model, test_set_que)
NDCG_q = ndcg_sum_q / num_q
log = 'Que Test Result: MAE: {:.4f} RMSE: {:.4f} NDCG {:.4f}'.format(MAE_q, RMSE_q, NDCG_q)
print(log)

supp_size, que_size = test_set_supp.size(0), test_set_que.size(0)
MAE = ( MAE_s * supp_size + MAE_q * que_size )/ (supp_size+que_size)
RMSE = np.sqrt( (RMSE_s**2 * supp_size + RMSE_q**2 * que_size) / (supp_size+que_size))
NDCG = (ndcg_sum_q + ndcg_sum_s) / (num_q + num_s)
log = 'All Test Result: MAE: {:.4f} RMSE: {:.4f} NDCG {:.4f}'.format(MAE, RMSE, NDCG)
print(log)