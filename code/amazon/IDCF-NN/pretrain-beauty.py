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
from utils import *

from datetime import datetime
import logging
logging.basicConfig(level=logging.INFO, filename='record.log', format='%(message)s')

import torch
import torch.nn.functional as F

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATASET = 'amazon-beauty'
LEARNING_RATE = 0.01
DECAYING_FACTOR = 0.95
LAMBDA_REG = 0.1
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 100000
n_epochs = 100

datadir = '../../../data/beauty_s20.pkl'
train_set_supp, train_set_que, test_set_supp, test_set_que, user_his_dic, user_supp_list = generate_data(datadir=datadir)
config = yaml.safe_load(open("./datainfo.yaml", 'r'))
n_user = config[DATASET]['n_user']
n_item = config[DATASET]['n_item']

train_set = torch.tensor(train_set_supp)
test_set = torch.tensor(test_set_supp)

train_set = train_set[torch.randperm(train_set.size(0))]
val_set = train_set[int(0.95*train_set.size(0)):]
train_set = train_set[:int(0.95*train_set.size(0))]

def neg_sampling(train_set_i, num_neg_per = 5):
	size = train_set_i.size(0)
	neg_iid = torch.randint(0, n_item, (num_neg_per * size, )).reshape(-1)
	return torch.stack([train_set_i[:, 0].repeat(num_neg_per), neg_iid, torch.zeros(num_neg_per * size)], dim=1)
	#return train_set_i


def train(model, optimizer, epoch, step, train_x, train_y):
	model.train()
	optimizer.zero_grad()
	pred_y = model(train_x)
	loss_r = F.binary_cross_entropy_with_logits(pred_y, train_y, reduction='sum')
	loss_reg = model.regularization_loss()
	loss = loss_r + LAMBDA_REG * loss_reg
	loss.backward()
	optimizer.step()
	return loss_r.item(), loss_reg.item()

def test(model, epoch, test_x, test_y):
	model.eval()
	with torch.no_grad():
		pred_y = torch.sigmoid(model(test_x))
		loss_r = F.binary_cross_entropy_with_logits(pred_y, test_y, reduction='mean')
	y_hat, y = pred_y.cpu().numpy().tolist(), test_y.cpu().numpy().tolist()
	score_label = []
	for i in range(len(y)):
		score_label.append([y_hat[i], y[i]])
	score_label = sorted(score_label, key=lambda d:d[0], reverse=True)
	AUC = auc_calc(score_label)

	return loss_r.item(), AUC

def save_model(model, path):
	torch.save(model.state_dict(), path+'model.pkl')

def load_model(model, path):
	model.load_model(path+'model.pkl')

start_time = datetime.now()
train_size, test_size = train_set.size(0), test_set.size(0)
n_iter = n_epochs * train_size // BATCH_SIZE_TRAIN
bestAUC = 0.0
step = 0

model = NNMFModel(n_user, n_item).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=0.)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAYING_FACTOR)
val_set_neg = neg_sampling(val_set)
val_set = torch.cat([val_set, val_set_neg], dim=0)
val_set_supp_x = val_set[:, :2].long().to(device)
val_set_supp_y = val_set[:, 2].float().to(device)
test_set_neg = neg_sampling(test_set)
test_set = torch.cat([test_set, test_set_neg], dim=0)
test_set_supp_x = test_set[:, :2].long().to(device)
test_set_supp_y = test_set[:, 2].float().to(device)
for epoch in range(n_epochs):
	train_set = train_set[torch.randperm(train_size)]
	iter_num, loss_r_sum, loss_reg_sum = 0, 0., 0.
	for i in range(train_size // BATCH_SIZE_TRAIN + 1):
		train_set_supp_i = train_set[i*BATCH_SIZE_TRAIN : (i+1)*BATCH_SIZE_TRAIN]
		train_set_supp_neg_i = neg_sampling(train_set_supp_i)
		train_set_supp_i = torch.cat([train_set_supp_i, train_set_supp_neg_i], dim=0)
		train_set_supp_i_x = train_set_supp_i[:, :2].long().to(device)
		train_set_supp_i_y = train_set_supp_i[:, 2].float().to(device)
		loss_r, loss_reg = train(model, optimizer, epoch, step, train_set_supp_i_x, train_set_supp_i_y)
		step += 1
		iter_num += 1
		loss_r_sum += loss_r
		loss_reg_sum += loss_reg
		# if step % 1000 == 0:
	loss_r_train = loss_r_sum / (iter_num * BATCH_SIZE_TRAIN)
	loss_reg_train = loss_reg_sum / (iter_num * BATCH_SIZE_TRAIN)
	print('Epoch {} Step {}: Train {:.4f} Reg: {:.4f}'.format(epoch, step, loss_r_train, loss_reg_train))
	iter_num, loss_r_sum, loss_reg_sum = 0, 0., 0.
	loss_r_val, AUC_val = test(model, epoch, val_set_supp_x, val_set_supp_y)
	loss_r_te, AUC_te = test(model, epoch, test_set_supp_x, test_set_supp_y)
	print('Val: {:.4f} AUC: {:.4f}'.format(loss_r_val, AUC_val))
	print('Test: {:.4f} AUC: {:.4f}'.format(loss_r_te, AUC_te))
	scheduler.step()

	if AUC_val > bestAUC:
		bestAUC = AUC_val
		save_model(model, path='./pretrain-beauty/')
		