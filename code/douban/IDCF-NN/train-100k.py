import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import argparse
import yaml
from model import IRMC_NN_Model
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

LEARNING_RATE = 0.001 # Extra 0.001 Inter 0.01
DECAYING_FACTOR = 0.95
LAMBDA_REC = 1.
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 1024
HIS_MAXLEN = 100
HIS_SAMPLE_NUM = 20
n_epochs = 500 # 500

DATASET = 'ml-100k'
SPLIT_WAY = 'threshold'
EXTRA = True
THRESHOLD = 30
SUPP_RATIO = 0.8
TRAINING_RATIO = 1
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

supp_users = torch.tensor(user_supp_list, dtype = torch.long)
if EXTRA:
	train_set = torch.tensor(train_set_supp)
else:
	train_set = torch.tensor(train_set_que)
test_set = torch.tensor(test_set_que)

train_set = train_set[torch.randperm(train_set.size(0))]
val_set = train_set[int(0.95*train_set.size(0)):]
if EXTRA:
	pass
else:
	train_set = train_set[:int(0.95*train_set.size(0))]

def sequence_adjust(seq):
	seq_new = seq
	if len(seq) <= 0:
		seq_new = [np.random.randint(0, n_item) for i in range(HIS_SAMPLE_NUM)]
	if len(seq) > HIS_MAXLEN:
		random.shuffle(seq)
		seq_new = seq[:HIS_MAXLEN]
	return seq_new

def train(model, optimizer, i):
	model.train()
	optimizer.zero_grad()
	
	train_set_que_i = train_set[i*BATCH_SIZE_TRAIN : (i+1)*BATCH_SIZE_TRAIN]
	
	train_set_i_x = train_set_que_i[:, :2].long().to(device)
	train_set_i_y = train_set_que_i[:, 2].float().to(device)
	train_set_his_i = [torch.tensor(
		sequence_adjust( user_his_dic[train_set_que_i[k][0].item()] ),
		dtype = torch.long
		)   for k in range(train_set_que_i.size(0))]
	train_set_hl_i = [train_set_his_i[k].size(0) for k in range(train_set_que_i.size(0))]
	train_set_his_i = torch.nn.utils.rnn.pad_sequence(train_set_his_i, batch_first = True, padding_value = 0.).to(device)
	train_set_hl_i = torch.tensor(train_set_hl_i, dtype=torch.long).to(device)
	if EXTRA:
		pred_y, user_emb_ind, user_emb_trd = model(train_set_i_x, train_set_his_i, train_set_hl_i, mode='EXTRA')
		loss = torch.sum((train_set_i_y - pred_y) ** 2)
		user_emb_trd_ = user_emb_trd.unsqueeze(0).repeat(user_emb_ind.size(0), 1, 1)
		user_emb_ind_ = user_emb_ind.unsqueeze(1).repeat(1, user_emb_trd.size(0), 1)
		dot_prod = torch.sum(torch.mul(user_emb_trd_, user_emb_ind_), dim=-1)
		loss_con = - torch.mean(
					dot_prod.diagonal() - torch.logsumexp(dot_prod, dim=-1)
				)
		loss += 10.0 * loss_con
	else:
		pred_y = model(train_set_i_x, train_set_his_i, train_set_hl_i)
		loss = torch.sum((train_set_i_y - pred_y) ** 2)
	loss.backward()
	optimizer.step()
	return loss.item(), 0.

def test(model, test_set, i):
	model.eval()
	with torch.no_grad():
		test_set_i = test_set[i*BATCH_SIZE_TEST : (i+1)*BATCH_SIZE_TEST]
		test_set_i_x = test_set_i[:, :2].long().to(device)
		test_set_i_y = test_set_i[:, 2].float().to(device)
		test_set_his_i = [torch.tensor(
			sequence_adjust( user_his_dic[test_set_i[k][0].item()] ),
			dtype = torch.long
			)   for k in range(test_set_i.size(0))]
		test_set_hl_i = [test_set_his_i[k].size(0) for k in range(test_set_i.size(0))]
		test_set_his_i = torch.nn.utils.rnn.pad_sequence(test_set_his_i, batch_first = True, padding_value = 0.).to(device)
		test_set_hl_i = torch.tensor(test_set_hl_i, dtype=torch.long).to(device)

		pred_y = model(test_set_i_x, test_set_his_i, test_set_hl_i)
		loss_r = torch.sum((test_set_i_y - pred_y) ** 2)
	y_hat, y = pred_y.cpu().numpy(), test_set_i_y.cpu().numpy()
	l1 = np.sum( np.abs(y_hat - y) )
	l2 = np.sum( np.square(y_hat - y) )

	return loss_r.item(), l1, l2

def save_model(model, path):
	if EXTRA:
		torch.save(model.state_dict(), path+'model-extra.pkl')
	else:
		torch.save(model.state_dict(), path+'model-inter.pkl')

def load_model(model, path):
	model.load_embedding_nn(path+'model.pkl')

train_size, val_size, test_size = train_set.size(0), val_set.size(0), test_set.size(0)
n_iter = n_epochs * train_size // BATCH_SIZE_TRAIN
bestRMSE = 10.0

model = IRMC_NN_Model(n_user = n_user, 
				n_item = n_item, 
				supp_users = supp_users, 
				device = device).to(device)
load_model(model, './pretrain-100k/')
optimizer = torch.optim.Adam( filter(lambda p: p.requires_grad, model.parameters()), lr = LEARNING_RATE, weight_decay=0.)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=DECAYING_FACTOR)
start_time = datetime.now()
for epoch in range(n_epochs):
	train_set = train_set[torch.randperm(train_size)]
	loss_r_sum, loss_rec_sum = 0., 0.
	for i in range(train_size // BATCH_SIZE_TRAIN + 1):
		loss_r, loss_rec = train(model, optimizer, i)
		loss_r_sum += loss_r
		loss_rec_sum += loss_rec
	loss_r_train = loss_r_sum / train_size
	loss_rec_train = loss_rec_sum / train_size
	cost_time = str((datetime.now() - start_time) / (epoch+1) * (n_epochs - epoch)).split('.')[0]
	print('Epoch {}: TrainLoss {:.4f} RecLoss: {:.4f} (left: {})'.format(epoch, loss_r_train, loss_rec_train, cost_time))
	scheduler.step()

	loss_r_test_sum, l1_sum, l2_sum = 0., 0., 0.
	for i in range(test_size // BATCH_SIZE_TEST + 1):
		loss_r_test, l1, l2 = test(model, test_set, i)
		loss_r_test_sum += loss_r_test
		l1_sum += l1
		l2_sum += l2
	TestLoss = loss_r_test_sum / test_size
	MAE = l1_sum / test_size
	RMSE = np.sqrt( l2_sum / test_size )
	print('TestLoss: {:.4f} MAE: {:.4f} RMSE: {:.4f}'.format(TestLoss, MAE, RMSE))

	if EXTRA:
		save_model(model, './train-100k/')
	else:
		loss_r_val_sum, l1_sum, l2_sum = 0., 0., 0.
		for i in range(val_size // BATCH_SIZE_TEST + 1):
			loss_r_val, l1, l2 = test(model, val_set, i)
			loss_r_val_sum += loss_r_val
			l1_sum += l1
			l2_sum += l2
		ValLoss = loss_r_val_sum / val_size
		MAE = l1_sum / val_size
		RMSE = np.sqrt( l2_sum / val_size )
		print('ValLoss: {:.4f} MAE: {:.4f} RMSE: {:.4f}'.format(ValLoss, MAE, RMSE))
		if RMSE < bestRMSE:
			bestRMSE = RMSE
			save_model(model, './train-100k/')
