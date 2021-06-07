import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import argparse
import yaml
from model import IRMC_NN_Model, NNMFModel
from utils import *
from datetime import datetime
import torch
import torch.nn.functional as F

import logging
logging.basicConfig(level=logging.INFO, filename='test.log', format='%(message)s')

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
fix_seed(1234)

parser = argparse.ArgumentParser(description='PMF')
parser.add_argument('--gpus', default='0', help='gpus')
parser.add_argument('--extra', action="store_true", help='whether extra or inter')
args = parser.parse_args()
config = yaml.safe_load(open("./datainfo.yaml", 'r'))

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
device = torch.device('cuda')

BATCH_SIZE_TEST = 1024
HIS_MAXLEN = 100
HIS_SAMPLE_NUM = 20
EXTRA = args.extra

DATASET = 'amazon-books'
datadir = '../../../data/books_s20.pkl'
train_set_supp, train_set_que, test_set_supp, test_set_que, user_his_dic, user_supp_list = generate_data(datadir=datadir)
config = yaml.safe_load(open("./datainfo.yaml", 'r'))
n_user = config[DATASET]['n_user']
n_item = config[DATASET]['n_item']

test_set_supp = torch.tensor(test_set_supp)
test_set_que = torch.tensor(test_set_que)
supp_users = torch.tensor(user_supp_list, dtype = torch.long)

user_supp_num = len(user_supp_list)
user_que_num = n_user - user_supp_num

def sequence_adjust(seq):
	seq_new = seq
	if len(seq) <= 0:
		seq_new = [np.random.randint(0, n_item) for i in range(HIS_SAMPLE_NUM)]
	if len(seq) > HIS_MAXLEN:
		random.shuffle(seq)
		seq_new = seq[:HIS_MAXLEN]
	return seq_new

def neg_sampling(train_set_i, num_neg_per = 5):
	size = train_set_i.size(0)
	neg_iid = torch.randint(0, n_item, (num_neg_per * size, )).reshape(-1)
	return torch.stack([train_set_i[:, 0].repeat(num_neg_per), neg_iid, torch.zeros(num_neg_per * size)], dim=1)

def test(model, test_set, supp_or_que):
	model.eval()
	test_size = test_set.size(0)
	user_score_dict, user_label_dict = {}, {}
	for k in user_his_dic.keys():
		user_score_dict[k] = []
		user_label_dict[k] = []
	score_label = []
	for i in range(test_size // BATCH_SIZE_TEST + 1):
		with torch.no_grad():
			test_set_i = test_set[i*BATCH_SIZE_TEST : (i+1)*BATCH_SIZE_TEST]
			test_set_neg_i = neg_sampling(test_set_i)
			test_set_i = torch.cat([test_set_i, test_set_neg_i], dim=0)
			test_set_i_x = test_set_i[:, :2].long().to(device)
			test_set_i_y = test_set_i[:, 2].float().to(device)
			test_set_his_i = [torch.tensor(
				sequence_adjust( user_his_dic[test_set_i[k][0].item()] ),
				dtype = torch.long
				)   for k in range(test_set_i.size(0))]
			test_set_hl_i = [test_set_his_i[k].size(0) for k in range(test_set_i.size(0))]
			test_set_his_i = torch.nn.utils.rnn.pad_sequence(test_set_his_i, batch_first = True, padding_value = 0.).to(device)
			test_set_hl_i = torch.tensor(test_set_hl_i, dtype=torch.long).to(device)

			if supp_or_que == 'supp':
				pred_y = model(test_set_i_x)
			else:
				pred_y = model(test_set_i_x, test_set_his_i, test_set_hl_i)
		y_hat, y = pred_y.cpu().numpy().tolist(), test_set_i_y.cpu().numpy().tolist()
		for i in range(len(y)):
			score_label.append([y_hat[i], y[i]])
		for k in range(test_set_i.size(0)):
			u, s, y = test_set_i_x[k, 0].item(), pred_y[k].item(), test_set_i_y[k].item()
			user_score_dict[u] += [s]
			user_label_dict[u] += [y]
	score_label = sorted(score_label, key=lambda d:d[0], reverse=True)
	ndcg_sum, num = 0., 0
	for k in user_score_dict.keys():
		if len(user_score_dict[k]) <= 1:
			continue
		ndcg_sum += ndcg_k(user_score_dict[k], user_label_dict[k], len(user_score_dict[k]))
		num += 1
	return score_label, ndcg_sum, num

def load_model_s(model, path):
	model.load_model(path+'model.pkl')

def load_model_q(model, path):
	if EXTRA:
		model.load_model(path + 'model-extra.pkl')
	else:
		model.load_model(path+'model-inter.pkl')


if EXTRA:
	model_q = IRMC_NN_Model(n_user=n_user,
							n_item=n_item,
							supp_users=supp_users,
							device=device).to(device)
	load_model_q(model_q, './train-books/')

	score_label_q, ndcg_sum_q, num_q = test(model_q, test_set_que, supp_or_que='que')
	AUC_q = auc_calc(score_label_q)
	NDCG_q = ndcg_sum_q / num_q
	log = 'Que Test Result AUC_q: {:.4f} NDCG_q: {:.4f}'.format(AUC_q, NDCG_q)
	print(log)
else:
	model_s = NNMFModel(n_user, n_item).to(device)
	load_model_s(model_s, './pretrain-books/')
	score_label_s, ndcg_sum_s, num_s = test(model_s, test_set_supp, supp_or_que='supp')
	AUC_s = auc_calc(score_label_s)
	NDCG_s = ndcg_sum_s / num_s
	log = 'Supp Test Result AUC_s: {:.4f} NDCG_s: {:.4f}'.format(AUC_s, NDCG_s)
	print(log)

	model_q = IRMC_NN_Model(n_user = n_user,
					n_item = n_item,
					supp_users = supp_users,
					device = device).to(device)
	load_model_q(model_q, './train-books/')

	score_label_q, ndcg_sum_q, num_q = test(model_q, test_set_que, supp_or_que='que')
	AUC_q = auc_calc(score_label_q)
	NDCG_q = ndcg_sum_q / num_q
	log = 'Que Test Result AUC_q: {:.4f} NDCG_q: {:.4f}'.format(AUC_q, NDCG_q)
	print(log)

	score_label = score_label_s + score_label_q
	score_label = sorted(score_label, key=lambda d:d[0], reverse=True)
	AUC = auc_calc(score_label)
	NDCG = ( ndcg_sum_s + ndcg_sum_q )/ (num_s+num_q)
	log = 'All Test Result AUC: {:.4f} NDCG: {:.4f}'.format(AUC, NDCG)
	print(log)
