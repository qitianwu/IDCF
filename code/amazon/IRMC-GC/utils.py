import pickle
import random
import yaml
import numpy as np

random.seed(1234)

def generate_data(datadir, sample_graph=True):
    try:
        with open(datadir, 'rb') as f:
            ucs_set = pickle.load(f)
            cs_set = pickle.load(f)
            u_his_list = pickle.load(f)
            i_his_list = pickle.load(f)
            ucs_count, cs_count, item_count = pickle.load(f)
    except:
        with open(datadir, 'rb') as f:
            ucs_set = pickle.load(f)
            cs_set = pickle.load(f)
            ucs_count, cs_count, item_count = pickle.load(f)

    user_supp_num, user_que_num = ucs_count, cs_count
    train_set_supp, test_set_supp = [], []
    train_set_que, test_set_que = [], []

    user_supp_list = [u for u in range(ucs_count)]

    for u in range(len(ucs_set)):
        train_set_supp += ucs_set[u][:-10]
        test_set_supp += ucs_set[u][-10:]

    for u in range(len(cs_set)):
        train_set_que += cs_set[u][:-10]
        test_set_que += cs_set[u][-10:]

    user_his_dic = {}
    for u in range(ucs_count):
        tmp = ucs_set[u][:-10]
        user_his_dic[u] = [ tmp[k][1] for k in range(len(tmp))]
    for u in range(0, cs_count):
        tmp = cs_set[u][:-10]
        user_his_dic[u+ucs_count] = [ tmp[k][1] for k in range(len(tmp))]

    train_uir = train_set_supp + train_set_que
    
    edge_array = np.array(train_uir, dtype=np.int32)
    
    if sample_graph:
        edge_UI = np.transpose(edge_array[:, :2], (1, 0))
    else:
        edge_UI = np.zeros((n_user, n_item), dtype=np.int)
        edge_UI[edge_array[:, 0], edge_array[:, 1]] = 1

    print("-------Dataset Info--------")
    print("support user {}, query user {}".format(user_supp_num, user_que_num))
    print("train set size: support/query {}/{}".format(len(train_set_supp), len(train_set_que)))
    print("test set size: support/query {}/{}".format(len(test_set_supp), len(test_set_que)))

    #print(user_neighbor_dic[0], user_neighbor_dic[1000])

    if sample_graph:
        '''
        with open('./ml-10m-preprocess.pkl', 'wb') as f:
            pickle.dump(train_set_supp, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(train_set_que, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(test_set_supp, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(test_set_que, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(user_supp_list, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(edge_array, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(user_din, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(item_din, f, pickle.HIGHEST_PROTOCOL)
        '''
        return train_set_supp, train_set_que, test_set_supp, test_set_que, user_his_dic, user_supp_list, edge_UI
    else:
        return train_set_supp, train_set_que, test_set_supp, test_set_que, user_his_dic, user_supp_list, edge_UI

def auc_calc(score_label):
    fp1, tp1, fp2, tp2, auc = 0.0, 0.0, 0.0, 0.0, 0.0
    for s in score_label:
        fp2 += (1-s[1]) # noclick
        tp2 += s[1] # click
        auc += (tp2 - tp1) * (fp2 + fp1) / 2
        fp1, tp1 = fp2, tp2
    try:
        return 1 - auc / (tp2 * fp2)
    except:
        return 0.5

def recall_calc(score_label):
	num, num_tp = 0, 0
	for s in score_label:
		if s[1] == 1:
			num += 1
			if s[0] >= 0.5:
				num_tp += 1
	return num_tp / num

def precision_calc(score_label):
	num, num_tp = 0, 0
	for s in score_label:
		if s[0] >= 0.5:
			num += 1
			if s[1] == 1:
				num_tp += 1
	return num_tp / num

def dcg_k(score_label, k):
    dcg, i = 0., 0
    for s in score_label:
        if i < k:
            dcg += (2**s[1]-1) / np.log2(2+i)
            i += 1
    return dcg

def ndcg_k(y_hat, y, k):
    score_label = np.stack([y_hat, y], axis=1).tolist()
    score_label = sorted(score_label, key=lambda d:d[0], reverse=True)
    score_label_ = sorted(score_label, key=lambda d:d[1], reverse=True)
    norm, i = 0., 0
    for s in score_label_:
        if i < k:
            norm += (2**s[1]-1) / np.log2(2+i)
            i += 1
    dcg = dcg_k(score_label, k)
    return dcg / norm
    