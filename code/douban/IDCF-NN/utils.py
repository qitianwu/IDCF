import pickle
import random
import yaml
import numpy as np

config = yaml.safe_load(open("./datainfo.yaml", 'r'))

def generate_data(datadir, dataset='ml-1m', split_way='threshold', threshold=50, supp_ratio=None, training_ratio=1):
    n_user = config[dataset]['n_user']
    n_item = config[dataset]['n_item']
    with open (datadir+dataset+'.pkl','rb') as f:  
        u = pickle.load(f)
        i = pickle.load(f)
        r = pickle.load(f)
        train_u = pickle.load(f)
        train_i = pickle.load(f)
        train_r = pickle.load(f)
        test_u = pickle.load(f)
        test_i = pickle.load(f)
        test_r = pickle.load(f)

    print(len(train_u)/len(u))

    
    index = [i for i in range(len(u))]
    random.shuffle(index)
    train_index, test_index = index[:int(0.9*len(u))], index[int(0.9*len(u)):]
    

    train_ui_dic = {}
    train_ur_dic = {}

    test_ui_dic = {}
    test_ur_dic = {}

    for user in range(n_user):
        train_ui_dic[user] = []
        train_ur_dic[user] = []
        test_ui_dic[user] = []
        test_ur_dic[user] = []

    
    for k in range(len(train_u)):
        train_ui_dic[train_u[k]].append(train_i[k])
        train_ur_dic[train_u[k]].append(train_r[k])

    for k in range(len(test_u)):
        test_ui_dic[test_u[k]].append(test_i[k])
        test_ur_dic[test_u[k]].append(test_r[k])


    user_supp_num, user_que_num = 0, 0
    train_set_supp, test_set_supp = [], []
    train_set_que, test_set_que = [], []
    test_set_supp_size, test_set_que_size = 0, 0

    user_supp_list = []

    if split_way == 'threshold':
        for u in train_ui_dic.keys():
            num = len(train_ui_dic[u])
            if num >= threshold:
                for index, i in enumerate(train_ui_dic[u]):
                    train_set_supp.append([u, i, train_ur_dic[u][index]])
                test_set_supp_u = []
                for index, i in enumerate(test_ui_dic[u]):
                    test_set_supp.append([u, i, test_ur_dic[u][index]])
                user_supp_num += 1
                user_supp_list.append(u)
            else:
                for index, i in enumerate(train_ui_dic[u]):
                    train_set_que.append([u, i, train_ur_dic[u][index]])
                test_set_que_u = []
                for index, i in enumerate(test_ui_dic[u]):
                    test_set_que.append([u, i, test_ur_dic[u][index]])
                user_que_num += 1
    
    if split_way == 'random':
        for u in train_ui_dic.keys():
            r = random.uniform(0, 1)
            if r <= supp_ratio:
                for index, i in enumerate(train_ui_dic[u]):
                    train_set_supp.append([u, i, train_ur_dic[u][index]])
                test_set_supp_u = []
                for index, i in enumerate(test_ui_dic[u]):
                    test_set_supp.append([u, i, test_ur_dic[u][index]])
                user_supp_num += 1
                user_supp_list.append(u)
            else:
                for index, i in enumerate(train_ui_dic[u]):
                    train_set_que.append([u, i, train_ur_dic[u][index]])
                test_set_que_u = []
                for index, i in enumerate(test_ui_dic[u]):
                    test_set_que.append([u, i, test_ur_dic[u][index]])
                user_que_num += 1

    if split_way == 'all':
        for u in train_ui_dic.keys():
            num = len(train_ui_dic[u])
            if num >= threshold:
                for index, i in enumerate(train_ui_dic[u]):
                    train_set_supp.append([u, i, train_ur_dic[u][index]])
                test_set_supp_u = []
                for index, i in enumerate(test_ui_dic[u]):
                    test_set_supp.append([u, i, test_ur_dic[u][index]])
                user_supp_num += 1
                user_supp_list.append(u)
            else:
                for index, i in enumerate(train_ui_dic[u]):
                    train_set_que.append([u, i, train_ur_dic[u][index]])
                test_set_que_u = []
                for index, i in enumerate(test_ui_dic[u]):
                    test_set_que.append([u, i, test_ur_dic[u][index]])
                user_que_num += 1
                user_supp_list.append(u)
    
    user_his_dic = {}
    for u in train_ui_dic.keys():
        user_his_dic[u] = train_ui_dic[u]



    print("-------Dataset Info--------")
    if split_way == 'threshold':
        print("split way [threshold] with threshold {} training_ratio {}".format(threshold, training_ratio))
    if split_way == 'random':
        print("split way [random] with supp_ratio {} training_ratio {}".format(supp_ratio, training_ratio))
    if split_way == 'all':
        print("split way [all] with threshold {} training_ratio {}".format(threshold, training_ratio))
    print("support user {}, query user {}".format(user_supp_num, user_que_num))
    print("train set size: support/query {}/{}".format(len(train_set_supp), len(train_set_que)))
    print("test set size: support/query {}/{}".format(len(test_set_supp), len(test_set_que)))

    return train_set_supp, train_set_que, test_set_supp, test_set_que, user_supp_list, user_his_dic

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