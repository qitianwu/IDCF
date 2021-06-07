import pickle
import random
import numpy as np

def get_list(int_list, uid):
    output_list = []
    for rec in int_list:
        if rec[0]==uid:
            output_list.append(rec)
    return output_list

def reindex(int_list, start_index=0):
    for i in range(len(int_list)):
        for k in range(len(int_list[i])):
            int_list[i][k][0] = i+start_index
    return int_list

def build_dataset(datadir, ucs_threshold=20):

    with open(datadir, 'rb') as f:
        matrix = pickle.load(f)
        user_count, item_count, example_count = pickle.load(f)

    u = 0
    u_int_list_u, u_int_list = [], []
    for rec in matrix:
        if rec[0]==u:
            u_int_list_u.append(rec)
        else:
            u_int_list.append(u_int_list_u)
            u += 1
            u_int_list_u = []
    u_int_list.append(u_int_list_u)
    
    random.shuffle(u_int_list)

    #assert(len(u_int_list) == user_count)

    ucs_int_list = []
    cs_int_list = []
    for u in range(len(u_int_list)):
        u_int_list_u = u_int_list[u]
        random.shuffle(u_int_list_u)
        if len(u_int_list_u) >= ucs_threshold:
            ucs_int_list.append(u_int_list_u)
        else:
            cs_int_list.append(u_int_list_u)

    ucs_count, cs_count = len(ucs_int_list), len(cs_int_list)
    ucs_int_list = reindex(ucs_int_list, 0)
    cs_int_list = reindex(cs_int_list, ucs_count)

    print(ucs_count, cs_count, user_count, item_count)
    print(ucs_int_list[:3])
    print(cs_int_list[:3])

    return ucs_int_list, cs_int_list, ucs_count, cs_count, user_count, item_count

def build_dataset2(datadir, ucs_threshold):
    data = np.loadtxt(datadir,
                      dtype=np.int32)

    user_count, item_count = 0, 0
    u_int_list_u, u_int_list = [], []
    for rec in data:
        u_k, i_k = rec[0]-1, rec[1]-1
        if u_k == user_count:
            u_int_list_u.append([u_k, i_k, 1])
        else:
            u_int_list.append(u_int_list_u)
            user_count += 1
            u_int_list_u = []
        if i_k > item_count:
            item_count = i_k

    user_count += 1
    item_count += 1
    u_int_list.append(u_int_list_u)

    random.shuffle(u_int_list)

    # assert(len(u_int_list) == user_count)

    ucs_int_list = []
    cs_int_list = []
    for u in range(len(u_int_list)):
        u_int_list_u = u_int_list[u]
        if len(u_int_list_u) <= 15:
            continue
        random.shuffle(u_int_list_u)
        if len(u_int_list_u) >= ucs_threshold:
            ucs_int_list.append(u_int_list_u)
        else:
            cs_int_list.append(u_int_list_u)

    ucs_count, cs_count = len(ucs_int_list), len(cs_int_list)
    ucs_int_list = reindex(ucs_int_list, 0)
    cs_int_list = reindex(cs_int_list, ucs_count)

    print(ucs_count, cs_count, user_count, item_count)
    print(ucs_int_list[:3])
    print(cs_int_list[:3])

    return ucs_int_list, cs_int_list, ucs_count, cs_count, user_count, item_count

if __name__ == "__main__":
    dataset = 'beauty'
    # datadir = '/home/shiliangliang/Qitian/MetaCF/data/' + dataset + '_remap.pkl'
    datadir = '../../data/Beauty.txt'

    ucs_int_list, cs_int_list, ucs_count, cs_count, user_count, item_count = build_dataset2(datadir, 30)

    with open('../../data/' + dataset + '_s20.pkl', 'wb') as f:
	    pickle.dump(ucs_int_list, f, pickle.HIGHEST_PROTOCOL)
	    pickle.dump(cs_int_list, f, pickle.HIGHEST_PROTOCOL)
	    pickle.dump((ucs_count, cs_count, item_count), f, pickle.HIGHEST_PROTOCOL)




        
