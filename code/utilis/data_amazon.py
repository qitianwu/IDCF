import random
import pickle
import numpy as np
import pandas as pd

random.seed(1234)

USER_HIST_MIN = 13# filter out threshold
USER_COLD_THRE = 20 # cold user threshold

def rate_map(rate):
	rate = float(rate)
	'''
	if rate > 3.0:
		return 1.0
	else:
		return 0.0
	'''
	return rate

def build_map(uid_list):
  key = uid_list
  m = dict(zip(key, range(len(key))))
  return m

def reindex(int_list, start_index=0):
    for i in range(len(int_list)):
        for k in range(len(int_list[i])):
            int_list[i][k][0] = i+start_index
    return int_list
 
workdir = '/home/shiliangliang/Qitian/MetaCF/data'
with open(workdir+'/Books_remap.pkl', 'rb') as f:
	rate_df = pickle.load(f)
	user_count, item_count, example_count = pickle.load(f)

u_his_list, i_his_list = [], []

ucs_set, cs_set = [], [] 
ucs_count, cs_count = 0, 0

#time_delta = 0.9 * max(rate_df['time'].tolist())
#train_df = rate_df[rate_df['time']<=time_delta]
#test_df = rate_df[rate_df['time']>time_delta]

rate_list_ = []
u_list = rate_df['uid'].tolist()
i_list = rate_df['iid'].tolist()
r_list = rate_df['rate'].tolist()
u_his_num = np.zeros(user_count)

for i in range(len(u_list)):
	u = u_list[i]
	u_his_num[u] += 1

r_dict = {}
for u in range(user_count):
	r_dict[u] = []

for i in range(len(u_list)):
	u, i, r = u_list[i], i_list[i], rate_map(r_list[i])
	tmp = r_dict[u]
	tmp.append([u,i,r])
	r_dict[u] = tmp

ucs_set, cs_set = [], []
ucs_r_num, cs_r_num = 0, 0
for u in range(user_count):
	hist_num = u_his_num[u]
	r_u = r_dict[u]
	if hist_num < USER_HIST_MIN and hist_num > 100:
		continue
	if hist_num > USER_COLD_THRE: # ucs
		ucs_set.append(r_u)
		ucs_count += 1
		ucs_r_num += hist_num
	else: # cs
		cs_set.append(r_u)
		cs_count += 1
		cs_r_num += hist_num

ucs_set = reindex(ucs_set, 0)
cs_set = reindex(cs_set, ucs_count)

cs_set[:10000]


#user_count = user_ucs_count+user_scs_count+user_tcs_count
print('UCS example: ', ucs_r_num, 'CS example: ', cs_r_num)
print('User count: ', user_count, 'Item count: ', item_count)
print('UCS user: ', ucs_count, 'CS user: ', cs_count)

print(ucs_set[:3])
print(cs_set[:3])


with open(workdir+'/Books_dataset.pkl', 'wb') as f:
	pickle.dump(ucs_set, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(cs_set, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump((ucs_count, cs_count, item_count), f, pickle.HIGHEST_PROTOCOL)


