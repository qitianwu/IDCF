import csv
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm


def build_map(df, col_name):
  key = sorted(df[col_name].unique().tolist())
  m = dict(zip(key, range(len(key))))
  df[col_name] = df[col_name].map(lambda x: m[x])
  return m, key

workdir = '/home/shiliangliang/Qitian/MetaCF/data'
savedir = '/home/shiliangliang/Qitian/MetaCF/data'

csv_reader = csv.reader(open(workdir+'/Books.csv', 'r'))
#csv_reader = pd.read_csv(workdir+'/Baby.csv', encoding='ANSI')

rate_list = [] #selected users and items
rate_list_ = [] #all users and items
for s in csv_reader:
  uid = s[0]
  iid = s[1]
  rate = s[2]
  try:
    time = int(s[3])
  except:
    continue
  rate_list_.append([uid, iid, rate, time])
rate_df_ = pd.DataFrame(rate_list_, columns=['uid', 'iid', 'rate', 'time'])
mintime = rate_df_['time'].min()
print(rate_df_.head())

uid_map, uid_key = build_map(rate_df_, 'uid')
iid_map, iid_key = build_map(rate_df_, 'iid')
print(rate_df_.head())
user_count_, item_count_, example_count_ =\
    len(uid_map), len(iid_map), rate_df_.shape[0]
print('Raw Statistics: user_count: %d\titem_count: %d\texample_count: %d' %
      (user_count_, item_count_, example_count_))

rate_list_ = []
u_list = rate_df_['uid'].tolist()
i_list = rate_df_['iid'].tolist()
r_list = rate_df_['rate'].tolist()
for i in range(len(u_list)):
  s = [u_list[i], i_list[i], r_list[i]]
  rate_list_.append(s)

item_num = np.zeros(user_count_)
user_num = np.zeros(item_count_)
for s in rate_list_:
  uid = s[0]
  iid = s[1]
  item_num[uid] = item_num[uid] + 1
  user_num[iid] = user_num[iid] + 1
for s in rate_list_:
  uid = s[0]
  iid = s[1]
  rate = s[2]
  #time = s[3] - mintime
  if item_num[uid]>=6 and user_num[iid]>=5:
    rate_list.append([uid, iid, rate])
rate_df = pd.DataFrame(rate_list, columns = ['uid', 'iid', 'rate'])

uid_map, uid_key = build_map(rate_df, 'uid')
iid_map, iid_key = build_map(rate_df, 'iid')
#rate_df = rate_df[rate_df['uid']<10000]
#rate_df = rate_df[rate_df['iid']<10000]
user_count = max(rate_df['uid'].tolist()) + 1
item_count = max(rate_df['iid'].tolist()) + 1
example_count = rate_df.shape[0]
print(rate_df.head())

print('New Statistics: user_count: %d\titem_count: %d\texample_count: %d' %
      (user_count, item_count, example_count))
'''
with open(workdir + '/Digital_remap_raw.pkl', 'wb') as f:
  pickle.dump(rate_df_, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count_, item_count_, example_count_), f, pickle.HIGHEST_PROTOCOL)
'''
with open(savedir + '/Books_remap.pkl', 'wb') as f:
  pickle.dump(rate_df, f, pickle.HIGHEST_PROTOCOL)
  pickle.dump((user_count, item_count, example_count), f, pickle.HIGHEST_PROTOCOL)
