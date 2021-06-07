import pickle

# datadir = '/home/shiliangliang/Qitian/MetaCF/data'
#
# # amazon_dataset, Gowalla_dataset, ml-1m_dataset
#
# with open(datadir+'/amazon_dataset.pkl', 'rb') as f:
# 	matrix = pickle.load(f)
# 	user_count, item_count = pickle.load(f)
#
# print(matrix[:3])
#
# print(len(matrix))
#
# print(user_count, item_count)

datadir = '../../data/split_1.mat'

import scipy.io as scio
import h5py
import numpy as np

data = h5py.File(datadir)
print(data.keys())
print(data['Otraining'])
print(data['Otest'])
print(data['M'])
M_data = np.transpose(data['M'])
train_data = np.transpose(data['Otraining'])
test_data = np.transpose(data['Otest'])
u, i, r = [], [], []
train_u, train_i, train_r = [], [], []
test_u, test_i, test_r = [], [], []

indice_M = M_data.nonzero()
for k in range(len(indice_M[0])):
	uk, ik = indice_M[0][k], indice_M[1][k]
	u.append(uk)
	i.append(ik)
	r.append(M_data[uk, ik])

indice_train = train_data.nonzero()
for k in range(len(indice_train[0])):
	uk, ik = indice_train[0][k], indice_train[1][k]
	train_u.append(uk)
	train_i.append(ik)
	train_r.append(M_data[uk, ik])

indice_test = test_data.nonzero()
for k in range(len(indice_test[0])):
	uk, ik = indice_test[0][k], indice_test[1][k]
	test_u.append(uk)
	test_i.append(ik)
	test_r.append(M_data[uk, ik])

print(len(u), len(train_u), len(test_u))
print(u[:10], i[:10], r[:10])
print(M_data[0, :20])
print(train_u[:10], train_i[:10], train_r[:10])
print(test_u[:10], test_i[:10], test_r[:10])


with open('/mnt/nas/home/wuqitian/IDCF/data/ml-100k.pkl', 'wb') as f:
	pickle.dump(u, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(i, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(r, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_u, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_i, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(train_r, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_u, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_i, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_r, f, pickle.HIGHEST_PROTOCOL)