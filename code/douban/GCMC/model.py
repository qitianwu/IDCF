import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
	def __init__(self, n_user, n_item, embedding_size=32):
		super(Embedding, self).__init__()
		self.user_embedding = nn.Parameter(torch.Tensor(n_user, embedding_size))
		self.item_embedding = nn.Parameter(torch.Tensor(n_item, embedding_size))

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.xavier_uniform_(self.user_embedding)
		nn.init.xavier_uniform_(self.item_embedding)

	def regularization_loss(self):
		loss_reg = 0.
		loss_reg += torch.sum( torch.sqrt( torch.sum(self.user_embedding ** 2, 1) ) )
		loss_reg += torch.sum( torch.sqrt( torch.sum(self.item_embedding ** 2, 1) ) )

		return loss_reg

class GCMCModel(nn.Module):
	def __init__(self, n_user, n_item, n_rating, device, edge_sparse=None, embedding_size=32, hidden_size=32):
		super(GCMCModel, self).__init__()
		self.n_user = n_user
		self.n_item = n_item
		self.n_rating = n_rating
		self.device = device
		self.edge_sparse = edge_sparse
		self.embedding_model = Embedding(n_user, n_item, embedding_size)
		self.user_embedding = self.embedding_model.user_embedding
		self.item_embedding = self.embedding_model.item_embedding

		self.GCN_user = nn.Linear(embedding_size, embedding_size)
		self.GCN_item = nn.Linear(embedding_size, embedding_size)
		
		self.fc1_user = nn.Linear(n_rating * embedding_size, embedding_size)
		self.fc1_item = nn.Linear(n_rating * embedding_size, embedding_size)

		self.l1 = nn.Linear(embedding_size*4, hidden_size*2)
		self.l2 = nn.Linear(hidden_size*2, hidden_size)
		self.l3 = nn.Linear(hidden_size, 1)

		self.user_bias = nn.Parameter(torch.Tensor(n_user, 1))
		self.item_bias = nn.Parameter(torch.Tensor(n_item, 1))	

		self.reset_parameters()
	
	def reset_parameters(self):
		nn.init.zeros_(self.user_bias)
		nn.init.zeros_(self.item_bias)

	def regularization_loss(self):
		return self.embedding_model.regularization_loss()

	def forward(self, x,  edge_UI = None, edge_IU = None, sample_graph = False):
		user_id = x[:, 0]
		item_id = x[:, 1]
        
		user_emb = self.user_embedding[user_id]
		item_emb = self.item_embedding[item_id]

		user_h = self.user_embedding
		item_h = self.item_embedding

		
		for n in range(self.n_rating):
			if sample_graph:
				edge_index_ui_n = self.edge_sparse[self.edge_sparse[:,2]==n][:, :2].transpose(1,0)
				edge_index_iu_n = edge_index_ui_n[[1,0], :]
				#if self.training: # for dropout sparse edge matrix
				#	edge_num_n = edge_index_ui_n.size(1)
				#	edge_index_ui_n = edge_index_ui_n[:, torch.randperm(edge_num_n)][:, :int(0.7*edge_num_n)]
				#	edge_index_iu_n = edge_index_iu_n[:, torch.randperm(edge_num_n)][:, :int(0.7*edge_num_n)]
				edge_ui_n = torch.sparse_coo_tensor(edge_index_ui_n, torch.ones(edge_index_ui_n.size(1)), size=torch.Size([self.n_user, self.n_item])).to(self.device)
				edge_iu_n = torch.sparse_coo_tensor(edge_index_iu_n, torch.ones(edge_index_iu_n.size(1)), size=torch.Size([self.n_item, self.n_user])).to(self.device)
				gcn_item_h_n = torch.sparse.mm(edge_ui_n, item_h)[user_id]
				gcn_user_h_n = torch.sparse.mm(edge_iu_n, user_h)[item_id]
				item_din = torch.sparse.mm(edge_ui_n, torch.ones(self.n_item, 1).to(self.device))[user_id] + 1
				user_din = torch.sparse.mm(edge_iu_n, torch.ones(self.n_user, 1).to(self.device))[item_id] + 1
				gcn_item_h_n = gcn_item_h_n / item_din
				gcn_user_h_n = gcn_user_h_n / user_din
				gcn_item_h_n = F.dropout(gcn_item_h_n, p=0.3, training=self.training)
				gcn_user_h_n = F.dropout(gcn_user_h_n, p=0.3, training=self.training)
			else:
				edge_UI_n = edge_UI[n].float()
				edge_IU_n = edge_IU[n].float()
				edge_UI_n = F.dropout(edge_UI_n, p=0.3, training=self.training)
				edge_IU_n = F.dropout(edge_IU_n, p=0.3, training=self.training)
				gcn_user_h_n = torch.matmul(edge_IU_n, user_h)
				gcn_item_h_n = torch.matmul(edge_UI_n, item_h)
			gcn_user_h_n = torch.relu(self.GCN_user(gcn_user_h_n))
			gcn_item_h_n = torch.relu(self.GCN_item(gcn_item_h_n))
			if n <= 0:
				gcn_user_h = gcn_user_h_n
				gcn_item_h = gcn_item_h_n
			else:
				gcn_user_h = torch.cat([gcn_user_h, gcn_user_h_n], dim=-1)
				gcn_item_h = torch.cat([gcn_item_h, gcn_item_h_n], dim=-1)

		gcn_user_output = self.fc1_user(gcn_user_h)
		gcn_item_output = self.fc1_item(gcn_item_h)
		
		interaction1 = torch.mul(user_emb, item_emb)
		interaction2 = torch.mul(user_emb, gcn_item_output)
		interaction3 = torch.mul(gcn_user_output, item_emb)
		interaction4 = torch.mul(gcn_user_output, gcn_item_output)
		x = torch.cat([interaction1, interaction2, interaction3, interaction4], dim=-1)
		x1 = torch.relu(self.l1(x))
		x2 = torch.relu(self.l2(x1))
		x3 = self.l3(x2).reshape(-1)

		user_b = self.user_bias[user_id].reshape(-1)
		item_b = self.item_bias[item_id].reshape(-1)

		output = x3 + user_b + item_b

		return output

	def load_model(self, path):
		model_dict = torch.load(path)
		self.load_state_dict(model_dict)

	def load_embedding(self, path):
		pretrained_dict = torch.load(path)
		model_dict = self.embedding_model.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		self.embedding_model.load_state_dict(model_dict)