import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
	def __init__(self, n_user, n_item, embedding_size=32, requires_grad = True):
		super(Embedding, self).__init__()
		self.user_embedding = nn.Parameter(torch.Tensor(n_user, embedding_size), requires_grad = requires_grad)
		self.item_embedding = nn.Parameter(torch.Tensor(n_item, embedding_size), requires_grad = requires_grad)

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.xavier_uniform_(self.user_embedding)
		nn.init.xavier_uniform_(self.item_embedding)

	def regularization_loss(self):
		loss_reg = 0.
		loss_reg += torch.sum( torch.sqrt( torch.sum(self.user_embedding ** 2, 1) ) )
		loss_reg += torch.sum( torch.sqrt( torch.sum(self.item_embedding ** 2, 1) ) )

		return loss_reg

class NNMFModel(nn.Module):
	def __init__(self, n_user, n_item, embedding_size=16, hidden_size=32):
		super(NNMFModel, self).__init__()
		self.embedding_model = Embedding(n_user, n_item, embedding_size)
		self.user_embedding = self.embedding_model.user_embedding
		self.item_embedding = self.embedding_model.item_embedding
		
		
		self.l1 = nn.Linear(embedding_size*3, hidden_size)
		self.l2 = nn.Linear(hidden_size, hidden_size // 2)
		self.l3 = nn.Linear(hidden_size // 2, 1)
		

		self.user_bias = nn.Parameter(torch.Tensor(n_user, 1))
		self.item_bias = nn.Parameter(torch.Tensor(n_item, 1))		

		self.reset_parameters()
	
	def reset_parameters(self):
		nn.init.zeros_(self.user_bias)
		nn.init.zeros_(self.item_bias)

	def regularization_loss(self):
		return self.embedding_model.regularization_loss()

	def forward(self, x):
		user_id = x[:, 0]
		item_id = x[:, 1]
        
		user_emb = self.user_embedding[user_id]
		item_emb = self.item_embedding[item_id]

		interaction = torch.mul(user_emb, item_emb)
		ratings = torch.sum(interaction, dim = 1)

		
		x = torch.cat([user_emb, item_emb, interaction], dim=1)
		x1 = torch.tanh(self.l1(x))
		x2 = torch.tanh(self.l2(x1))
		x3 = self.l3(x2).reshape(-1)

		user_b = self.user_bias[user_id].reshape(-1)
		item_b = self.item_bias[item_id].reshape(-1)

		output = (ratings + x3) / 2. + user_b + item_b

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

class RelationGAT(nn.Module):
	def __init__(self, in_size, out_size):
		super(RelationGAT, self).__init__()
		self.wq = nn.Linear(in_size, out_size, bias = False)
		self.wk = nn.Linear(in_size, out_size, bias = False)
		self.wv = nn.Linear(in_size, out_size, bias = False)

		self.reset_parameters()

	def reset_parameters(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight.data)

	def forward(self, x, neighbor):
		x = self.wq(x).unsqueeze(1)
		neighbor = self.wk(neighbor)
		gat_input = torch.sum(
			torch.mul(x.repeat(1, neighbor.size(1), 1), neighbor), dim=2
		)
		attn = F.softmax(gat_input, dim=1)
		neighbor = neighbor.transpose(1, 2).contiguous()
		gat_output = self.wv(
			torch.matmul(neighbor, attn.unsqueeze(2)).squeeze(2)
		)
		return gat_output

class IRMC_NN_Model(nn.Module):
	def __init__(self, n_user, n_item, supp_users, device, 
			embedding_size = 16, 
			out_size = None, 
			hidden_size = 32, 
			head_num = 4, 
			sample_num = 200):
		super(IRMC_NN_Model, self).__init__()
		self.device = device
		self.supp_users = supp_users
		self.supp_user_num = supp_users.size(0)
		self.head_num = head_num
		self.sample_num = sample_num
		self.GAT_unit = nn.ModuleList()
		if out_size is None:
			out_size = embedding_size
		for i in range(head_num):
			self.GAT_unit.append(RelationGAT(embedding_size, out_size))
		self.w_out = nn.Linear(out_size * head_num, out_size, bias = False)

		self.embedding_model = Embedding(n_user, n_item, embedding_size, requires_grad=False)
		self.user_embedding = self.embedding_model.user_embedding
		self.item_embedding = self.embedding_model.item_embedding
		
		self.l1 = nn.Linear(embedding_size*3, hidden_size)
		self.l2 = nn.Linear(hidden_size, hidden_size // 2)
		self.l3 = nn.Linear(hidden_size // 2, 1)

		self.user_bias = nn.Parameter(torch.Tensor(n_user, 1), requires_grad = True)
		self.item_bias = nn.Parameter(torch.Tensor(n_item, 1), requires_grad = False)

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.zeros_(self.user_bias)
		nn.init.zeros_(self.item_bias)
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_normal_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.1)

	def forward(self, x, history, history_len, mode='INTER'):
		user_id = x[:, 0]
		item_id = x[:, 1]

		mask = torch.arange(history.size(1))[None, :].to(self.device)
		mask = mask < history_len[:, None]
		history_emb = self.item_embedding[history]
		history_emb[~mask] = torch.zeros(self.item_embedding.size(1)).to(self.device)
		user_init_emb = torch.sum( self.item_embedding[history], dim=1)
		user_init_emb /= history_len[:, None].float()
		
		for i in range(self.head_num):
			if self.training:
				sample_index = torch.randint(0, self.supp_user_num, (x.size(0), self.sample_num)).to(self.device)
			else:
				sample_index = torch.arange(0, self.supp_user_num).unsqueeze(0).repeat(x.size(0), 1)
			sample_users = self.supp_users[sample_index]
			sample_user_emb = self.user_embedding[sample_users]
			gat_output_i = self.GAT_unit[i](user_init_emb, sample_user_emb)
			if i == 0:
				gat_output = gat_output_i
			else:
				gat_output = torch.cat([gat_output, gat_output_i], dim=1)
		user_emb = self.w_out(gat_output)
        
		item_emb = self.item_embedding[item_id]

		interaction = torch.mul(user_emb, item_emb)
		ratings = torch.sum(interaction, dim = 1)

		x = torch.cat([user_emb, item_emb, interaction], dim=1)
		x1 = torch.tanh(self.l1(x))
		x2 = torch.tanh(self.l2(x1))
		x3 = self.l3(x2).reshape(-1)

		user_b = self.user_bias[user_id].reshape(-1)
		item_b = self.item_bias[item_id].reshape(-1)

		output = (ratings + x3) / 2. + user_b + item_b

		if mode == 'EXTRA':
			user_emb_trd = self.user_embedding[user_id]
			return output, user_emb, user_emb_trd
		else:
			return output
	
	def embedding_lookup(self, x):
		return self.user_embedding[x]

	def load_model(self, path):
		model_dict = torch.load(path)
		self.load_state_dict(model_dict)

	def load_embedding(self, path):
		pretrained_dict = torch.load(path)
		model_dict = self.embedding_model.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		self.embedding_model.load_state_dict(model_dict)

	def load_embedding_nn(self, path):
		pretrained_dict = torch.load(path)
		model_dict = self.state_dict()
		pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
		model_dict.update(pretrained_dict)
		self.load_state_dict(model_dict, strict=False)

			


