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
	def __init__(self, n_user, n_item, embedding_size=32, hidden_size=64):
		super(NNMFModel, self).__init__()
		self.embedding_model = Embedding(n_user, n_item, embedding_size)
		self.user_embedding = self.embedding_model.user_embedding
		self.item_embedding = self.embedding_model.item_embedding
		
		self.l1 = nn.Linear(embedding_size*3, hidden_size)
		self.l2 = nn.Linear(hidden_size, hidden_size)
		self.l3 = nn.Linear(hidden_size, 1)

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
