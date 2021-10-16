import torch
import numpy as np
from torch._C import dtype

class HeteroGraph():
	""" The Heterogenous Graph Representation
	How to use:
		1. graph = HeteroGraph(types=3)
		2. A = graph.get_adjacency()
		3. A = code to modify A
		4. normalized_A = graph.normalize_adjacency(A)
	"""
	def __init__(self, num_node=120, num_hetero_types=3, **kwargs):
		self.num_node = num_node
		self.num_hetero_types = num_hetero_types

	def get_adjacency_batch(self, relation_matrix):
		"""
		relation_matrix is shape of (batch_size, num_obj, num_obj)
		relation_matrix[i,j,k] is the relation type between the j-th node and k-th node in the i-th example.
		"""
		adjacency = (relation_matrix > 0).float()
		self.relation_matrix = relation_matrix
		return adjacency

	def normalize_adjacency_batch(self, A):
		Dl = A.sum(axis=1)
		Dl[Dl == 0] = float('inf')
		# Dn = torch.zeros_like(A, dtype=torch.float32)
		# batch_indices, obj_indices = torch.where(Dl > 0)
		# Dn[batch_indices, obj_indices, obj_indices] = Dl.view(-1) ** (-1)
		AD = A / Dl.unsqueeze(1) # (batch_size, object_num, object_num) / (batch_size, 1, object_num) -> (batch_size, object_num, object_num)
		A_normalized = torch.zeros((A.size(0), self.num_hetero_types, self.num_node, self.num_node))
		for i, type_ in enumerate(range(1, self.num_hetero_types + 1)):
			A_normalized[:, i][self.relation_matrix == type_] = AD[self.relation_matrix == type_]

		return A_normalized

	def get_adjacency(self, relation_matrix):
		"""
		relation_matrix[i,j] is the relation type between the i-th node and j-th node.
		"""
		assert int(relation_matrix.max()) <= self.num_hetero_types and int(relation_matrix.min()) >= 0
		adjacency = (relation_matrix > 0).astype(float)
		self.relation_matrix = relation_matrix
		return adjacency
	
	def normalize_adjacency(self, A):
		Dl = np.sum(A, 0)
		Dn = np.zeros((self.num_node, self.num_node))
		for i in range(self.num_node):
			if Dl[i] > 0:
				Dn[i, i] = Dl[i] ** (-1)
		AD = np.dot(A, Dn)

		A = np.zeros((self.num_hetero_types, self.num_node, self.num_node))
		for i, type_ in enumerate(range(1, self.num_hetero_types+1)):
			A[i][self.relation_matrix == type_] = AD[self.relation_matrix == type_]
		
		return A



class Graph():
	""" The Graph Representation
	How to use:
		1. graph = Graph(max_hop=1)
		2. A = graph.get_adjacency()
		3. A = code to modify A
		4. normalized_A = graph.normalize_adjacency(A)
	"""
	def __init__(self,
				 num_node=120,
				 max_hop=1,
				 **kwargs):
		self.max_hop = max_hop
		self.num_node = num_node 

	def get_adjacency_batch(self, A):
		"""
		A is shape of (batch_size, num_node, num_node)
		"""
		transfer_mat = [torch.matrix_power(A, d) for d in range(self.max_hop + 1)]
		arrive_mat = torch.stack(transfer_mat, dim=1) > 0 # (batch_size, max_hop+1, num_node, num_node)
		self.hop_dis = torch.zeros((A.size(0), self.num_node, self.num_node)) + np.inf
		for d in range(self.max_hop, -1, -1):
			self.hop_dis[arrive_mat[:, d]] = d # when d = 0, all self-connections will be set to 0
		adjacency = (self.hop_dis <= self.max_hop).float()
		return adjacency

	def normalize_adjacency_batch(self, A):
		Dl = A.sum(axis=1)
		Dl[Dl == 0] = float('inf')
		# Dn = torch.zeros_like(A, dtype=torch.float32)
		# batch_indices, obj_indices = torch.where(Dl > 0)
		# Dn[batch_indices, obj_indices, obj_indices] = Dl.view(-1) ** (-1)
		AD = A / Dl.unsqueeze(1) # (batch_size, object_num, object_num) / (batch_size, 1, object_num) -> (batch_size, object_num, object_num)
		A_normalized = torch.zeros((A.size(0), self.max_hop+1, self.num_node, self.num_node))
		for i, type_ in enumerate(range(0, self.max_hop + 1)):
			A_normalized[:, i][self.hop_dis == type_] = AD[self.hop_dis == type_]

		return A_normalized

	def get_adjacency(self, A):
		# compute hop steps
		self.hop_dis = np.zeros((self.num_node, self.num_node)) + np.inf
		transfer_mat = [np.linalg.matrix_power(A, d) for d in range(self.max_hop + 1)]
		arrive_mat = (np.stack(transfer_mat) > 0)
		for d in range(self.max_hop, -1, -1):
			self.hop_dis[arrive_mat[d]] = d

		# compute adjacency
		adjacency = (self.hop_dis <= self.max_hop).astype(float)
		# valid_hop = range(0, self.max_hop + 1)
		# adjacency = np.zeros((self.num_node, self.num_node))
		# for hop in valid_hop:
		# 	adjacency[self.hop_dis == hop] = 1
		return adjacency

	def normalize_adjacency(self, A):
		Dl = np.sum(A, 0)
		num_node = A.shape[0]
		Dn = np.zeros((num_node, num_node))
		for i in range(num_node):
			if Dl[i] > 0:
				Dn[i, i] = Dl[i]**(-1)
		AD = np.dot(A, Dn)

		valid_hop = range(0, self.max_hop + 1)
		A = np.zeros((len(valid_hop), self.num_node, self.num_node))
		for i, hop in enumerate(valid_hop):
			A[i][self.hop_dis == hop] = AD[self.hop_dis == hop]
		return A


	