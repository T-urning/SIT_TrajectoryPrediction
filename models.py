from typing_extensions import OrderedDict
import torch
import torch.nn as nn

from layers.graph import Graph
from layers.graph_conv_block import Graph_Conv_Block
from layers.seq2seq import Seq2Seq, TransformerSeq2Seq
import numpy as np

class TPHGI(nn.Module):
	def __init__(self, in_channels, graph_args, edge_importance_weighting, predict_lane=False, **kwargs):
		
		super().__init__()

		# load graph
		self.graph = Graph(**graph_args)
		A_shape = (graph_args['max_hop']+1, graph_args['num_node'], graph_args['num_node'])

		# build networks
		num_hetero_types = A_shape[0]
		temporal_kernel_size = 5 #9 #5 # 3
		kernel_size = (temporal_kernel_size, num_hetero_types)
		self.out_channels = 64
		# best
		self.st_gcn_networks = nn.ModuleList((
			nn.BatchNorm2d(in_channels),
			Graph_Conv_Block(in_channels, self.out_channels, kernel_size, 1, residual=True, use_hetero_graph=True, **kwargs),
			Graph_Conv_Block(self.out_channels, self.out_channels, kernel_size, 1, use_hetero_graph=True, **kwargs),
			Graph_Conv_Block(self.out_channels, self.out_channels, kernel_size, 1, use_hetero_graph=True, **kwargs),
		))

		
		# initialize transform matrix's parameters for the using of heterogenous graph
		self.transform_matrix = nn.Parameter(torch.rand(num_hetero_types, 64, 64), requires_grad=True)

		# initialize parameters for edge importance weighting
		if edge_importance_weighting:
			self.edge_importance = nn.ParameterList(
				[nn.Parameter(torch.ones(A_shape), requires_grad=True) for i in self.st_gcn_networks]
			)
		else:
			self.edge_importance = [1] * len(self.st_gcn_networks)

		self.num_node = num_node = self.graph.num_node
		self.out_dim_per_node = out_dim_per_node = 2 #(x, y) coordinate
		self.seq2seq_type = kwargs.get('seq2seq_type', 'gru')
		if self.seq2seq_type in ['gru', 'lstm']:
			self.seq2seq = Seq2Seq(self.seq2seq_type, input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)
		else:
			self.seq2seq = TransformerSeq2Seq()
		# self.seq2seq_human = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)
		# self.seq2seq_bike = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)
		# lane id predictor
		self.predict_lane = predict_lane
		if self.predict_lane:
			# self.seq2seq.decoder.lstm.hidden_size = out_dim_per_node * 30
			input_size = self.seq2seq.decoder.rnn.hidden_size * 2
			self.lane_predictor = nn.Sequential(OrderedDict([
				('linear1', nn.Linear(input_size, 64)),
				('relu', nn.ReLU()),
				('linear2', nn.Linear(64, 8))
			])) # the valid lane id range is [1, 7], we let 0 represent the unknown lane id.


	def reshape_for_lstm(self, feature):
		# prepare for skeleton prediction model
		'''
		N: batch_size
		C: channel
		T: time_step
		V: nodes
		'''
		N, C, T, V = feature.size() 
		now_feat = feature.permute(0, 3, 2, 1).contiguous() # to (N, V, T, C)
		now_feat = now_feat.view(N*V, T, C) 
		return now_feat

	def reshape_from_lstm(self, predicted):
		# predicted (N*V, T, C)
		NV, T, C = predicted.size()
		now_feat = predicted.view(-1, self.num_node, T, self.out_dim_per_node) # (N, T, V, C) -> (N, C, T, V) [(N, V, T, C)]
		now_feat = now_feat.permute(0, 3, 2, 1).contiguous() # (N, C, T, V)
		return now_feat

	def forward(self, pra_x, pra_A, pra_pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=None):
		x = pra_x
		
		# forwad
		for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
			if type(gcn) is nn.BatchNorm2d:
				x = gcn(x)
			else:
				x, _ = gcn(x, pra_A + importance, self.transform_matrix)
				
		# prepare for seq2seq lstm model
		graph_conv_feature = self.reshape_for_lstm(x) # (N*V, T, 64)
		last_position = self.reshape_for_lstm(pra_x[:,:2]) #(N, C, T, V)[:, :2] -> (N, T, V*2) [(N*V, T, 2)]

		if pra_teacher_forcing_ratio > 0 and type(pra_teacher_location) is not type(None):
			pra_teacher_location = self.reshape_for_lstm(pra_teacher_location)

		# now_predict.shape = (N, T, V*C)
		now_predict, lane_id_logits = self.seq2seq(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
		now_predict = self.reshape_from_lstm(now_predict) # (N, C, T, V)

		lane_id_predict = None
		if self.predict_lane:
			lane_id_predict = self.lane_predictor(lane_id_logits) # (N*V, T, 10)
			
		# now_predict_human = self.seq2seq_human(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
		# now_predict_human = self.reshape_from_lstm(now_predict_human) # (N, C, T, V)

		# now_predict_bike = self.seq2seq_bike(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
		# now_predict_bike = self.reshape_from_lstm(now_predict_bike) # (N, C, T, V)


		return now_predict, lane_id_predict

class GRIP(nn.Module):
	def __init__(self, in_channels, graph_args, edge_importance_weighting, **kwargs):
		super().__init__()

		# load graph
		self.graph = Graph(**graph_args)
		A_shape = (graph_args['max_hop']+1, graph_args['num_node'], graph_args['num_node'])

		# build networks
		spatial_kernel_size = A_shape[0]
		temporal_kernel_size = 5 #9 #5 # 3
		kernel_size = (temporal_kernel_size, spatial_kernel_size)

		# best
		self.st_gcn_networks = nn.ModuleList((
			nn.BatchNorm2d(in_channels),
			Graph_Conv_Block(in_channels, 64, kernel_size, 1, residual=True, **kwargs),
			Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
			Graph_Conv_Block(64, 64, kernel_size, 1, **kwargs),
		))

		# initialize parameters for edge importance weighting
		if edge_importance_weighting:
			self.edge_importance = nn.ParameterList(
				[nn.Parameter(torch.ones(A_shape)) for i in self.st_gcn_networks]
				)
		else:
			self.edge_importance = [1] * len(self.st_gcn_networks)

		self.num_node = num_node = self.graph.num_node
		self.out_dim_per_node = out_dim_per_node = 2 #(x, y) coordinate
		self.seq2seq_type = kwargs.get('seq2seq_type', 'gru')
		if self.seq2seq_type in ['gru', 'lstm']:
			self.seq2seq = Seq2Seq(self.seq2seq_type, input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)
		else:
			self.seq2seq = TransformerSeq2Seq()
		
		# self.seq2seq_human = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)
		# self.seq2seq_bike = Seq2Seq(input_size=(64), hidden_size=out_dim_per_node, num_layers=2, dropout=0.5, isCuda=True)


	def reshape_for_lstm(self, feature):
		# prepare for skeleton prediction model
		'''
		N: batch_size
		C: channel
		T: time_step
		V: nodes
		'''
		N, C, T, V = feature.size() 
		now_feat = feature.permute(0, 3, 2, 1).contiguous() # to (N, V, T, C)
		now_feat = now_feat.view(N*V, T, C) 
		return now_feat

	def reshape_from_lstm(self, predicted):
		# predicted (N*V, T, C)
		NV, T, C = predicted.size()
		now_feat = predicted.view(-1, self.num_node, T, self.out_dim_per_node) # (N, T, V, C) -> (N, C, T, V) [(N, V, T, C)]
		now_feat = now_feat.permute(0, 3, 2, 1).contiguous() # (N, C, T, V)
		return now_feat

	def forward(self, pra_x, pra_A, pra_pred_length, pra_teacher_forcing_ratio=0, pra_teacher_location=None):
		x = pra_x
		
		# forwad
		for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
			if type(gcn) is nn.BatchNorm2d:
				x = gcn(x)
			else:
				x, _ = gcn(x, pra_A + importance)
				
		# prepare for seq2seq lstm model
		graph_conv_feature = self.reshape_for_lstm(x)
		last_position = self.reshape_for_lstm(pra_x[:,:2]) #(N, C, T, V)[:, :2] -> (N, T, V*2) [(N*V, T, C)]

		if pra_teacher_forcing_ratio > 0 and type(pra_teacher_location) is not type(None):
			pra_teacher_location = self.reshape_for_lstm(pra_teacher_location)

		# now_predict.shape = (N, T, V*C)
		now_predict, _ = self.seq2seq(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
		now_predict = self.reshape_from_lstm(now_predict) # (N, C, T, V)

		# now_predict_human = self.seq2seq_human(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
		# now_predict_human = self.reshape_from_lstm(now_predict_human) # (N, C, T, V)

		# now_predict_bike = self.seq2seq_bike(in_data=graph_conv_feature, last_location=last_position[:,-1:,:], pred_length=pra_pred_length, teacher_forcing_ratio=pra_teacher_forcing_ratio, teacher_location=pra_teacher_location)
		# now_predict_bike = self.reshape_from_lstm(now_predict_bike) # (N, C, T, V)


		return now_predict, None

if __name__ == '__main__':
	model = GRIP(in_channels=3, pred_length=6, graph_args={}, edge_importance_weighting=True)
	print(model)
