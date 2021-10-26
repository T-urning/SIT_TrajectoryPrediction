import einops
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np 

####################################################
# Seq2Seq LSTM AutoEncoder Model
# 	- predict locations
####################################################

class EncoderRNN(nn.Module):
	def __init__(self, type, input_size, hidden_size, num_layers):
		super(EncoderRNN, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		# self.lstm = nn.LSTM(input_size, hidden_size*30, num_layers, batch_first=True)
		if type == 'lstm':
			self.rnn = nn.LSTM(input_size, hidden_size*30, num_layers, batch_first=True)
		else:
			self.rnn = nn.GRU(input_size, hidden_size*30, num_layers, batch_first=True)
		
	def forward(self, input):
		output, hidden = self.rnn(input)
		return output, hidden

class DecoderRNN(nn.Module):
	def __init__(self, type, hidden_size, output_size, num_layers, dropout=0.5):
		super(DecoderRNN, self).__init__()
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.num_layers = num_layers
		# self.lstm = nn.LSTM(hidden_size, output_size, num_layers, batch_first=True)
		if type == 'lstm':
			self.rnn = nn.LSTM(hidden_size, output_size*30, num_layers, batch_first=True)
		else:
			self.rnn = nn.GRU(hidden_size, output_size*30, num_layers, batch_first=True)

		#self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()
		self.dropout = nn.Dropout(p=dropout)
		self.linear = nn.Linear(output_size*30, output_size)
		self.tanh = nn.Tanh()
	
	def forward(self, encoded_input, hidden):
		decoded_output, hidden = self.rnn(encoded_input, hidden)
		# decoded_output = self.tanh(decoded_output)
		# decoded_output = self.sigmoid(decoded_output)
		decoded_output = self.dropout(decoded_output)
		# decoded_output = self.tanh(self.linear(decoded_output))
		decoded_output = self.linear(decoded_output)
		#TODO decoded_output = self.sigmoid(self.linear(decoded_output))
		decoded_output = self.tanh(decoded_output)
		return decoded_output, hidden

class Seq2Seq(nn.Module):
	def __init__(self, type, input_size, hidden_size, num_layers, dropout=0.5, interact_in_decoding=False, max_num_object=260):
		super(Seq2Seq, self).__init__()

		# hidden_size = 2, input_size = 64
		self.num_layers = num_layers
		self.max_num_object = max_num_object
		self.interact_in_decoding = interact_in_decoding
		self.dropout = nn.Dropout(p=dropout)
		decoder_in_size = hidden_size
		if self.interact_in_decoding:
			self.self_attention = SelfAttention(hidden_size * 30)
			self.linear_pos_to_hidden = nn.Linear(hidden_size, hidden_size * 30)
			decoder_in_size = hidden_size * 30 * 2 # 120 = 60 (position_hidden) + 60 (interacted_hidden) 
		self.encoder = EncoderRNN(type, input_size, hidden_size, num_layers)
		self.decoder = DecoderRNN(type, decoder_in_size, hidden_size, num_layers, dropout)
	
	def forward(self, in_data, last_location, pred_length, teacher_forcing_ratio=0, teacher_location=None):
		batch_size = in_data.shape[0]
		out_dim = self.decoder.output_size

		outputs = torch.zeros(batch_size, pred_length, out_dim).to(in_data.device)
		encoded_output, hidden = self.encoder(in_data) # (N * V, T, 60), (2, N * V, 60)
		#hidden_ = hidden[0] if isinstance(hidden, tuple) else hidden
		hidden_outputs = torch.zeros(batch_size, pred_length, encoded_output.size(-1) * self.num_layers).to(in_data.device)
		decoder_input = last_location # (N * V, 1, 2)
		if self.interact_in_decoding:
			pos_hidden = self.linear_pos_to_hidden(last_location) # (N * V, 1, 2) -> (N * V, 1, 60)
			interact = self.message_passing(hidden.mean(dim=0)) # hidden: (self.num_layers, N * V, 60) -> (N * V, 60)
			decoder_input = torch.cat([pos_hidden, interact.unsqueeze(1)], dim=-1) # (N * V, 1, 60)
		
		for t in range(pred_length):
			# encoded_input = torch.cat((now_label, encoded_input), dim=-1) # merge class label into input feature
			now_out, hidden = self.decoder(decoder_input, hidden) # now_out is shape of (N * V, 1, 2), hidden is (2, N * V, 60)
			#TODO we force the model to predict the change of velocity by adding a residual connection
			# now_out += last_location
			outputs[:, t:t+1] = now_out
			hidden_ = hidden[0] if isinstance(hidden, tuple) else hidden # because GRU and LSTM have different outputs
			hidden_outputs[:, t:t+1] = hidden_.permute(1, 0, 2).contiguous().view(batch_size, 1, -1) # batch_size = N * V, (N * V, 1, 120)
			teacher_force = np.random.random() < teacher_forcing_ratio
			last_location = (teacher_location[:,t:t+1] if (type(teacher_location) is not type(None)) and teacher_force else now_out)
			decoder_input = last_location

			if self.interact_in_decoding:
				pos_hidden = self.linear_pos_to_hidden(last_location)
				interact = self.message_passing(hidden_.mean(dim=0))
				decoder_input = torch.cat([pos_hidden, interact.unsqueeze(1)], dim=-1) # (N * V, 1, 60)

		return outputs, hidden_outputs

	def message_passing(self, origin_input, mask=None):
		"""
		origin_input: (2, N*V, 60)
		mask: (N, V, V)
		"""
		#output_size, NV, hidden_size = origin_input.size()
		# (NV, 60) -> (N, V, 60)
		# einops.rearrange(origin_input, '(n v) o -> n v o', v=self.num_object)
		input = origin_input.view(-1, self.max_num_object, origin_input.size(-1))
		#input = origin_input.permute(1, 0, 2).contiguous().view(NV, output_size*hidden_size).view(-1, self.num_object, output_size*hidden_size)
		output = self.self_attention(input, mask) # (N, V, 60)
		output = self.activate(self.dropout(output))

		return output.view(-1, origin_input.size(-1)) # (N * V, 60)


class SelfAttention(nn.Module):
    """
    Implementation of plain self attention mechanism with einsum operations
    Paper: https://arxiv.org/abs/1706.03762
    Blog: https://theaisummer.com/transformer/
    """
    def __init__(self, dim):
        """
        Args:
            dim: for NLP it is the dimension of the embedding vector
            the last dimension size that will be provided in forward(x),
            where x is a 3D tensor
        """
        super().__init__()
        # for Step 1
        self.to_qvk = nn.Linear(dim, dim * 3, bias=False)
        # for Step 2
        self.scale_factor = dim ** -0.5  # 1/np.sqrt(dim)

    def forward(self, x, mask=None):
        assert x.dim() == 3, '3D tensor must be provided'

        # Step 1
        qkv = self.to_qvk(x)  # [batch, tokens, dim*3 ]

        # decomposition to q,v,k
        # rearrange tensor to [3, batch, tokens, dim] and cast to tuple
        q, k, v = tuple(rearrange(qkv, 'b t (d k) -> k b t d ', k=3))

        # Step 2
        # Resulting shape: [batch, tokens, tokens]
        scaled_dot_prod = torch.einsum('b i d , b j d -> b i j', q, k) * self.scale_factor

        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[1:]
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1)

        # Step 3
        return torch.einsum('b i j , b j d -> b i d', attention, v)

class TransformerSeq2Seq(nn.Module):
	def __init__(self):
		super().__init__()
		
####################################################
####################################################