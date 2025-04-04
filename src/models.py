import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


def initialize_weights(module: nn.Module) -> None:
	"""
	Initialize weights for the model using Kaiming initialization
	
	Args:
		module: PyTorch module to initialize
	"""
	for m in module.modules():
		if isinstance(m, nn.Linear):
			nn.init.kaiming_normal_(m.weight)
			m.bias.data.zero_()


class GatedAttention(nn.Module):
	"""
	Gated attention mechanism for hierarchical multi-instance learning
	
	This module implements a gated attention mechanism that learns to focus on
	relevant parts of the input features for classification.
	"""
	
	def __init__(self, 
				 input_dim: int = 512, 
				 hidden_dim: int = 128, 
				 dropout: bool = False, 
				 n_classes: int = 1):
		"""
		Initialize the gated attention module
		
		Args:
			input_dim: Dimension of input features
			hidden_dim: Dimension of hidden layer
			dropout: Whether to use dropout
			n_classes: Number of output classes
		"""
		super(GatedAttention, self).__init__()
		
		# Attention mechanism components
		self.attention_a = [
			nn.Linear(input_dim, hidden_dim),
			nn.Tanh()
		]
		
		self.attention_b = [
			nn.Linear(input_dim, hidden_dim),
			nn.Sigmoid()
		]
		
		if dropout:
			self.attention_a.append(nn.Dropout(0.25))
			self.attention_b.append(nn.Dropout(0.25))
			
		self.attention_a = nn.Sequential(*self.attention_a)
		self.attention_b = nn.Sequential(*self.attention_b)
		self.attention_c = nn.Linear(hidden_dim, n_classes)
		
	def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Forward pass of the gated attention module
		
		Args:
			x: Input tensor of shape (batch_size, input_dim)
			
		Returns:
			Tuple of (attention scores, input features)
		"""
		a = self.attention_a(x)
		b = self.attention_b(x)
		A = a.mul(b)
		A = self.attention_c(A)  # N x n_classes
		return A, x


class HMIL(nn.Module):
	"""
	Hierarchical Multi-Instance Learning model
	
	This model implements a hierarchical multi-instance learning approach with
	gated attention mechanisms at both coarse and fine levels.
	"""
	
	def __init__(self, n_classes: List[int], dropout: bool = False):
		"""
		Initialize the hierarchical MIL model
		
		Args:
			n_classes: List of [coarse_classes, fine_classes]
			dropout: Whether to use dropout
		"""
		super().__init__()
		
		self.n_classes = n_classes
		
		# Network architecture
		size = [1024, 256, 128, 64]
		
		# Feature extraction layers
		self.feature_extractor = nn.Sequential(
			nn.Linear(size[0], size[1]),
			nn.ReLU(),
			nn.Linear(size[1], size[0])
		)
		
		self.feature_reducer = nn.Sequential(
			nn.Linear(size[0], size[1])
		)
		
		# Attention networks
		self.coarse_attention = nn.Sequential(
			GatedAttention(L=size[0], D=size[1], dropout=dropout, n_classes=n_classes[0])
		)
		
		self.fine_attention = nn.Sequential(
			GatedAttention(L=size[1], D=size[3], dropout=dropout, n_classes=n_classes[1])
		)
		
		# Classifiers
		self.coarse_classifier = nn.Linear(size[0], 1)
		self.fine_classifier = nn.Linear(size[1], 1)
		
		initialize_weights(self)
		
	def _process_features(self, 
						 x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Process input features through the network
		
		Args:
			x: Input tensor
			
		Returns:
			Tuple of (coarse features, fine features, normalized fine features)
		"""
		x = x.squeeze()
		h1 = self.feature_extractor(x)
		h2 = self.feature_reducer(h1)
		
		# Apply attention
		A1, _ = self.coarse_attention(h1)
		A2, _ = self.fine_attention(h2)
		
		# Transpose and apply softmax
		A1 = F.softmax(torch.transpose(A1, 1, 0), dim=1)
		A2 = F.softmax(torch.transpose(A2, 1, 0), dim=1)
		
		# Compute weighted features
		M1 = torch.mm(A1, h1)
		M2 = torch.mm(A2, h2)
		
		return M1, M2, F.normalize(M2, p=2, dim=1)
		
	def _compute_logits(self, 
					   M1: torch.Tensor, 
					   M2: torch.Tensor) -> List[torch.Tensor]:
		"""
		Compute classification logits
		
		Args:
			M1: Coarse level features
			M2: Fine level features
			
		Returns:
			List of [coarse_logits, fine_logits]
		"""
		logits_h1 = self.coarse_classifier(M1).permute(1, 0)
		logits_h2 = self.fine_classifier(M2).permute(1, 0)
		return [logits_h1, logits_h2]
		
	def test(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
		"""
		Test forward pass
		
		Args:
			x: Input tensor
			
		Returns:
			Tuple of (attention maps, logits)
		"""
		M1, M2, _ = self._process_features(x)
		logits = self._compute_logits(M1, M2)
		
		# Get attention maps
		A1, _ = self.coarse_attention(self.feature_extractor(x.squeeze()))
		A2, _ = self.fine_attention(self.feature_reducer(self.feature_extractor(x.squeeze())))
		A1 = F.softmax(torch.transpose(A1, 1, 0), dim=1)
		A2 = F.softmax(torch.transpose(A2, 1, 0), dim=1)
		
		return [A1, A2], logits
		
	def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
		"""
		Forward pass
		
		Args:
			x: Input tensor
			
		Returns:
			Tuple of (attention maps, logits, normalized fine features)
		"""
		M1, M2, normalized_M2 = self._process_features(x)
		logits = self._compute_logits(M1, M2)
		
		# Get attention maps
		A1, _ = self.coarse_attention(self.feature_extractor(x.squeeze()))
		A2, _ = self.fine_attention(self.feature_reducer(self.feature_extractor(x.squeeze())))
		A1 = F.softmax(torch.transpose(A1, 1, 0), dim=1)
		A2 = F.softmax(torch.transpose(A2, 1, 0), dim=1)
		
		return [A1, A2], logits, normalized_M2