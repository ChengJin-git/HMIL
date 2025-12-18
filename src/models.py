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

		# Network architecture dimensions
		self.feature_dim = 1024
		self.reduced_dim = 256
		self.attention_hidden_dim = 128
		self.attention_output_dim = 64

		# Feature extraction layers
		self.feature_extractor = nn.Sequential(
			nn.Linear(self.feature_dim, self.reduced_dim),
			nn.ReLU(),
			nn.Linear(self.reduced_dim, self.feature_dim)
		)

		self.feature_reducer = nn.Sequential(
			nn.Linear(self.feature_dim, self.reduced_dim)
		)

		# Attention networks with clear parameter names
		self.coarse_attention = GatedAttention(
			input_dim=self.feature_dim,
			hidden_dim=self.reduced_dim,
			dropout=dropout,
			n_classes=n_classes[0]
		)

		self.fine_attention = GatedAttention(
			input_dim=self.reduced_dim,
			hidden_dim=self.attention_output_dim,
			dropout=dropout,
			n_classes=n_classes[1]
		)

		# Classifiers
		self.coarse_classifier = nn.Linear(self.feature_dim, n_classes[0])
		self.fine_classifier = nn.Linear(self.reduced_dim, n_classes[1])

		initialize_weights(self)
		
	def _extract_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Extract hierarchical features from input

		Args:
			x: Input tensor

		Returns:
			Tuple of (coarse_features, fine_features)
		"""
		x = x.squeeze()
		coarse_features = self.feature_extractor(x)
		fine_features = self.feature_reducer(coarse_features)
		return coarse_features, fine_features

	def _compute_attention(self,
						  coarse_features: torch.Tensor,
						  fine_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		"""
		Compute attention weights for both levels

		Args:
			coarse_features: Coarse level features
			fine_features: Fine level features

		Returns:
			Tuple of (coarse_attention, fine_attention)
		"""
		A1, _ = self.coarse_attention(coarse_features)
		A2, _ = self.fine_attention(fine_features)

		# Transpose and apply softmax
		A1 = F.softmax(torch.transpose(A1, 1, 0), dim=1)
		A2 = F.softmax(torch.transpose(A2, 1, 0), dim=1)

		return A1, A2

	def _aggregate_features(self,
						   attention_coarse: torch.Tensor,
						   attention_fine: torch.Tensor,
						   coarse_features: torch.Tensor,
						   fine_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
		"""
		Aggregate features using attention weights

		Args:
			attention_coarse: Coarse attention weights
			attention_fine: Fine attention weights
			coarse_features: Coarse level features
			fine_features: Fine level features

		Returns:
			Tuple of (aggregated_coarse, aggregated_fine, normalized_fine)
		"""
		M1 = torch.mm(attention_coarse, coarse_features)
		M2 = torch.mm(attention_fine, fine_features)
		normalized_M2 = F.normalize(M2, p=2, dim=1)

		return M1, M2, normalized_M2

	def _compute_logits(self,
					   M1: torch.Tensor,
					   M2: torch.Tensor) -> List[torch.Tensor]:
		"""
		Compute classification logits

		Args:
			M1: Coarse level aggregated features
			M2: Fine level aggregated features

		Returns:
			List of [coarse_logits, fine_logits]
		"""
		logits_coarse = self.coarse_classifier(M1)
		logits_fine = self.fine_classifier(M2)
		return [logits_coarse, logits_fine]

	def test(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
		"""
		Test forward pass (no gradient computation needed)

		Args:
			x: Input tensor

		Returns:
			Tuple of (attention maps, logits)
		"""
		# Extract features once
		coarse_features, fine_features = self._extract_features(x)

		# Compute attention
		A1, A2 = self._compute_attention(coarse_features, fine_features)

		# Aggregate and classify
		M1, M2, _ = self._aggregate_features(A1, A2, coarse_features, fine_features)
		logits = self._compute_logits(M1, M2)

		return [A1, A2], logits

	def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
		"""
		Forward pass for training

		Args:
			x: Input tensor

		Returns:
			Tuple of (attention maps, logits, normalized fine features)
		"""
		# Extract features once (avoid redundant computation)
		coarse_features, fine_features = self._extract_features(x)

		# Compute attention
		A1, A2 = self._compute_attention(coarse_features, fine_features)

		# Aggregate and classify
		M1, M2, normalized_M2 = self._aggregate_features(A1, A2, coarse_features, fine_features)
		logits = self._compute_logits(M1, M2)

		return [A1, A2], logits, normalized_M2