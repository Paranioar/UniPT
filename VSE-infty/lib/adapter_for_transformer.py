import math
import random

import numpy as np

import torch
import torch.nn as nn

import transformers.adapters as adapters
from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.bert.configuration_bert import BertConfig

from lib.utils import l1norm, l2norm

import logging
logger = logging.getLogger(__name__)


class Our_tuning_for_transformer(nn.Module):
    def __init__(self, num_layer=13, hidden_size=None, embed_size=None, downsample_D_factor=None, **kwargs):
        super(Our_tuning_for_transformer, self).__init__()

        assert num_layer == len(hidden_size)

        self.num_layer = num_layer
        mapping_size = embed_size // downsample_D_factor

        self.downsample_layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size[i], mapping_size),
                                                              nn.LayerNorm(mapping_size))
                                               for i in range(num_layer)])
        
        self.projection_layer = nn.Sequential(nn.Linear(mapping_size, mapping_size),
                                              nn.LayerNorm(mapping_size))
        
        self.upsample_layer = nn.Sequential(nn.ReLU(), nn.Linear(mapping_size, embed_size))

        self.gate_layer = nn.Linear(mapping_size, 1)

    def _compute_mask(self, max_len, valid_len):
        mask = torch.arange(max_len).expand(valid_len.size(0), max_len).to(valid_len.device)
        mask = (mask < valid_len.long().unsqueeze(1))

        return mask
    
    def _aggregate_feature(self, features, mask=None):
        
        if mask is None:
            agg_feature = features.mean(dim=1)
        else:
            agg_mask = mask.unsqueeze(-1).float()
            features = features * agg_mask
            agg_feature = features.sum(dim=1) / agg_mask.sum(dim=1)

        return agg_feature

    def _cross_attention(self, query, context, mask=None, **kwargs):

        cross_weights = torch.matmul(query, context.permute(0, 2, 1))

        if mask is not None:
            cross_weights = cross_weights * mask.unsqueeze(1)

        cross_weights = l1norm(torch.relu(cross_weights), dim=-1)
        cross_weights += torch.eye(cross_weights.size(-1)).to(cross_weights.device)

        wcontext = torch.matmul(cross_weights, context)

        return wcontext, cross_weights

    def forward(self, all_hidden_features, lengths=None, **kwargs):

        assert len(all_hidden_features) == self.num_layer
        n_instance = all_hidden_features[0].size(1)

        mask = self._compute_mask(max_len=n_instance, valid_len=lengths)

        anchor_features = self.downsample_layers[-1](all_hidden_features[-1])
        anchor = self._aggregate_feature(anchor_features, mask)

        confidence_set = []
        merged_features_set = []
        for index in range(self.num_layer-1):

            target_features = self.downsample_layers[index](all_hidden_features[index])
            target = self._aggregate_feature(target_features, mask)
            merged_features, _ = self._cross_attention(anchor_features, target_features, mask=mask)

            merged_features_set.append(merged_features)
            confidence_set.append(self.gate_layer(anchor * target))

        confidence_norm = torch.softmax(torch.cat(confidence_set, dim=1), dim=1)
        all_merged_features = torch.stack(merged_features_set, dim=1)
        merged_features = torch.sum(all_merged_features * confidence_norm[:, :, None, None], dim=1)
        output_features = anchor_features + self.projection_layer(merged_features)

        return self.upsample_layer(output_features), lengths
    

side_tuning_factory_for_transformer = {
    'our_tuning': Our_tuning_for_transformer,
}