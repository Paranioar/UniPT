import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from transformers.models.bert.modeling_bert import BertLayer
from transformers.models.bert.configuration_bert import BertConfig

from lib.utils import l1norm, l2norm, count_params

import logging
logger = logging.getLogger(__name__)


def flat_featuremap(feature):
    assert len(feature.shape) == 4
    return feature.reshape(feature.size(0), feature.size(1), -1).permute(0, 2, 1)


def roll_featuremap(feature):
    assert len(feature.shape) == 3
    size = int(math.sqrt(feature.size(1)))
    return feature.permute(0, 2, 1).reshape(feature.size(0), feature.size(2), size, size)


class Our_tuning_for_resnet(nn.Module):
    def __init__(self, num_layer=4, hidden_size=None, embed_size=None, downsample_D_factor=None, **kwargs):
        super(Our_tuning_for_resnet, self).__init__()

        assert num_layer == len(hidden_size)

        self.HW_rate = [8, 4, 2, 1]
        self.HW_size = [128, 64, 32, 16]
        self.num_layer = num_layer
        mapping_size = embed_size // downsample_D_factor

        self.downsample_layers = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(hidden_size[i], mapping_size, kernel_size=1, stride=1),
                           nn.BatchNorm2d(mapping_size))
            for i in range(num_layer)])
        
        self.projection_layer = nn.Sequential(nn.Conv2d(mapping_size, mapping_size, kernel_size=1, stride=1),
                                              nn.BatchNorm2d(mapping_size))

        self.upsample_layer = nn.Sequential(nn.ReLU(), nn.Conv2d(mapping_size, embed_size, kernel_size=1, stride=1))
        
        self.identity_layers = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(hidden_size[-1], hidden_size[n], kernel_size=1, stride=1), nn.ReLU())
            for n in range(num_layer-1)])

        self.unfold_layers = nn.ModuleList([nn.Unfold(kernel_size=self.HW_rate[j], stride=self.HW_rate[j]) 
                                            for j in range(num_layer-1)])
        
        self.fold_layers = nn.ModuleList([nn.Fold(output_size=self.HW_size[j], kernel_size=self.HW_rate[j], stride=self.HW_rate[j])
                                          for j in range(num_layer-1)])
        
        self.pooling_layers = nn.ModuleList([nn.AvgPool2d(self.HW_rate[m], stride=self.HW_rate[m], divisor_override=1) 
                                             for m in range(num_layer-1)])

        self.gate_layer = nn.Linear(mapping_size, 1)

    def _upsample_feature(self, feature, size):
        '''Upsample the feature map.
        '''
        return F.interpolate(feature, size=size, mode='nearest')
    
    def _cross_attention(self, query, context, mask=None, **kwargs):

        cross_weights = torch.matmul(query, context.permute(0, 2, 1))

        if mask is not None:
            cross_weights = cross_weights * mask.unsqueeze(1)

        cross_weights = l1norm(torch.relu(cross_weights), dim=-1)
        cross_weights += torch.eye(cross_weights.size(-1)).to(cross_weights.device)
        
        wcontext = torch.matmul(cross_weights, context)

        return wcontext, cross_weights

    def forward(self, all_hidden_features, **kwargs):

        assert len(all_hidden_features) == self.num_layer

        anchor_features = self.downsample_layers[-1](all_hidden_features[-1])
        flat_anchor_features = flat_featuremap(anchor_features)
        anchor = flat_anchor_features.mean(dim=1)

        confidence_set = []
        flat_merged_features_set = []
        for index in range(self.num_layer-1):

            raw_target_features = all_hidden_features[index]
            ide_anchor_features = self.identity_layers[index](all_hidden_features[-1])

            mixed_features = self._upsample_feature(ide_anchor_features, raw_target_features.size()[2:]) * raw_target_features
            target_weights = self.unfold_layers[index](mixed_features.sum(dim=1, keepdim=True))
            target_weights = self.fold_layers[index](l1norm(torch.relu(target_weights), dim=1))
            target_weights += 1.0 / pow(self.HW_rate[index], 2)

            target_features = self.pooling_layers[index](target_weights * raw_target_features)
            
            target_features = self.downsample_layers[index](target_features)
            flat_target_features = flat_featuremap(target_features)

            flat_merged_features, _ = self._cross_attention(flat_anchor_features, flat_target_features)
            flat_merged_features_set.append(flat_merged_features)
            target = flat_target_features.mean(dim=1)

            confidence_set.append(self.gate_layer(anchor * target))

        confidence_norm = torch.softmax(torch.cat(confidence_set, dim=1), dim=1)
        all_flat_merged_features = torch.stack(flat_merged_features_set, dim=1)
        flat_merged_features = torch.sum(all_flat_merged_features * confidence_norm[:, :, None, None], dim=1)
        output_features = anchor_features + self.projection_layer(roll_featuremap(flat_merged_features))

        return flat_featuremap(self.upsample_layer(output_features))


side_tuning_factory_for_resnet = {
    'our_tuning': Our_tuning_for_resnet,
}


if __name__=='__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES']='0'

    model = Our_tuning_for_resnet(num_layer=4,
                                  embed_size=1024,
                                  hidden_size=[256, 512, 1024, 2048],
                                  downsample_D_factor=2)
    
    num_params = count_params(model.parameters())
    print('Params for training: %.2f M' % (num_params / 1e6))
    
    tensorA = torch.randn(128, 256, 128, 128)
    tensorB = torch.randn(128, 512, 64, 64)
    tensorC = torch.randn(128, 1024, 32, 32)
    tensorD = torch.randn(128, 2048, 16, 16)

    output = model([tensorA, tensorB, tensorC, tensorD])
    print(output)

    print('finished')