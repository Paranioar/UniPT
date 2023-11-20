import torch
import torch.nn as nn


def l1norm(X, dim, eps=1e-6):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True)
    out = torch.div(X, norm.clamp(min=eps))
    return out


def l2norm(X, dim, eps=1e-6):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt()
    X = torch.div(X, norm.clamp(min=eps))
    return X


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    

class Our_tuning_for_transformer(nn.Module):
    def __init__(self, num_layer=13, hidden_size=None, embed_size=None, downsample_D_factor=None, **kwargs):
        super(Our_tuning_for_transformer, self).__init__()

        assert num_layer == len(hidden_size)
        self.num_layer = num_layer

        mapping_size = embed_size // downsample_D_factor
        self.downsample_layers = nn.ModuleList([nn.Sequential(nn.ReLU(), nn.Linear(hidden_size[i], mapping_size))
                                                for i in range(num_layer)])
        self.downsample_layer = nn.Sequential(nn.ReLU(), nn.Linear(embed_size, mapping_size))
        self.upsample_layer = nn.Sequential(nn.ReLU(), nn.Linear(mapping_size, embed_size))
        
        self.gate_layer = nn.Linear(mapping_size, 1)

    def _aggregate_feature(self, features, mask=None):
        
        if mask is None:
            agg_feature = features.mean(dim=1)
        else:
            agg_mask = mask.unsqueeze(-1)
            features = features * agg_mask
            agg_feature = features.sum(dim=1) / (agg_mask.sum(dim=1)).clamp(min=1.0)

        return agg_feature

    def _cross_attention(self, query, context, mask=None, **kwargs):

        cross_weights = torch.matmul(query, context.permute(0, 2, 1))

        if mask is not None:
            cross_weights = cross_weights * mask.unsqueeze(1)
        cross_weights = l1norm(torch.relu(cross_weights), dim=-1)

        wcontext = torch.matmul(cross_weights, context)

        return wcontext, cross_weights
    
    @property
    def dtype(self):
        return self.gate_layer.weight.dtype

    def forward(self, all_hidden_features, raw_anchor_features, mask=None, position=None, **kwargs):

        anchor_features = self.downsample_layer(raw_anchor_features.type(self.dtype))
        anchor = anchor_features[:, 0, :]

        confidence_set = []
        merged_features_set = []
        for index in range(self.num_layer):

            target_features = self.downsample_layers[index](all_hidden_features[index].type(self.dtype))
            target = self._aggregate_feature(target_features, mask)
            merged_features, _ = self._cross_attention(anchor_features, target_features, mask=mask)

            if position is not None:
                merged_features += target_features[torch.arange(target_features.shape[0]), position].unsqueeze(1)
            else:
                merged_features += target_features[:, 0, :].unsqueeze(1)

            merged_features_set.append(merged_features)
            confidence_set.append(self.gate_layer(anchor * target))

        confidence_norm = torch.softmax(torch.cat(confidence_set, dim=1), dim=1)
        all_merged_features = torch.stack(merged_features_set, dim=1)
        merged_features = torch.sum(all_merged_features * confidence_norm[:, :, None, None], dim=1)
        output_features = raw_anchor_features.type(self.dtype) + self.upsample_layer(merged_features)

        return output_features.squeeze(1)


side_tuning_factory_for_transformer = {
    'our_tuning': Our_tuning_for_transformer,
}
