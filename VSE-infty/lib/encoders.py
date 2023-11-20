"""VSE modules"""

import torch
import torch.nn as nn
import numpy as np

from collections import OrderedDict
from transformers import BertModel
from transformers.adapters import BertAdapterModel

from lib.utils import l2norm, count_params, random_drop_feature
from lib.modules.resnet import ResnetFeatureExtractor
from lib.modules.aggr.gpo import GPO
from lib.modules.mlp import MLP
from lib.adapter_for_cnn import side_tuning_factory_for_resnet
from lib.adapter_for_transformer import side_tuning_factory_for_transformer

import logging
logger = logging.getLogger(__name__)


def get_text_encoder(opt, embed_size, no_txtnorm=False):
    return EncoderText(opt, embed_size, no_txtnorm=no_txtnorm)


def get_image_encoder(opt, img_dim, embed_size, precomp_enc_type='basic',
                      backbone_source=None, backbone_path=None, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImageAggr(
            img_dim, embed_size, precomp_enc_type, no_imgnorm)
    elif precomp_enc_type == 'backbone':
        img_enc = EncoderImageFull(
            opt, backbone_source, backbone_path, img_dim, embed_size, precomp_enc_type, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.precomp_enc_type = precomp_enc_type
        if precomp_enc_type == 'basic':
            self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)
        self.gpool = GPO(32, 32)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, image_lengths):
        """Extract image feature vectors."""
        features = self.fc(images)
        if self.precomp_enc_type == 'basic':
            # When using pre-extracted region features, add an extra MLP for the embedding transformation
            features = self.mlp(images) + features

        features, pool_weights = self.gpool(features, image_lengths)

        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features


class EncoderImageFull(nn.Module):
    def __init__(self, opt, backbone_source, backbone_path, img_dim, embed_size, precomp_enc_type='backbone', no_imgnorm=False):
        super(EncoderImageFull, self).__init__()
        self.backbone_freezed = False
        self.adapter_name = opt.img_adapter_name

        logger.info('=> loading adapter (name: {}) into Resnet'.format(self.adapter_name))

        if self.adapter_name in side_tuning_factory_for_resnet.keys():
            self.backbone = ResnetFeatureExtractor(backbone_source, backbone_path, fixed_blocks=2)
            self.freeze_backbone()

            self.side_tuning_module = \
                side_tuning_factory_for_resnet[self.adapter_name](num_layer=4,
                                                                  embed_size=opt.embed_size,
                                                                  hidden_size=[256, 512, 1024, 2048],
                                                                  downsample_D_factor=opt.img_downsample_D_factor)
            self.resnet_need_train_params = list(self.side_tuning_module.parameters())

            self.image_encoder = GPO(32, 32)
            self.rest_need_train_params = list(self.image_encoder.parameters())

        else:
            self.backbone = ResnetFeatureExtractor(backbone_source, backbone_path, fixed_blocks=2)
            self.resnet_need_train_params = list(self.backbone.parameters())

            self.image_encoder = EncoderImageAggr(img_dim, embed_size, precomp_enc_type, no_imgnorm)
            self.rest_need_train_params = list(self.image_encoder.parameters())

    def forward(self, images):
        """Extract image feature vectors."""

        if self.adapter_name in side_tuning_factory_for_resnet.keys():
            with torch.no_grad():
                hidden_features = self.backbone(images)[1]
            base_features = self.side_tuning_module(hidden_features)
        else:
            base_features = self.backbone(images)[0]

        if self.training:
            # Size Augmentation during training, randomly drop grids
            base_features, feat_lengths = random_drop_feature(base_features, 0.2)
        else:
            feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device)
            feat_lengths[:] = base_features.size(1)

        if self.adapter_name in side_tuning_factory_for_resnet.keys():
            features, _ = self.image_encoder(base_features, feat_lengths)
            features = l2norm(features, dim=-1)
        else:
            features = self.image_encoder(base_features, feat_lengths)

        return features

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info('Backbone freezed.')

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.backbone.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True
        self.backbone.set_fixed_blocks(fixed_blocks)
        self.backbone.unfreeze_base()
        logger.info('Backbone unfreezed, fixed blocks {}'.format(self.backbone.get_fixed_blocks()))


# Language Model with BERT
class EncoderText(nn.Module):
    def __init__(self, opt, embed_size, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        self.adapter_name = opt.txt_adapter_name

        logger.info('=> loading adapter (name: {}) into bert'.format(self.adapter_name))
        self.bert_need_train_params = list()
        self.rest_need_train_params = list()

        if self.adapter_name in side_tuning_factory_for_transformer.keys():
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.freeze_bert()

            self.side_tuning_module = \
                side_tuning_factory_for_transformer[self.adapter_name](num_layer=13,
                                                                       hidden_size=[768] * 13,
                                                                       embed_size=opt.embed_size,
                                                                       downsample_D_factor=opt.txt_downsample_D_factor)
            self.bert_need_train_params = list(self.side_tuning_module.parameters())
            
        else:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.bert_need_train_params = list(self.bert.parameters())

            self.linear = nn.Linear(768, embed_size)
            self.rest_need_train_params.extend(list(self.linear.parameters()))

        self.gpool = GPO(32, 32)
        self.rest_need_train_params.extend(list(self.gpool.parameters()))

    def freeze_bert(self):
        for param in self.bert.parameters():
            param.requires_grad = False
        logger.info('BERT freezed.')

    def unfreeze_bert(self):
        for param in self.bert.parameters(): 
            param.requires_grad = True
        logger.info('BERT unfreezed.')

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        bert_attention_mask = (x != 0).float()

        if self.adapter_name in side_tuning_factory_for_transformer.keys():
            with torch.no_grad():
                hidden_features = self.bert(x, bert_attention_mask, output_hidden_states=True)[2]
            cap_emb, cap_len = self.side_tuning_module(hidden_features, lengths=lengths)
        else:
            bert_emb = self.bert(x, bert_attention_mask)[0]  # B x N x D
            cap_emb = self.linear(bert_emb)
            cap_len = lengths     

        pooled_features, pool_weights = self.gpool(cap_emb, cap_len.to(cap_emb.device))

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            pooled_features = l2norm(pooled_features, dim=-1)

        return pooled_features


if __name__ == '__main__':
    import os 
    from arguments import get_argument_parser
    
    os.environ['CUDA_VISIBLE_DEVICES']='0'

    parser = get_argument_parser()
    opt = parser.parse_args()

    adapter_list = [None, 'our_tuning', 'ladder_side_tuning', \
        'layernorm', 'bitfit', 'prefix_tuning', 'prefix_tuning_flat', \
        'bottleneck_adapter', 'lang_adapter', 'lora_adapter', 'ia3_adapter', 'dummy', \
        'union_adapter', 'mam_adapter', 'unipelt']

    for adapter_name in adapter_list[1:2]:
        opt.txt_adapter_name = adapter_name
        opt.img_adapter_name = adapter_name
        model = EncoderText(opt, 1024, False).cuda()

        print('# ------------------------------------- #')
        print(adapter_name)
        # if adapter_name in adapter_factory_for_transformer.keys():
        #     print(model.bert.adapter_summary())
        
        num_param = count_params(model.bert_need_train_params) / 1e6
        print('Params for training: %.2f M' % (num_param), 'Its proportion of BERT: %.2f' % (num_param / (109.49)))

        tensorA = torch.Tensor([[i for i in range(101, 121)] for _ in range(13)]).long().cuda()
        lens = torch.FloatTensor([10, 15, 19, 16, 9, 10, 15, 19, 16, 9, 10, 15, 19]).cuda()
        output = model(tensorA, lens)
        print(output)
