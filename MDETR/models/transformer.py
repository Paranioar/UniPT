# Copyright (c) Aishwarya Kamath & Nicolas Carion. Licensed under the Apache License 2.0. All Rights Reserved
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers import RobertaModel, RobertaTokenizerFast

# --------------------------------------------------------------
from models.position_encoding import PositionEmbeddingSine
from util.misc import NestedTensor
# --------------------------------------------------------------


class Transformer(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
        return_intermediate_dec=False,
        pass_pos_and_query=True,
        text_encoder_type="roberta-base",
        freeze_text_encoder=False,
        contrastive_loss=False,
    ):
        super().__init__()

        self.pass_pos_and_query = pass_pos_and_query
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm, return_intermediate=True)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(
            decoder_layer, num_decoder_layers, decoder_norm, return_intermediate=return_intermediate_dec)

        self.CLS = nn.Embedding(1, d_model) if contrastive_loss else None

        self._reset_parameters()

        self.tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_type)
        self.text_encoder = RobertaModel.from_pretrained(text_encoder_type)

        if freeze_text_encoder:
            for p in self.text_encoder.parameters():
                p.requires_grad_(False)
        
        self.expander_dropout = 0.1
        config = self.text_encoder.config
        self.resizer = FeatureResizer(
            input_feat_size=config.hidden_size,
            output_feat_size=d_model,
            dropout=self.expander_dropout,
        )

        self.d_model = d_model
        self.nhead = nhead

        # -----------------------------------------------------------------------
        self.d_side = d_model // 2
        select_layers = [1, 3, 5]
        self.select_layers = {}
        for value, key in enumerate(select_layers):
            self.select_layers[key] = value

        self.side_downsample0_encoder = nn.Sequential(nn.Linear(d_model, self.d_side), 
                                                      nn.LayerNorm(self.d_side))
        self.side_downsamples_encoder = nn.ModuleList([nn.Sequential(nn.Linear(d_model, self.d_side),
                                                                     nn.LayerNorm(self.d_side)) 
                                                       for i in range(len(self.select_layers))])
        self.side_projection_encoder = nn.Sequential(nn.Linear(self.d_side, d_model),
                                                     nn.ReLU(),
                                                     nn.Linear(d_model, self.d_side),
                                                     nn.LayerNorm(self.d_side))
        self.side_gate_layer_encoder = nn.Linear(self.d_side, 1)

        self.side_downsample0_decoder = nn.Sequential(nn.Linear(d_model, self.d_side), 
                                                      nn.LayerNorm(self.d_side))
        self.side_downsamples_decoder = nn.ModuleList([nn.Sequential(nn.Linear(d_model, self.d_side),
                                                                     nn.LayerNorm(self.d_side)) 
                                                       for i in range(len(self.select_layers))])
        self.side_projection_decoder = nn.Sequential(nn.Linear(self.d_side, d_model),
                                                     nn.ReLU(),
                                                     nn.Linear(d_model, self.d_side),
                                                     nn.LayerNorm(self.d_side))
        self.side_gate_layer_decoder = nn.Linear(self.d_side, 1)

        self.side_projection_crosser = nn.ModuleList([nn.Sequential(nn.Linear(self.d_side, self.d_side),
                                                                    nn.LayerNorm(self.d_side))
                                                      for i in range(len(self.select_layers))])

        self.side_final_upsample = nn.Sequential(nn.ReLU(),
                                                 nn.Linear(self.d_side, d_model),
                                                 nn.LayerNorm(d_model))
        
        # -----------------------------------------------------------------------
        # self.side_projection0_crosser = nn.Sequential(nn.Linear(self.d_side, self.d_side), 
        #                                               nn.LayerNorm(self.d_side))
        # -----------------------------------------------------------------------

        for n, p in self.named_parameters():
            if 'side' in n:
                p.requires_grad_(True)
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            else:
                p.requires_grad_(False)
        # -----------------------------------------------------------------------

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # -----------------------------------------------------------------------
    def l1norm(self, X, dim, eps=1e-7):
        """L1-normalize columns of X
        """
        norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
        X = torch.div(X, norm)
        return X

    def aggregate_feature(self, features, mask=None):
        
        if mask is None:
            agg_feature = features.mean(dim=1)
        else:
            agg_mask = mask.unsqueeze(-1).float()
            features = features * agg_mask
            agg_feature = features.sum(dim=1) / agg_mask.sum(dim=1)

        return agg_feature

    def cross_attention(self, query, context, mask=None, residual=True, **kwargs):

        cross_weights = torch.matmul(query, context.permute(0, 2, 1))

        if mask is not None:
            cross_weights = cross_weights * mask.float().unsqueeze(1)

        cross_weights = self.l1norm(torch.relu(cross_weights), dim=-1)

        if residual:
            cross_weights += torch.eye(cross_weights.size(-1)).to(cross_weights.device)

        wcontext = torch.matmul(cross_weights, context)

        return wcontext, cross_weights
    
    def forward(
        self,
        src=None,
        mask=None,
        query_embed=None,
        pos_embed=None,
        text=None,
        encode_and_save=True,
        text_memory=None,
        img_memory=None,
        text_attention_mask=None,
        memory_cache=None
    ):
    # -----------------------------------------------------------------------
        if encode_and_save:
            # flatten NxCxHxW to HWxNxC
            bs, c, h, w = src.shape
            raw_src = src
            raw_mask = mask
            src = src.flatten(2).permute(2, 0, 1)
            device = src.device
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            mask = mask.flatten(1)

            if self.CLS is not None:
                # We add a CLS token to the image, to be used for contrastive loss

                CLS = self.CLS.weight.view(1, 1, -1).repeat(1, bs, 1)
                # Add the CLS token to the incoming features
                src = torch.cat((CLS, src))

                # Adding zeros as the first token in the sequence to be compatible with the CLS token
                pos_embed = torch.cat((torch.zeros(1, bs, self.d_model, device=device), pos_embed))

                # Adding one mask item to the beginning of the mask to be compatible with CLS token
                cls_pad = torch.zeros(bs, 1).bool().to(device)
                mask = torch.cat((cls_pad, mask), dim=1)

            if self.pass_pos_and_query:
                tgt = torch.zeros_like(query_embed)
            else:
                src, tgt, query_embed, pos_embed = src + 0.1 * pos_embed, query_embed, None, None

            device = src.device
            if isinstance(text[0], str):
                # Encode the text
                tokenized = self.tokenizer.batch_encode_plus(text, padding="longest", return_tensors="pt").to(device)
                encoded_text = self.text_encoder(**tokenized)

                # Transpose memory because pytorch's attention expects sequence first
                text_memory = encoded_text.last_hidden_state.transpose(0, 1)
                # Invert attention mask that we get from huggingface because its the opposite in pytorch transformer
                text_attention_mask = tokenized.attention_mask.ne(1).bool()

                # Resize the encoder hidden states to be of the same d_model as the decoder
                text_memory_resized = self.resizer(text_memory)
            else:
                # The text is already encoded, use as is.
                text_attention_mask, text_memory_resized, tokenized = text

            # Concat on the sequence dimension
            src = torch.cat([src, text_memory_resized], dim=0)
            # For mask, sequence dimension is second
            mask = torch.cat([mask, text_attention_mask], dim=1)
            # Pad the pos_embed with 0 so that the addition will be a no-op for the text tokens
            pos_embed = torch.cat([pos_embed, torch.zeros_like(text_memory_resized)], dim=0)

            # -----------------------------------------------------------------------
            all_img_memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
            img_memory = all_img_memory[-1]
            text_memory = img_memory[-len(text_memory_resized) :]
            assert img_memory.shape[1] == text_memory.shape[1] == tgt.shape[1]

            confidence_set = []
            merged_features_set = []  
            encoder_mask = ~mask

            anchor_features = self.side_downsample0_encoder(img_memory.permute(1, 0, 2))
            anchor = self.aggregate_feature(anchor_features, mask=encoder_mask)
            encoder_features = torch.cat([src[None], all_img_memory[:-1]], dim=0)
         
            for idx in range(self.encoder.num_layers):

                if idx in self.select_layers:
                    target_features = self.side_downsamples_encoder[self.select_layers[idx]](encoder_features[idx].permute(1, 0, 2))
                    target = self.aggregate_feature(target_features, mask=encoder_mask)
                    merged_features, _ = self.cross_attention(anchor_features, target_features, mask=encoder_mask)

                    merged_features_set.append(merged_features)
                    confidence_set.append(self.side_gate_layer_encoder(anchor * target))

            confidence_norm = torch.softmax(torch.cat(confidence_set, dim=1), dim=1)
            all_merged_features = torch.stack(merged_features_set, dim=1)
            merged_features = torch.sum(all_merged_features * confidence_norm[:, :, None, None], dim=1)
            side_img_memory = anchor_features + self.side_projection_encoder(merged_features)
        
            memory_cache = {
                "text_memory_resized": text_memory_resized,
                "text_memory": text_memory,
                "img_memory": img_memory,
                "text_pooled_op": encoded_text.pooler_output if self.CLS is not None else None,
                "img_pooled_op": img_memory[0] if self.CLS is not None else None,  # Return the CLS token
                "mask": mask,
                "text_attention_mask": text_attention_mask,
                "pos_embed": pos_embed,
                "query_embed": query_embed,
                "tokenized": tokenized,
                "side_img_memory": side_img_memory,
                "encoder_mask": encoder_mask
            }
            return memory_cache
            # -----------------------------------------------------------------------            
        else:
            if self.pass_pos_and_query:
                tgt = torch.zeros_like(query_embed)
            else:
                src, tgt, query_embed, pos_embed = src + 0.1 * pos_embed, query_embed, None, None

            assert img_memory.shape[1] == text_memory.shape[1] == tgt.shape[1]

            hs = self.decoder(
                tgt,
                img_memory,
                text_memory,
                memory_key_padding_mask=mask,
                text_memory_key_padding_mask=text_attention_mask,
                pos=pos_embed,
                query_pos=query_embed,
            )

            # -----------------------------------------------------------------------
            confidence_set = []
            merged_features_set = []
            layer_features_set = []

            anchor_features = self.side_downsample0_decoder(hs[-1].permute(1, 0, 2))

            # -----------------------------------------------------------------------
            # related_memory, _ = self.cross_attention(anchor_features, 
            #                                          memory_cache["side_img_memory"], 
            #                                          mask=memory_cache["encoder_mask"],
            #                                          residual=False)
            # anchor_features = anchor_features + self.side_projection0_crosser(related_memory)            
            # -----------------------------------------------------------------------
          
            anchor = self.aggregate_feature(anchor_features)
            decoder_features = torch.cat([tgt[None], hs[:-1]], dim=0)
         
            for idx in range(self.decoder.num_layers):

                if idx in self.select_layers:
                    target_features = self.side_downsamples_decoder[self.select_layers[idx]](decoder_features[idx].permute(1, 0, 2))
                    related_memory, _ = self.cross_attention(target_features, 
                                                             memory_cache["side_img_memory"], 
                                                             mask=memory_cache["encoder_mask"],
                                                             residual=False)
                    target_features = target_features + self.side_projection_crosser[self.select_layers[idx]](related_memory)
                    layer_features_set.append(target_features)
                                      
                    target = self.aggregate_feature(target_features)
                    merged_features, _ = self.cross_attention(anchor_features, target_features)

                    merged_features_set.append(merged_features)
                    confidence_set.append(self.side_gate_layer_decoder(anchor * target))

            confidence_norm = torch.softmax(torch.cat(confidence_set, dim=1), dim=1)
            all_merged_features = torch.stack(merged_features_set, dim=1)
            merged_features = torch.sum(all_merged_features * confidence_norm[:, :, None, None], dim=1)
            
            side_tgt_memory = anchor_features + self.side_projection_decoder(merged_features)
            layer_features_set.append(side_tgt_memory)

            hs = self.side_final_upsample(torch.stack(layer_features_set))

            return hs
            # -----------------------------------------------------------------------

class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):

        output = src
        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask, 
                src_key_padding_mask=src_key_padding_mask, 
                pos=pos
            )
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)
        
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                text_memory=text_memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                text_memory_key_padding_mask=text_memory_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.cross_attn_text = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        # self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    # For now, trying one version where its self attn -> cross attn text -> cross attn image -> FFN
    def forward_post(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)

        # Self attention
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention to text
        # tgt2 = self.cross_attn_text(
        #     query=self.with_pos_embed(tgt, query_pos),
        #     key=text_memory,
        #     value=text_memory,
        #     attn_mask=None,
        #     key_padding_mask=text_memory_key_padding_mask,
        # )[0]
        # tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)

        # Cross attention to image
        tgt2 = self.cross_attn_image(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        assert False, "not implemented yet"
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        text_memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        text_memory_key_padding_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos
            )
        return self.forward_post(
            tgt,
            memory,
            text_memory,
            tgt_mask,
            memory_mask,
            text_memory_key_padding_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


class FeatureResizer(nn.Module):
    """
    This class takes as input a set of embeddings of dimension C1 and outputs a set of
    embedding of dimension C2, after a linear transformation, dropout and normalization (LN).
    """

    def __init__(self, input_feat_size, output_feat_size, dropout, do_ln=True):
        super().__init__()
        self.do_ln = do_ln
        # Object feature encoding
        self.fc = nn.Linear(input_feat_size, output_feat_size, bias=True)
        self.layer_norm = nn.LayerNorm(output_feat_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_features):
        x = self.fc(encoder_features)
        if self.do_ln:
            x = self.layer_norm(x)
        output = self.dropout(x)
        return output


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        pass_pos_and_query=args.pass_pos_and_query,
        text_encoder_type=args.text_encoder_type,
        freeze_text_encoder=args.freeze_text_encoder,
        contrastive_loss=args.contrastive_loss,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
