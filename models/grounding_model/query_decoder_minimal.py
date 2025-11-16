import numpy as np
from typing import List, Optional
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.ops.boxes import box_area
from models.net_utils import MLP, gen_sineembed_for_position, inverse_sigmoid
from .position_encoding import SeqEmbeddingLearned, SeqEmbeddingSine
from .attention import MultiheadAttention
from ..bert_model.bert_module import BertLayerNorm
from easydict import EasyDict as EDict


class QueryDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.MODEL.TASTVG.HIDDEN
        nhead = cfg.MODEL.TASTVG.HEADS
        num_layers = cfg.MODEL.TASTVG.DEC_LAYERS

        self.d_model = d_model
        self.query_pos_dim = cfg.MODEL.TASTVG.QUERY_DIM
        self.nhead = nhead
        self.video_max_len = cfg.INPUT.MAX_VIDEO_LEN
        self.return_weights = cfg.SOLVER.USE_ATTN
        return_intermediate_dec = True

        # MODIFIED: Replace PosDecoder with LinearPosDecoder
        self.decoder = LinearPosDecoder(
            cfg,
            num_layers,
            return_intermediate=return_intermediate_dec,
            return_weights=self.return_weights,
            d_model=d_model,
            query_dim=self.query_pos_dim
        )

        # MODIFIED: Replace TimeDecoder with LinearTimeDecoder
        self.time_decoder = LinearTimeDecoder(
            cfg,
            num_layers,
            return_intermediate=return_intermediate_dec,
            return_weights=True,
            d_model=d_model
        )

        # The position embedding of global tokens
        if cfg.MODEL.TASTVG.USE_LEARN_TIME_EMBED:
            self.time_embed = SeqEmbeddingLearned(self.video_max_len + 1, d_model)
        else:
            self.time_embed = SeqEmbeddingSine(self.video_max_len + 1, d_model)

        self.pos_fc = nn.Sequential(  # 300->768
            BertLayerNorm(256, eps=1e-12),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
            nn.ReLU(True),
            BertLayerNorm(4, eps=1e-12),
        )

        self.time_fc = nn.Sequential(  # 300->768
            BertLayerNorm(256, eps=1e-12),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(True),
            BertLayerNorm(256, eps=1e-12),
        )
        self.time_embed2 = None
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, encoded_info, vis_pos=None, itq=None, isq=None):
        encoded_feature = encoded_info["encoded_feature"]  # len, n_frame, d_model
        encoded_mask = encoded_info["encoded_mask"]  # n_frame, len
        n_vis_tokens = encoded_info["fea_map_size"][0] * encoded_info["fea_map_size"][1]
        encoded_pos = vis_pos.flatten(2).permute(2, 0, 1)
        encoded_pos_s = torch.cat([encoded_pos, torch.zeros_like(encoded_feature[n_vis_tokens:-n_vis_tokens])], dim=0)
        encoded_pos_t = torch.cat([torch.zeros_like(encoded_feature[n_vis_tokens:-n_vis_tokens]), encoded_pos], dim=0)
        # the contextual feature to generate dynamic learnable anchors
        frames_cls = encoded_info["frames_cls"]  # [n_frames, d_model]
        videos_cls = encoded_info["videos_cls"]  # the video-level gloabl contextual token, b x d_model

        b = len(encoded_info["durations"])
        t = max(encoded_info["durations"])
        device = encoded_feature.device

        pos_query, content_query = self.pos_fc(frames_cls), self.time_fc(videos_cls)

        pos_query = pos_query.sigmoid().unsqueeze(1) 
        content_query = content_query.expand(t, content_query.size(-1)).unsqueeze(1)

        query_mask = torch.zeros(b, t).bool().to(device)
        query_time_embed = self.time_embed(t).repeat(1, b, 1)  

        encoded_mask = encoded_mask[:, :-n_vis_tokens]

        tgt_t = torch.zeros(t, b, self.d_model).to(device) if itq is None else itq[None, None, :].expand(t, 1, 256)

        outputs_time = self.time_decoder(
            query_tgt=tgt_t,
            query_content=content_query, 
            query_time=query_time_embed,
            query_mask=query_mask,
            encoded_feature=encoded_feature[n_vis_tokens:],
            encoded_pos=encoded_pos_t,  
            encoded_mask=encoded_mask
        )

        tgt_s = torch.zeros(t, b, self.d_model).to(device) if isq is None else isq[None, None, :].expand(t, 1, 256)

        outputs_pos = self.decoder(
            query_tgt=tgt_s, 
            pred_boxes=pos_query, 
            query_time=query_time_embed,
            query_mask=query_mask, 
            encoded_feature=encoded_feature[:-n_vis_tokens], 
            encoded_pos=encoded_pos_s,  
            encoded_mask=encoded_mask, 
        )

        return outputs_pos, outputs_time


# MODIFIED: Simplified PosDecoder with linear heads
class LinearPosDecoder(nn.Module):
    def __init__(self, cfg, num_layers, return_intermediate=False, return_weights=False, d_model=256, query_dim=4):
        super().__init__()
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate
        self.return_weights = False
        self.query_dim = query_dim
        self.d_model = d_model

        # Linear prediction heads instead of attention layers
        self.bbox_embed = nn.ModuleList([
            MLP(d_model, d_model, 4, 3) for _ in range(num_layers)
        ])
        
        self.query_update = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_layers - 1)
        ])

    def forward(
            self,
            query_tgt: Optional[Tensor] = None,  
            pred_boxes: Optional[Tensor] = None,  
            query_time=None,  
            query_mask: Optional[Tensor] = None,  
            encoded_feature: Optional[Tensor] = None, 
            encoded_pos: Optional[Tensor] = None,  
            encoded_mask: Optional[Tensor] = None,
    ):
        ref_anchors = []
        
        # Pool encoded features across spatial dimension
        pooled_features = encoded_feature.mean(dim=0)  # [n_frames, d_model]
        t, b, _ = query_tgt.shape
        pooled_features = pooled_features.unsqueeze(1).expand(-1, b, -1)  # [t, b, d_model]
        
        # Combine with query
        query_tgt = query_tgt + pooled_features + query_time

        for layer_id in range(self.num_layers):
            # Predict boxes using linear head
            new_pred_boxes = self.bbox_embed[layer_id](query_tgt).sigmoid()
            ref_anchors.append(new_pred_boxes)
            
            # Update query for next layer
            if layer_id < self.num_layers - 1:
                query_tgt = query_tgt + self.query_update[layer_id](query_tgt)

        return torch.stack(ref_anchors).transpose(1, 2)


# MODIFIED: Simplified TimeDecoder with linear heads
class LinearTimeDecoder(nn.Module):
    def __init__(self, cfg, num_layers, return_intermediate=False, return_weights=False, d_model=256):
        super().__init__()
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate
        self.return_weights = return_weights
        
        # Linear prediction heads instead of attention layers
        self.time_embed = nn.ModuleList([
            MLP(d_model, d_model, d_model, 2) for _ in range(num_layers)
        ])
        
        self.query_update = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_layers - 1)
        ])

    def forward(
            self,
            query_tgt: Optional[Tensor] = None,
            query_content: Optional[Tensor] = None,
            query_time: Optional[Tensor] = None,
            query_mask: Optional[Tensor] = None,
            encoded_feature: Optional[Tensor] = None,
            encoded_pos: Optional[Tensor] = None,
            encoded_mask: Optional[Tensor] = None
    ):
        intermediate = []
        
        # Pool encoded features
        pooled_features = encoded_feature.mean(dim=0)  # [n_frames, d_model]
        t, b, _ = query_tgt.shape
        pooled_features = pooled_features.unsqueeze(1).expand(-1, b, -1)  # [t, b, d_model]
        
        # Combine with query
        query_tgt = query_tgt + pooled_features + query_content + query_time

        for layer_id in range(self.num_layers):
            # Predict temporal features using linear head
            query_tgt_new = self.time_embed[layer_id](query_tgt)
            
            if self.return_intermediate:
                intermediate.append(self.norm(query_tgt_new))
            
            # Update query for next layer
            if layer_id < self.num_layers - 1:
                query_tgt = query_tgt + self.query_update[layer_id](query_tgt_new)
            else:
                query_tgt = query_tgt_new

        if self.return_intermediate:
            return torch.stack(intermediate).transpose(1, 2)
        else:
            return self.norm(query_tgt).unsqueeze(0).transpose(1, 2)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
