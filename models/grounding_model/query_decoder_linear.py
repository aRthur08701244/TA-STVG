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

        self.decoder = PosDecoder(
            cfg,
            num_layers,
            return_intermediate=return_intermediate_dec,
            return_weights=self.return_weights,
            d_model=d_model,
            query_dim=self.query_pos_dim
        )

        self.time_decoder = TimeDecoder(
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


class PosDecoder(nn.Module):
    def __init__(self, cfg, num_layers, return_intermediate=False, return_weights=False, d_model=256, query_dim=4):
        super().__init__()
        # keep interface
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.return_weights = False
        self.query_dim = query_dim
        self.d_model = d_model

        # Simple MLP that maps from d_model -> d_model (feature update)
        dim_feed = max(d_model, 256)
        self.update_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feed),
                nn.ReLU(inplace=True),
                nn.LayerNorm(dim_feed),
                nn.Linear(dim_feed, d_model),
                nn.ReLU(inplace=True),
                nn.LayerNorm(d_model),
            ) for _ in range(num_layers)
        ])

        # bbox head: map d_model -> 4 (box coords)
        self.bbox_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(inplace=True),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, query_dim)  # final -> 4
            ) for _ in range(num_layers)
        ])

        # keep some attribute names used elsewhere (even if unused)
        self.query_scale = None
        self.ref_point_head = None
        self.bbox_embed = None
        self.gf_mlp = None
        self.gf_mlp2 = None
        self.fuse_linear = None
        self.norm = nn.LayerNorm(d_model)

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
        """
        Minimal MLP-based substitute for the original transformer decoder.
        - pred_boxes: expected shape (t, b, 4) or (t, 1, 4) (we handle broadcasting).
        - query_tgt: expected shape (t, b, d_model) -- used as initial query features if provided,
                     otherwise zeroed features are created.
        Returns: stacked boxes of shape (num_layers, b, t, 4).
        """
        device = pred_boxes.device if pred_boxes is not None else (query_tgt.device if query_tgt is not None else torch.device('cpu'))

        # ensure query_tgt: t x b x d_model
        if query_tgt is None:
            # create zeros with same t and b inferred from pred_boxes
            if pred_boxes is None:
                raise ValueError("PosDecoder forward expects pred_boxes or query_tgt.")
            t, b = pred_boxes.shape[0], pred_boxes.shape[1]
            query_tgt = torch.zeros(t, b, self.d_model, device=device)
        else:
            t, b, _ = query_tgt.shape

        # normalize/expand pred_boxes to t x b x query_dim
        if pred_boxes is None:
            # start from zeros
            pred_boxes = torch.zeros(t, b, self.query_dim, device=device)
        else:
            # if pred_boxes shape is (t,1,4) expand to (t,b,4)
            if pred_boxes.shape[1] == 1 and b != 1:
                pred_boxes = pred_boxes.expand(t, b, pred_boxes.shape[2]).contiguous()

        layer_boxes = []
        x = query_tgt  # t x b x d_model

        for i in range(self.num_layers):
            # apply update MLP independently per query (merge time & batch dims for MLP)
            x_flat = x.view(t * b, -1)
            x_up = self.update_mlps[i](x_flat)
            x_up = x_up.view(t, b, -1)
            # produce bbox from updated features
            bbox_flat = self.bbox_heads[i](x_up.view(t * b, -1))
            bbox = bbox_flat.view(t, b, self.query_dim).sigmoid()  # sigmoid to keep in [0,1]
            layer_boxes.append(bbox)  # t x b x 4
            # optionally feed the bbox back as additional input (concat projected back to d_model)
            # simple residual: add a linear-projection of bbox into feature space (here just expand)
            # keep it simple: add small perturbation from bbox
            perturb = F.pad(bbox, (0, self.d_model - self.query_dim)) if self.d_model > self.query_dim else bbox[..., :self.d_model]
            perturb = perturb.to(x.dtype)
            x = x + 0.1 * perturb  # small influence of predicted box

        # stack -> (num_layers, t, b, 4) then transpose to (num_layers, b, t, 4)
        stacked = torch.stack(layer_boxes)  # L x t x b x 4
        stacked = stacked.transpose(1, 2)   # L x b x t x 4
        return stacked



class PosDecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Decoder Self-Attention
        d_model = cfg.MODEL.TASTVG.HIDDEN
        nhead = cfg.MODEL.TASTVG.HEADS
        dim_feedforward = cfg.MODEL.TASTVG.FFN_DIM
        dropout = cfg.MODEL.TASTVG.DROPOUT
        activation = "relu"
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_qtime_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_ktime_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_qtime_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)

        self.from_scratch_cross_attn = cfg.MODEL.TASTVG.FROM_SCRATCH
        self.cross_attn_image = None
        self.cross_attn = None
        self.tgt_proj = None

        if self.from_scratch_cross_attn:
            self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)
        else:
            self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
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

     
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
            self,
            query_tgt: Optional[Tensor] = None,
            query_pos: Optional[Tensor] = None,
            query_time_embed=None,
            query_sine_embed=None,
            query_mask: Optional[Tensor] = None,
            encoded_feature: Optional[Tensor] = None,
            encoded_pos: Optional[Tensor] = None,
            encoded_mask: Optional[Tensor] = None,
            good_rf=None,
            is_first=False,
    ):
        # Apply projections here
        # shape: num_queries x batch_size x 256
        # ========== Begin of Self-Attention =============
        q_content = self.sa_qcontent_proj(query_tgt)  # target is the input of the first decoder layer. zero by default.
        q_time = self.sa_qtime_proj(query_time_embed)
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(query_tgt)
        k_time = self.sa_ktime_proj(query_time_embed)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(query_tgt)

        q = q_content + q_time + q_pos
        k = k_content + k_time + k_pos

        # Temporal Self attention
        tgt2, weights = self.self_attn(q, k, value=v)
        tgt = query_tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # ========== End of Self-Attention =============

        # ========== Begin of Cross-Attention =============
        # Time Aligned Cross attention
        t, b, c = tgt.shape    # b is the video number
        n_tokens, bs, f = encoded_feature.shape   # bs is the total frames in a batch
        assert f == c   # all the token dim should be same

        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(encoded_feature)
        v = self.ca_v_proj(encoded_feature)

        k_pos = self.ca_kpos_proj(encoded_pos)

        if is_first:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(t, b, self.nhead, c // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(t, b, self.nhead, c // self.nhead)

        if self.from_scratch_cross_attn:
            q = torch.cat([q, query_sine_embed], dim=3).view(t, b, c * 2)
        else:
            q = (q + query_sine_embed).view(t, b, c)
            q = q + self.ca_qtime_proj(query_time_embed)

        k = k.view(n_tokens, bs, self.nhead, f//self.nhead)
        k_pos = k_pos.view(n_tokens, bs, self.nhead, f//self.nhead)

        if self.from_scratch_cross_attn:
            k = torch.cat([k, k_pos], dim=3).view(n_tokens, bs, f * 2)
        else:
            k = (k + k_pos).view(n_tokens, bs, f)

        # extract the actual video length query
        device = tgt.device
        if self.from_scratch_cross_attn:
            q_cross = torch.zeros(1,bs,2 * c).to(device)
        else:
            q_cross = torch.zeros(1,bs,c).to(device)

        q_clip = q[:,0,:]   # t x f
        q_cross[0,:] = q_clip[:]

        if self.from_scratch_cross_attn:
            tgt2, _ = self.cross_attn(
                query=q_cross,
                key=k,
                value=v
            )
        else:
            tgt2, _ = self.cross_attn_image(
                query=q_cross,
                key=k,
                value=v
            )

        # reshape to the batched query
        tgt2_pad = torch.zeros(1,t*b,c).to(device)

        tgt2_pad[0, :] = tgt2[0, :]

        tgt2 = tgt2_pad
        tgt2 = tgt2.view(b, t, f).transpose(0, 1)  # 1x(b*t)xf -> bxtxf -> txbxf

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm4(tgt)
        return tgt, weights



class TimeDecoder(nn.Module):
    def __init__(self, cfg, num_layers, return_intermediate=False, return_weights=False, d_model=256):
        super().__init__()
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.return_weights = return_weights
        self.norm = nn.LayerNorm(d_model)
        self.d_model = d_model

        # MLPs for updating temporal queries
        dim_feed = max(d_model, 256)
        self.update_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feed),
                nn.ReLU(inplace=True),
                nn.LayerNorm(dim_feed),
                nn.Linear(dim_feed, d_model),
                nn.ReLU(inplace=True),
                nn.LayerNorm(d_model),
            ) for _ in range(num_layers)
        ])

        # optionally a small projection for query_content (video-level) -> d_model
        self.content_proj = nn.Linear(d_model, d_model)
        # keep simple time embedding projection if provided
        self.time_proj = nn.Linear(d_model, d_model)

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
        """
        MLP-based time decoder:
        - query_tgt: t x b x d_model (initial or zeros)
        - query_content: t x d_model (or t x 1 x d_model?) -- expand if needed
        - query_time: t x b x d_model (positional/time embeddings)
        Returns: stacked intermediate query features with shape (num_layers, b, t, d_model)
        """
        # infer shapes
        if query_tgt is None:
            # infer t,b from query_time or query_content
            if query_time is not None:
                t, b, _ = query_time.shape
                query_tgt = torch.zeros(t, b, self.d_model, device=query_time.device)
            elif query_content is not None:
                # query_content may be (t, d) or (t, 1, d)
                if query_content.dim() == 2:
                    t = query_content.shape[0]
                    b = 1
                else:
                    t, b, _ = query_content.shape
                query_tgt = torch.zeros(t, b, self.d_model, device=query_content.device)
            else:
                raise ValueError("TimeDecoder forward expects at least one of query_tgt/query_time/query_content.")

        t, b, _ = query_tgt.shape
        intermediates = []

        # normalize/expand query_content and query_time to t x b x d_model if possible
        if query_content is not None:
            if query_content.dim() == 2:
                # t x d -> t x 1 x d -> expand to t x b x d
                qc = query_content.unsqueeze(1).expand(t, b, -1)
            elif query_content.dim() == 3 and query_content.shape[1] == 1:
                qc = query_content.expand(t, b, query_content.shape[2])
            else:
                qc = query_content
            qc = self.content_proj(qc.view(t * b, -1)).view(t, b, -1)
        else:
            qc = torch.zeros(t, b, self.d_model, device=query_tgt.device)

        if query_time is not None:
            qt = self.time_proj(query_time.view(t * b, -1)).view(t, b, -1)
        else:
            qt = torch.zeros(t, b, self.d_model, device=query_tgt.device)

        x = query_tgt  # t x b x d_model
        for i in range(self.num_layers):
            # combine current state with content & time signals
            x_input = x + qc * 0.1 + qt * 0.1
            x_flat = x_input.view(t * b, -1)
            x_up = self.update_mlps[i](x_flat)
            x = x_up.view(t, b, -1)
            # save normalized intermediate if required
            if self.return_intermediate:
                intermediates.append(self.norm(x))
            else:
                # to keep same semantics as original when return_intermediate True, still collect
                intermediates.append(self.norm(x))

        # stack intermediates into (num_layers, t, b, d_model) -> transpose to (num_layers, b, t, d_model)
        stacked = torch.stack(intermediates).transpose(1, 2)
        return stacked


class TimeDecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.MODEL.TASTVG.HIDDEN
        nhead = cfg.MODEL.TASTVG.HEADS
        dim_feedforward = cfg.MODEL.TASTVG.FFN_DIM
        dropout = cfg.MODEL.TASTVG.DROPOUT
        activation = "relu"

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_image = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

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

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

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
        q = k = self.with_pos_embed(query_tgt, query_time)

        # Temporal Self attention
        query_tgt2, weights = self.self_attn(q, k, value=query_tgt, key_padding_mask=query_mask)
        query_tgt = self.norm1(query_tgt + self.dropout1(query_tgt2))

        query_tgt2, _ = self.cross_attn_image(
            query=query_tgt.permute(1, 0, 2),
            key=self.with_pos_embed(encoded_feature, encoded_pos),
            value=encoded_feature,
            key_padding_mask=encoded_mask,
        )

        query_tgt2 = query_tgt2.transpose(0, 1)  # 1x(b*t)xf -> bxtxf -> txbxf
        query_tgt = self.norm3(query_tgt + self.dropout3(query_tgt2))

        # FFN
        query_tgt2 = self.linear2(self.dropout(self.activation(self.linear1(query_tgt))))
        query_tgt = query_tgt + self.dropout4(query_tgt2)
        query_tgt = self.norm4(query_tgt)
        return query_tgt, weights


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
