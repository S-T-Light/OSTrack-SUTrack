"""
Basic SUTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from .encoder import build_encoder
# from .clip import build_text_encoder
# from .decoder import build_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh


class SUTrack(nn.Module):
    """ This is the base class for SUTrack """

    def __init__(self, encoder, box_head, aux_loss=False, head_type="CENTER"):
        """ Initializes the model """
        super().__init__()
        self.encoder = encoder
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template_list=None, search_list=None):
        x = self.encoder(template_list, search_list)

        out = self.forward_head(x, None)
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the encoder, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # 改为取前面的search部分，因为是xz
        enc_opt = cat_feature[:, :self.feat_len_s]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_sutrack(cfg, training=True):
    # current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    # pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    # if cfg.MODEL.PRETRAIN_FILE and ('SUTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
    #     pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    # else:
    #     pretrained = ''
    
    encoder = build_encoder(cfg)

    # 没有用sutrack的decoder，还是ostrack的head
    box_head = build_box_head(cfg, encoder)

    model = SUTrack(
        encoder,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.DECODER.TYPE,
    )

    # if 'SUTrack' in cfg.MODEL.PRETRAIN_FILE and training:
    #     checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
    #     missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
    #     print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
