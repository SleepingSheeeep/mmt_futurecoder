# conding=utf-8

import torch
import torch.nn as nn

class ImgEmbeddings(nn.Module):
    """
    Img embeddings for encoder/decoder.
    """
    def __init__(self, hidden_size):
        super(ImgEmbeddings, self).__init__()
        dropout = 0.3
        self.pre_dp = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.img_to_emb = nn.Linear(hidden_size, 512, bias=True)

    def forward(self, img_feats):
        if self.hidden_size == 2048:
            # print(img_feats.shape[0], " ", img_feats.shape[1])
            img_feats = img_feats.reshape(img_feats.shape[0], img_feats.shape[1], -1).permute(0, 2, 1)
        img_emb = self.img_to_emb(img_feats)
        img_emb = self.pre_dp(img_emb)
        return img_emb