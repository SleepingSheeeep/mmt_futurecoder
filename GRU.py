# coding=utf-8
# @Time    : 2021/3/4 
# @Author  : ZTY

import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_num, hidden_num, output_num):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_num
        self.grucell = nn.GRUCell(input_num, hidden_num)
        self.out_linear = nn.Linear(hidden_num, output_num)

    def forward(self, x, hid):
        # x : seq_lenth * batch * embed_size(=input_size)
        # hid : seq_lenth * batch * embed_size(=hidden_size)
        # return :[
        #
        # ]
        if hid is None:
            hid = torch.randn(x.shape[0], x.shape[1], self.hidden_size)
        y = torch.zeros(x.shape[0], x.shape[1], self.hidden_size)
        for i in range(x.shape[0]):
            #
            y[i] = self.grucell(x[i], hid[i])
            # print("y[i]: {}".format(y[i].shape))

        # print("y: {}".format(y.shape))

        return y, hid.detach()