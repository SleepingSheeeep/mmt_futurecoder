# coding=utf-8
# @Time    : 2021/3/4 
# @Author  : ZTY

import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_num, output_num):
        super(GRUModel, self).__init__()
        self.hidden_size = output_num
        self.grucell = nn.GRUCell(input_num, output_num)

    def forward(self, x, hid):
        # x : seq_lenth * batch * embed_size(=input_size)
        # hid : batch * embed_size(=hidden_size)
        output = []
        if hid is None:
            hid = torch.randn(x.shape[0], self.hidden_size)
        for i in range(x.shape[0]):
            hx = self.grucell(x[i], hid)
            output.append(hx)
        # print(output)
        output = torch.stack(output)

        return output, hx