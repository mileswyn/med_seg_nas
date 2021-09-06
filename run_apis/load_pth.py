import torch
import os

dir = '/hdd1/wyn/DenseNAS/log/20201225-091613-search_res_promise12/output/weights_best.pt'

a = torch.load(dir)

print(dir.data.shape)
