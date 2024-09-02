#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import re
import numpy as np
import pandas as pd
import torch
import torch.utils.data.dataset as Dataset
from pathlib import Path
import torch.utils.data.dataloader as DataLoader
import os
from torch import nn
import torch.nn.functional as F
from typing import List,Dict,Union
MultiHeadAttention = nn.MultiheadAttention
from loguru import logger
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from copy import deepcopy
import sys
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from typing import List
from pathlib import Path
import yaml
import argparse

parser = argparse.ArgumentParser(description='Process sequence to predict the mutation.')
parser.add_argument('-s', '--seqfile', type=str, help='Input the sequence file')
parser.add_argument('-m', '--mutation', type=str, help='Mutation need to Predict')
parser.add_argument('-f', '--featurePath', type=str, help='The ESM feature file for the sequence')
parser.add_argument('-o', '--output', type=str, help='Path to save the predition file')

args = parser.parse_args()
seq_file_path = args.seqfile
seq_file = seq_file_path.split('/')[-1]
mut = args.mutation
fea_path = args.featurePath
output_path = args.output

os.chdir('~/MutStab-ESM/example/')

f = open(seq_file_path)
seqdict = dict()
for line in f:
    if line.startswith('>'):
        name=line.replace('\n','').split()[0]
        seqdict[name]=''
    else:
        seqdict[name]+=line.replace('\n','').strip()
f.close()

mutpos = int(mut[1:-1])
ref_embedding_need = torch.load(fea_path + seq_file.split('.')[0] + '.pt')['representations'][36][mutpos-1]
mut_embedding_need = torch.load(fea_path + seq_file.split('.')[0] + '_' + mut + '.pt')['representations'][36][mutpos-1]

#######################################
# run_Prediction

# module
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = nn.Linear(embed_dim, embed_dim)
        self.key_dense = nn.Linear(embed_dim, embed_dim)
        self.value_dense = nn.Linear(embed_dim, embed_dim)
        self.combine_heads = nn.Linear(embed_dim, embed_dim)

    def attention(self, query, key, value, mask=None):
        score = torch.matmul(query, key.transpose(-2, -1))
        dim_key = torch.tensor(key.shape[-1], dtype=torch.float32)
        scaled_score = score / torch.sqrt(dim_key)
        if mask is not None:
            scaled_score += (mask * -1e9)
        weights = F.softmax(scaled_score, dim=-1)
        output = torch.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.projection_dim)
        return x.permute(0, 2, 1, 3)
    def forward(self, inputs, mask=None):
        batch_size = inputs.size(0)
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(query, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(key, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(value, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value, mask)
        attention = attention.permute(0, 2, 1, 3)  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = attention.reshape(batch_size, -1, self.embed_dim)  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(concat_attention)  # (batch_size, seq_len, embed_dim)
        return output
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
        })
        return config

class transformer_block(nn.Module):
    def __init__(self, input_dim,output_dim, head,dropout_rate):
        super(transformer_block, self).__init__()
        self.attention1 = MultiHeadSelfAttention(input_dim,head) if head is not None else None     
        self.layer_norm1 = nn.LayerNorm(normalized_shape=input_dim , eps = 1e-06)
        self.layer_norm2 = nn.LayerNorm(normalized_shape=input_dim , eps = 1e-06)
        self.fnn = nn.Sequential(nn.Linear(input_dim, output_dim),nn.ReLU(),nn.Linear(output_dim, output_dim))
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.dropout2 = nn.Dropout(p=dropout_rate)
    def forward(self,x):
        x1 = self.attention1(x)
        x1 = self.dropout1(x1)
        x2 = self.layer_norm1(x + x1)
        x3 = self.fnn(x2)
        x3 = self.dropout2(x3)
        x4 = self.layer_norm2(x2 + x3)
        return x4
        
class TokenPositionEmbedding(nn.Module):
    def __init__(self, maxlen, pos_embed_dim):
        super().__init__()
        self.maxlen = maxlen
        self.pos_embed_dim = pos_embed_dim
        self.pos_emb = nn.Embedding(maxlen, pos_embed_dim)
        self.init_weights()
    def init_weights(self):
        position_enc = np.array([
            [pos / np.power(10000, 2 * (i // 2) / self.pos_embed_dim) for i in range(self.pos_embed_dim)]
            if pos != 0 else np.zeros(self.pos_embed_dim)
            for pos in range(self.maxlen)
        ])
        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        self.pos_emb.weight.data.copy_(torch.tensor(position_enc))

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        positions = torch.arange(0, seq_len).unsqueeze(0).repeat(batch_size, 1).to(x.device)
        position_emb = self.pos_emb(positions)
        x1 = x + position_emb
        return x1

class cnn1d_Residual_block(nn.Module):
    def __init__(self, in_channel, out_channel,out_channel2, relu_par0, pool_kernel_size, stride=1):
        super(cnn1d_Residual_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, 3, stride=stride, padding=1)
        nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.conv2 = nn.Conv1d(out_channel, out_channel2, 3, stride=stride, padding=1)
        nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        self.leakyReLu1 = nn.LeakyReLU(negative_slope=relu_par0)
        self.maxpooling1 = nn.MaxPool1d(pool_kernel_size)

    def forward(self, x):
        x1 = x.permute(0, 2, 1)
        x2 = self.conv1(x1.float())
        x3 = self.leakyReLu1(x2)
        x4 = self.conv2(x3.float())
        x5 = x4.permute(0, 2, 1)
        x6 = torch.add(x, x5)
        x7 = self.leakyReLu1(x6)
        x8 = self.maxpooling1(x7)
        x9 = x8.permute(0, 2, 1)
        return x9

class MLPs_block(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(self.embed_dim, self.embed_dim // 2)
        self.bn1 = nn.BatchNorm1d(self.embed_dim // 2)
        self.fc2 = nn.Linear(self.embed_dim // 2, self.embed_dim // 8)
        self.bn2 = nn.BatchNorm1d(self.embed_dim // 8)
        self.fc3 = nn.Linear(self.embed_dim // 8, 1)
        self.leakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.leakyReLU(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.leakyReLU(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

class PremPS2(nn.Module):
    def __init__(self, input_dim=2560):
        super().__init__()
        self.embed_dim = input_dim  # 添加这一行
        self.bn0 = nn.BatchNorm1d(input_dim)
        self.PE_mut = TokenPositionEmbedding(maxlen = input_dim, pos_embed_dim = 1)
        self.transa1 = transformer_block(1,1,head = 1,dropout_rate = 0.1)
        self.mlps = MLPs_block(self.embed_dim)
    def forward(self, mutation):
        mutation = self.bn0(mutation)
        mutation = torch.unsqueeze(mutation, 1)
        mutation = mutation.permute(0,2,1)
        mutation = self.PE_mut(mutation)
        mutation = self.transa1(mutation.to(torch.float32)) 
        mutation = mutation.permute(0,2,1)
        mutation = torch.squeeze(mutation)
        results1 = self.mlps(mutation.unsqueeze(0))
        return results1

# run
mutsta_esm = PremPS2()
mutsta_esm_weights = torch.load('~/MutStab-ESM/Model/MLP_TE2_esm.pt',map_location=torch.device('cpu'))
mutsta_esm.load_state_dict(mutsta_esm_weights)
mutsta_esm.eval()

# predict
input_sub_embedding = ref_embedding_need - mut_embedding_need
torch_input_sub_embedding = input_sub_embedding.unsqueeze(0)
with torch.no_grad():
    output = mutsta_esm(torch_input_sub_embedding)

if output > 0.5:
    output_label = 1
    out_type = 'Stabilizing'
else :
    output_label = 0
    out_type = 'Destablizing'

result_list = []
outlist = [seq_file.split('.')[0], mut, output.tolist()[0][0], output_label, out_type]
out_col = ['Protein', 'mutation', 'output_logits', 'label', 'type']
result_list.append(outlist)
outdat = pd.DataFrame(result_list,columns=out_col)
outdat.to_csv(output_path + '/' + seq_file.split('.')[0] + '_' + mut + '.txt',index=False,sep='\t')

print('Finish predict!')
