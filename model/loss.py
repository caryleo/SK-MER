'''
Description: 
Autor: Gary Liu
Date: 2021-07-02 11:56:32
LastEditors: Gary Liu
LastEditTime: 2022-09-07 22:21:43
'''
import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy_loss(output, target):
    return F.cross_entropy(output, target)

def cross_entropy_loss_with_weight(output, target, weight):
    return F.cross_entropy(output, target, weight=weight)

def mse_loss(output, target):
    return F.mse_loss(output, target)


def info_nce_loss_ck(feature_batch_list, kecrs_batch_list, kecrs_pos_batch_list, kecrs_neg_batch_list, temperature):
    # 正负样本之间的对比
    knowledge_batch = F.normalize(torch.cat(kecrs_batch_list, dim=0)) # batch*seq_len, d_knowledge
    feature_batch = F.normalize(torch.cat(feature_batch_list, dim=0)) # batch*seq_len, d_knowledge
    knowledge_pos_batch = F.normalize(torch.cat(kecrs_pos_batch_list, dim=0)) # batch*seq_len, d_knowledge
    knowledge_neg_batch = F.normalize(torch.cat(kecrs_neg_batch_list, dim=0)) # batch*seq_len, d_knowledge

    # print(knowledge_pos_batch)
    # print(knowledge_neg_batch)
    # print(feature_batch)

    logits_pos = (knowledge_pos_batch * feature_batch).sum(dim=-1, keepdim=True) # batch*seq_len, 1
    logits_neg = (knowledge_neg_batch * feature_batch).sum(dim=-1, keepdim=True) # batch*seq_len, 1
    logits = torch.cat([logits_pos, logits_neg], dim=-1) # batch*seq_len, 2
    labels = torch.zeros(len(logits), dtype=torch.long, device=logits.device) # batch*seq_len


    return F.cross_entropy(logits / temperature, labels)
