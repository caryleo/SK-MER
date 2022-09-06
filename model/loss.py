'''
Description: 
Autor: Gary Liu
Date: 2021-07-02 11:56:32
LastEditors: Gary Liu
LastEditTime: 2021-12-17 20:40:35
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

def info_nce_loss(output_anchor, output_pos, output_neg, temperature=1.0):
    score_pos = torch.div(torch.cosine_similarity(output_anchor, output_pos), temperature) # batch
    score_neg = torch.div(torch.cosine_similarity(output_anchor, output_neg), temperature) # batch
    
    logits = torch.cat([score_pos.unsqueeze(1), score_neg.unsqueeze(1)], dim=1) # batch, n_pos+n_neg(2)
    # labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)
    # output = torch.cat([output_pos.unsqueeze(1), output_neg.unsqueeze(1)], dim=1) # batch, n_pos+n_neg(2), dim_feature 
    # logits = torch.div(torch.matmul(output_anchor.unsqueeze(1), output.transpose(-2, -1)), temperature)  # batch, n_pos+n_neg(2)

    logits_max = logits.max(dim=-1, keepdim=True)[0] # batch, 1, n_pos+n_neg(2)
    logits = logits - logits_max.detach()  # batch, 1, n_pos+n_neg(2)
    
    labels = torch.zeros(logits.size(0), dtype=torch.long).to(logits.device)  # batch

    # print(logits.device, labels.device)
    # print(logits.size(), labels.size())

    # output_anchor_normed = F.normalize(output_anchor, dim=-1) # batch, class
    # output_pos_normed = F.normalize(output_pos, dim=-1) if output_pos is not None else None # batch, class
    # output_neg_normed = F.normalize(output_neg, dim=-1) if output_pos is not None else None # batch, class

    # if output_neg_normed is not None:
    #     score_pos = torch.sum(output_anchor_normed * output_pos_normed, dim=1, keepdim=True) # batch, 1
    #     score_neg = output_anchor_normed @ output_neg_normed.transpose(-2, -1) # batch, batch

    #     logits = torch.cat([score_pos, score_neg], dim=-1) # batch, 1+batch
    #     labels = torch.zeros(len(logits), dtype=torch.long, device=output_anchor_normed.device)
    # else:
    #     # 未指定负样本，其他的样本的正样本就都是负样本(对角线)
    #     logits = output_anchor_normed @ output_pos_normed.transpose(-2, -1) # batch, batch
    #     labels = torch.arange(len(output_anchor_normed), device=output_anchor_normed.device)

    # return F.cross_entropy(logits /temperature, labels)
    return F.cross_entropy(logits, labels)

def triplet_cl_loss(output_origin, output_pos, output_neg, margin=1.0):
    score_pos = torch.cosine_similarity(output_origin, output_pos)
    score_neg = torch.cosine_similarity(output_origin, output_neg)

    # return F.relu(score_pos - score_neg + margin).mean()
    return F.relu(margin - score_pos + score_pos).mean()

def triplet_cl_loss_2(target, output_pos, output_neg, margin=1.0):
    score_pos = F.nll_loss(output_pos, target.squeeze(1))
    score_neg = F.nll_loss(output_neg, target.squeeze(1))

    # return F.relu(score_pos - score_neg + margin)
    return F.relu(score_pos - score_neg + margin)

def triplet_cl_loss_3(target, output_pos, output_neg, margin=1.0):
    print(target.size(), output_pos.size(), output_neg.size())

    score_pos = F.nll_loss(output_pos, target.squeeze(1))
    score_neg = F.nll_loss(output_neg, target.squeeze(1))
    
    print(score_pos, score_neg)

    # return F.relu(score_pos - score_neg + margin)
    return score_pos + F.relu(margin - score_neg)

def distance_smooth_l1_loss(feature_processed_batch, kecrs_processed_batch, kecrs_processed_batch_neg, k_index_neg_list):
    feature_list = list()
    knowledge_pos_list = list()
    knowledge_neg_list = list()
    for i in range(feature_processed_batch.size(0)):
        feature_list.append(feature_processed_batch[k_index_neg_list[i]])
        knowledge_pos_list.append(kecrs_processed_batch[k_index_neg_list[i]])
        knowledge_neg_list.append(kecrs_processed_batch_neg[k_index_neg_list[i]])

    feature = torch.stack(feature_list, dim=0)
    knowledge_pos = torch.stack(knowledge_pos_list, dim=0)
    knowledge_neg = torch.stack(knowledge_neg_list, dim=0)

    distance_pos = F.smooth_l1_loss(feature, knowledge_pos)
    distance_neg = F.smooth_l1_loss(feature, knowledge_neg)

    return distance_pos, distance_neg

def info_nce(score_list, temperature):
    # 第一个方案，score+list已经算好了，直接进行计算
    loss = 0.0
    loss_num = 0
    for score in score_list: # batch
        labels = torch.arange(len(score), device=score.device)

        loss += F.cross_entropy(score / temperature, labels)

        loss_num += 1

    assert loss_num == len(score_list)

    return loss / loss_num

def info_nce_2(score_batch, temperature):
    # 第二个方案，scorebatch已经算好了，直接进行计算
    logits = score_batch # batch*seq_len, batch*seq_len
    labels = torch.arange(len(score_batch), device=score_batch.device)  # batch*seq_len

    return F.cross_entropy(logits / temperature, labels)

def info_nce_mlp(knowledge_batch_list, feature_batch_list, temperature):
    loss = 0.0
    loss_num = 0

    for k, f in zip(knowledge_batch_list, feature_batch_list): # batch
        k_normed = F.normalize(k, dim=1) # seq_len, d_knowledge
        f_normed = F.normalize(f, dim=1) # seq_len, d_knowledge

        logits = k @ f.transpose(-2, -1) # seq_len, seq_len
        labels = torch.arange(len(k), device=k.device)

        loss += F.cross_entropy(logits / temperature, labels)
        loss_num += 1

    assert loss_num == len(knowledge_batch_list)

    return loss / loss_num


def info_nce_mlp_2(knowledge_batch_list, feature_batch_list, temperature):
    loss = 0.0
    loss_num = 0

    knowledge_batch = torch.cat(knowledge_batch_list, dim=0) # batch*seq_len, d_knowledge
    feature_batch = torch.cat(feature_batch_list, dim=0) # batch*seq_len, d_knowledge

    knowledge_normed = F.normalize(knowledge_batch, dim=1) # batch*seq_len, d_knowledge
    feature_normed = F.normalize(feature_batch, dim=1) # batch*seq_len, d_knowledge

    logits = knowledge_normed @ feature_normed.transpose(-2, -1) # batch*seq_len, batch*seq_len

    labels = torch.arange(len(knowledge_batch), device=knowledge_batch.device)

    assert logits.size(0) == labels.size(0)

    return F.cross_entropy(logits / temperature, labels)

# def triplet_2(score_pos, score_neg, k_index_neg_list, margin=1.0):
#     score_pos_list, score_neg_list = list(), list() # batch, 1
#     for i, (score_p, score_n) in enumerate(zip(score_pos, score_neg)):
#         score_pos_list.append(score_pos[i][k_index_neg_list[i]]) # batch, 1
#         score_neg_list.append(score_neg[i][k_index_neg_list[i]]) # batch, 1

#     # score_pos_norm, score_neg_norm = F.normalize(score_pos, p=1, dim=-1), F.normalize(score_neg, p=1, dim=-1)
#     score_pos = torch.stack(score_pos_list, dim=0) # batch, 1
#     score_neg = torch.stack(score_neg_list, dim=0) # batch, 1

#     # return F.relu(score_pos - score_neg + margin).sum()
#     return F.relu(score_pos - score_neg + margin).mean()

#     # return max(0, distance_pos - distance_neg + margin)

# def ranking_2(score_pos, score_neg, k_index_neg_list, margin=1.0):
#     score_pos_list, score_neg_list = list(), list() # batch, 1
#     for i, (score_p, score_n) in enumerate(zip(score_pos, score_neg)):
#         score_pos_list.append(score_pos[i][k_index_neg_list[i]]) # batch, 1
#         score_neg_list.append(score_neg[i][k_index_neg_list[i]]) # batch, 1

#     # score_pos_norm, score_neg_norm = F.normalize(score_pos, p=1, dim=-1), F.normalize(score_neg, p=1, dim=-1)
#     score_pos = torch.stack(score_pos_list, dim=0) # batch, 1
#     score_neg = torch.stack(score_neg_list, dim=0) # batch, 1

#     return (score_pos + F.relu(margin - score_neg)).sum()
    

# def info_nce_2(score_pos, score_neg, k_index_neg_list, temperature=1.0):
#     score_pos_list, score_neg_list = list(), list() # batch, 1
#     for i, (score_p, score_n) in enumerate(zip(score_pos, score_neg)):
#         score_pos_list.append(score_p[k_index_neg_list[i]]) # batch, 1
#         score_neg_list.append(score_n[k_index_neg_list[i]]) # batch, 1 

#     score_pos = torch.stack(score_pos_list, dim=0) # batch, 1
#     score_neg = torch.stack(score_neg_list, dim=0) # batch, 1
    
#     # score_pos_norm, score_neg_norm = F.normalize(score_pos, p=1, dim=-1), F.normalize(score_neg, p=1, dim=-1)
#     # logits = torch.cat([score_pos_norm, score_neg_norm], dim=-1) # batch, 2
#     logits = torch.cat([score_pos, score_neg], dim=-1) # batch, 2
#     # labels = torch.zeros(len(logits), dtype=torch.long, device=score_pos_norm.device)
#     labels = torch.zeros(len(logits), dtype=torch.long, device=score_pos.device)

#     return F.cross_entropy(logits /temperature, labels)

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

def info_nce_loss_ck_sg(feature_batch_list, kecrs_batch_list, kecrs_pos_batch_list, kecrs_neg_batch_list, temperature):
    # 正负样本间对比 + 片段内对比
    # knowledge_batch = F.normalize(torch.cat(kecrs_batch_list, dim=0)) # batch*seq_len, d_knowledge
    feature_batch = F.normalize(torch.cat(feature_batch_list, dim=0)) # batch*seq_len, d_knowledge
    knowledge_pos_batch = F.normalize(torch.cat(kecrs_pos_batch_list, dim=0)) # batch*seq_len, d_knowledge
    knowledge_neg_batch = F.normalize(torch.cat(kecrs_neg_batch_list, dim=0)) # batch*seq_len, d_knowledge


    logits_pos = (knowledge_pos_batch * feature_batch).sum(dim=-1, keepdim=True) # batch*seq_len, 1
    logits_neg = (knowledge_neg_batch * feature_batch).sum(dim=-1, keepdim=True) # batch*seq_len, 1
    logits_ck = torch.cat([logits_pos, logits_neg], dim=-1) # batch*seq_len, 2
    labels_ck = torch.zeros(len(logits_ck), dtype=torch.long, device=logits_ck.device) # batch*seq_len

    loss_ck = F.cross_entropy(logits_ck / temperature, labels_ck)

    loss_sg = 0.0
    loss_sg_num = 0

    for k, f in zip(kecrs_batch_list, feature_batch_list): # batch
        k_normed = F.normalize(k, dim=1) # seq_len, d_knowledge
        f_normed = F.normalize(f, dim=1) # seq_len, d_knowledge

        logits = k_normed @ f_normed.transpose(-2, -1) # seq_len, seq_len
        labels = torch.arange(len(k), device=k.device)

        loss_sg += F.cross_entropy(logits / temperature, labels)
        loss_sg_num += 1

    assert loss_sg_num == len(kecrs_batch_list)

    return loss_sg / loss_sg_num + loss_ck

def info_nce_loss_sg(feature_batch_list, kecrs_batch_list, kecrs_pos_batch_list, kecrs_neg_batch_list, temperature):
    # 正负样本间对比 + 片段内对比
    # knowledge_batch = F.normalize(torch.cat(kecrs_batch_list, dim=0)) # batch*seq_len, d_knowledge
    # feature_batch = F.normalize(torch.cat(feature_batch_list, dim=0)) # batch*seq_len, d_knowledge
    loss_sg = 0.0
    loss_sg_num = 0

    for k, f in zip(kecrs_batch_list, feature_batch_list): # batch
        k_normed = F.normalize(k, dim=1) # seq_len, d_knowledge
        f_normed = F.normalize(f, dim=1) # seq_len, d_knowledge

        logits = k_normed @ f_normed.transpose(-2, -1) # seq_len, seq_len
        labels = torch.arange(len(k), device=k.device)

        loss_sg += F.cross_entropy(logits / temperature, labels)
        loss_sg_num += 1

    assert loss_sg_num == len(kecrs_batch_list)

    return loss_sg / loss_sg_num


def info_nce_loss_ck_sp(feature_batch_list, kecrs_batch_list, kecrs_pos_batch_list, kecrs_neg_batch_list, temperature):
    # 正负样本间对比 + 片段内对比
    knowledge_batch = F.normalize(torch.cat(kecrs_batch_list, dim=0)) # batch*seq_len, d_knowledge
    feature_batch = F.normalize(torch.cat(feature_batch_list, dim=0)) # batch*seq_len, d_knowledge
    knowledge_pos_batch = F.normalize(torch.cat(kecrs_pos_batch_list, dim=0)) # batch*seq_len, d_knowledge
    knowledge_neg_batch = F.normalize(torch.cat(kecrs_neg_batch_list, dim=0)) # batch*seq_len, d_knowledge


    logits_pos = (knowledge_pos_batch * feature_batch).sum(dim=-1, keepdim=True) # batch*seq_len, 1
    logits_neg = (knowledge_neg_batch * feature_batch).sum(dim=-1, keepdim=True) # batch*seq_len, 1
    logits_ck = torch.cat([logits_pos, logits_neg], dim=-1) # batch*seq_len, 2
    labels_ck = torch.zeros(len(logits_ck), dtype=torch.long, device=logits_ck.device) # batch*seq_len

    loss_ck = F.cross_entropy(logits_ck / temperature, labels_ck)

    logits_sp = knowledge_batch @ feature_batch.transpose(-2, -1) # batch*seq_len, batch*seq_len

    labels_sp = torch.arange(len(logits_sp), device=logits_sp.device)

    loss_sp = F.cross_entropy(logits_sp / temperature, labels_sp)

    return loss_ck + loss_sp

def info_nce_loss_sp(feature_batch_list, kecrs_batch_list, kecrs_pos_batch_list, kecrs_neg_batch_list, temperature):
    # 正负样本间对比 + 片段内对比
    knowledge_batch = F.normalize(torch.cat(kecrs_batch_list, dim=0)) # batch*seq_len, d_knowledge
    feature_batch = F.normalize(torch.cat(feature_batch_list, dim=0)) # batch*seq_len, d_knowledge

    logits_sp = knowledge_batch @ feature_batch.transpose(-2, -1) # batch*seq_len, batch*seq_len

    labels_sp = torch.arange(len(logits_sp), device=logits_sp.device)

    loss_sp = F.cross_entropy(logits_sp / temperature, labels_sp)

    return loss_sp