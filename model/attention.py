import sys
from typing import Optional, Tuple
from torch import Tensor
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy

from utils import build_vocab

class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        # print(q.size(), k.size(), v.size(), )
        attn_w = torch.matmul(q / self.temperature, k.transpose(-2, -1))

        if mask is not None:
            attn_w = attn_w.masked_fill(mask == 0, -1e9)

        attn_w = self.dropout(F.softmax(attn_w, dim=-1))
        # if mask is not None:
        #     print(mask.size(), attn.size(), v.size())
        output = torch.matmul(attn_w, v)

        return output, attn_w


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_query, d_key, d_value, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_value
        assert self.d_model % h == 0 # d_model = d_val
        self.d_k = self.d_model // h
        self.h = h
        self.attention = ScaledDotProductAttention(self.d_k ** 0.5, attn_dropout=dropout)
        # self.linears = clones(nn.Linear(d_model, d_model), 4)
        linear_list = list()
        linear_list.append(nn.Linear(d_query, self.d_model))
        linear_list.append(nn.Linear(d_key, self.d_model))
        linear_list.append(nn.Linear(d_value, self.d_model))
        linear_list.append(nn.Linear(self.d_model, self.d_model))
        self.linears = nn.ModuleList(linear_list)
        # self.linears = nn.ModuleList([copy.deepcopy(nn.Linear(d_model, d_model)) for i in range(4)])
        self.attn = None
        # self.dropout = nn.Dropout(p=dropout)
        self.a = nn.Parameter(torch.Tensor([1]))
        self.b = nn.Parameter(torch.Tensor([1]))
        
    def forward(self, query, key, value, mask=None):
        """
            query: (batch_size, seq_len, d_model), seq_len can be either src_seq_len or tgt_seq_len
            key: (batch_size, seq_len, d_model), seq_len in key, value and mask are the same
            value: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, seq_len) or (batch_size, tgt_seq_len, tgt_seq_len) (legacy)
        """
        # if print_dims:
        #     print("{0}: query: type: {1}, shape: {2}".format(self.__class__.__name__, query.type(), query.shape))
        #     print("{0}: key: type: {1}, shape: {2}".format(self.__class__.__name__, key.type(), key.shape))
        #     print("{0}: value: type: {1}, shape: {2}".format(self.__class__.__name__, value.type(), value.shape))
        #     print("{0}: mask: type: {1}, shape: {2}".format(self.__class__.__name__, mask.type(), mask.shape))

        # some, 2 * dim_feature + dim_representation
        if mask is not None:
            mask = mask.unsqueeze(0)
        # nbatches = query.size(0)
        
        # 1) Do all linear projections in batch from d_model to (h, d_k)
        # print(self.__class__.__name__, sys._getframe().f_code.co_name, query.size(), key.size(), value.size(), mask.size() if mask is not None else None)
        # if mask is not None:
        #     print(query.size(), key.size(), value.size(), mask.size())
        query, key, value = [l(x).view(-1, self.h, self.d_k).transpose(0, 1) # num_head, some, d_k
            for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch
        # x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout) # (batch_size, h, seq_len, d_k)  # h, some, d_k
        # print(self.__class__.__name__, sys._getframe().f_code.co_name, query.size(), key.size(), value.size(), mask.size() if mask is not None else None)
        # if mask is not None:
        #     print(query.size(), key.size(), value.size(), mask.size())
        x, self.attn = self.attention(query, key, value, mask=mask) # (batch_size, h, seq_len, d_k)  # h, some, d_k
        # if print_dims:
        #     print("{0}: x (after attention): type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))

        # 3) Concatenate and apply a final linear
        # print(self.__class__.__name__, sys._getframe().f_code.co_name, x.size())
        x = x.transpose(0, 1).contiguous().view(-1, self.h * self.d_k) # some, 2 * dim_feature + dim_representation
        x = self.linears[-1](x) # (batch_size, seq_len, d_model)
        # if print_dims:
        #     print("{0}: x (after concatenation and linear): type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        # print(self.__class__.__name__, sys._getframe().f_code.co_name, x.size())
        return x, self.attn


class MultiHeadAttention_2(nn.Module):
    def __init__(self, h, d_query, d_key, d_value, dropout=0.1):
        super(MultiHeadAttention_2, self).__init__()
        self.d_model = d_value
        assert self.d_model % h == 0
        self.d_k = self.d_model // h
        self.h = h
        # self.linears = clones(nn.Linear(d_model, d_model), 4)
        linear_list = list()
        linear_list.append(nn.Linear(d_query, self.d_model))
        linear_list.append(nn.Linear(d_key, self.d_model))
        linear_list.append(nn.Linear(d_value, self.d_model))
        linear_list.append(nn.Linear(self.d_model, self.d_model))
        self.linears = nn.ModuleList(linear_list)

        self.attention = ScaledDotProductAttention(self.d_k ** 0.5, attn_dropout=dropout)
        self.attn_w = None
        self.dropout = nn.Dropout(p=dropout)
        self.a = nn.Parameter(torch.Tensor([1]))
        self.b = nn.Parameter(torch.Tensor([1]))
        
    def forward(self, query, key, value, mask=None):
        """
            query: (batch_size, seq_len, d_model), seq_len can be either src_seq_len or tgt_seq_len
            key: (batch_size, seq_len, d_model), seq_len in key, value and mask are the same
            value: (batch_size, seq_len, d_model)
            mask: (batch_size, 1, seq_len) or (batch_size, tgt_seq_len, tgt_seq_len) (legacy)
        """

        # if print_dims:
        #     print("{0}: query: type: {1}, shape: {2}".format(self.__class__.__name__, query.type(), query.shape))
        #     print("{0}: key: type: {1}, shape: {2}".format(self.__class__.__name__, key.type(), key.shape))
        #     print("{0}: value: type: {1}, shape: {2}".format(self.__class__.__name__, value.type(), value.shape))
        #     print("{0}: mask: type: {1}, shape: {2}".format(self.__class__.__name__, mask.type(), mask.shape))

        if mask is not None:
            mask = mask.unsqueeze(1) # batch_size, 1, 1, seq_len
        nbatches = query.size(0) # batch_size
        
        # 1) Do all linear projections in batch from d_model to (h, d_k)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2)
            for l, x in zip(self.linears, (query, key, value))] # batch_size, h, seq_len, d_k
        
        # 2) Apply attention on all the projected vectors in batch
        x, self.attn_w = self.attention(query, key, value, mask=mask) # (batch_size, h, seq_len, d_k)
        # if print_dims:
        #     print("{0}: x (after attention): type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))

        # 3) Concatenate and apply a final linear
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k) # batch_size, seq_len, d_feature
        x = self.linears[-1](x) # (batch_size, seq_len, d_model)
        # if print_dims:
        #     print("{0}: x (after concatenation and linear): type: {1}, shape: {2}".format(self.__class__.__name__, x.type(), x.shape))
        return x, self.attn_w


class GraphAttention(nn.Module):
    def __init__(self, config, vocab_size):
        super(GraphAttention, self).__init__()
        ##############
        # 因为没有随机性，所以这里采取了重复创建的方案
        # concept2id, _ = build_vocab(config, modal)
        #################
        # self.vocab_size = len(concept2id)
        self.vocab_size = vocab_size
        self.embed_size = config["knowledge"]["embedding_dim"] # embed_size 不用
        self.embed_kb = nn.Embedding(self.vocab_size, config["knowledge"]["embedding_dim"])
        self.GAW = config["model"]["args"]["graph_attention_weight"] if config["model"]["args"]["graph_attention_weight"] >= 0 else None
        self.concentration_factor = config["model"]["args"]["concentrator_factor"]
        ####################
        # self.attn_concept = ScaledDotProductAttention((2 * self.embed_size) ** 0.5, attn_dropout=0)
        self.embed = nn.Embedding(self.vocab_size, config["knowledge"]["embedding_dim"])
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.embed_size, self.embed_size),
        # )
        
        ####################

    def init_params(self, edge_matrix=None, affectiveness=None, embedding_concept=None, device=None):
        edge_matrix_range = (edge_matrix.max(dim=1)[0] - edge_matrix.min(dim=1)[0]).unsqueeze(1)
        edge_matrix = edge_matrix/(edge_matrix_range + (edge_matrix_range == 0).float()) # normalization
        self.edge_matrix = edge_matrix.to(device)

        self.affectiveness = affectiveness.to(device)
        
        if self.GAW is not None:
            self._lambda = nn.Parameter(torch.full((self.vocab_size,), self.GAW)).to(device)
        else:
            self._lambda = nn.Parameter(torch.full((self.vocab_size,), 0.5)).to(device)

        # self.embed.weight.data.copy_(torch.from_numpy(embedding_concept))

        self.embed.weight.data.copy_(torch.from_numpy(embedding_concept))
        # self.embed_kb.weight.data.copy_(torch.from_numpy(embedding_concept))

        self.empty_tensor = torch.zeros(self.embed_size).to(device)
        

    def get_context_representation(self, concepts_embed, concepts_length): 
        # 将每一个上下文表示，包括当前本身，编码成一个上下文表示
        # 对于一组概念，暂时使用取平均的方式
        # 输入尺寸为seqlen, padlen, embed_size和seqlen
        # 预期输出尺寸为dim_representation

        context_rep_final = list()
        for i in range(concepts_embed.size(0)):
            if concepts_length[i] > 0:
                context_rep_final.append(concepts_embed[i, :concepts_length[i]].mean(dim=0)) # embed_size
            else:
                context_rep_final.append(self.empty_tensor)

        if len(context_rep_final) > 0:
            # print(context_rep_final)
            # 这里后面考虑修改一下，使用别的上下文构建方案
            context_representation = torch.stack(context_rep_final, dim=0).mean(dim=0) # dim
        else:
            context_representation = self.empty_tensor

        return context_representation

    def forward(self, concepts_list, concepts_length):
        # 处理权重，范围一个等长度的seqlen, dim_representation的知识
        # 输入尺寸seqlen, padlen和seqlen
        # if modal == "text":
        #     print(concepts_list)
        #     print(concepts_length)
        concepts_embedded = self.embed(concepts_list) # seqlen, pad_len, embed_size
        # get context representation
        context_representation = self.get_context_representation(concepts_embedded, concepts_length) # embed_size
        
        # get concept embedding
        cosine_similarity = torch.abs(torch.cosine_similarity(context_representation.unsqueeze(0), self.embed_kb.weight, dim=1)) # (vocab_size)

        # seqlen, padlen, vocab_size
        # print(self.edge_matrix.device, concepts_list.device, cosine_similarity.device)
        rel = self.edge_matrix[concepts_list] * cosine_similarity # seqlen, padlen, vocab_size
        aff = (self.edge_matrix[concepts_list] > 0).float() * self.affectiveness # seqlen, padlen, vocab_size
        concepts_weights = self._lambda * rel + (1 - self._lambda) * aff # seqlen, padlen, vocab_size
        concepts_embedding = torch.matmul(torch.softmax(concepts_weights * self.concentration_factor, dim=2), self.embed_kb.weight) # seqlen, padlen, dim_representation
        
        # 因为每一个片段只有一个特征，因此直接恢复回去，暂时不区分是否有效
        concepts_embedding_final = list()
        for i in range(concepts_list.size(0)):
            # 整合方法为平均
            if concepts_length[i] > 0:
                concepts_embedding_final.append(concepts_embedding[i, :concepts_length[i]].mean(dim=0)) # dim_representation
            else:
                concepts_embedding_final.append(self.empty_tensor)

        concepts_embedding = torch.stack(concepts_embedding_final, dim=0) # seqlen, dim_representation
        return concepts_embedding


class GraphAttention_CL(nn.Module):
    def __init__(self, config, vocab_size):
        super(GraphAttention_CL, self).__init__()
        ##############
        # 因为没有随机性，所以这里采取了重复创建的方案
        # concept2id, _ = build_vocab(config, modal)
        #################
        # self.vocab_size = len(concept2id)
        self.vocab_size = vocab_size
        self.d_representation = config["knowledge"]["embedding_dim"] 
        self.embed_size = config["knowledge"]["embedding_dim"] # embed_size 不用
        self.embed_kb = nn.Embedding(self.vocab_size, config["knowledge"]["embedding_dim"])
        self.GAW = config["model"]["args"]["graph_attention_weight"] if config["model"]["args"]["graph_attention_weight"] >= 0 else None
        self.concentration_factor = config["model"]["args"]["concentrator_factor"]
        ####################
        # self.attn_concept = ScaledDotProductAttention((2 * self.embed_size) ** 0.5, attn_dropout=0)
        self.embed = nn.Embedding(self.vocab_size, config["knowledge"]["embedding_dim"])
        # self.mlp = nn.Sequential(
        #     nn.Linear(self.embed_size, self.embed_size),
        # )
        ####################

    def init_params(self, edge_matrix=None, affectiveness=None, embedding_concept=None, device=None):
        edge_matrix_range = (edge_matrix.max(dim=1)[0] - edge_matrix.min(dim=1)[0]).unsqueeze(1)
        edge_matrix = edge_matrix/(edge_matrix_range + (edge_matrix_range == 0).float()) # normalization
        self.edge_matrix = edge_matrix.to(device)

        self.affectiveness = affectiveness.to(device)
        
        if self.GAW is not None:
            self._lambda = nn.Parameter(torch.full((self.vocab_size,), self.GAW)).to(device)
        else:
            self._lambda = nn.Parameter(torch.full((self.vocab_size,), 0.5)).to(device)

        # self.embed.weight.data.copy_(torch.from_numpy(embedding_concept))
        self.embed.weight.data.copy_(torch.from_numpy(embedding_concept))
        # self.embed_kb.weight.data.copy_(torch.from_numpy(embedding_concept))

        self.empty_tensor = torch.zeros(self.d_representation).to(device) # dim_representation
        self.empty_tensor.requires_grad = True
        

    def get_context_representation(self, concepts_embed, concepts_length): 
        # 将每一个上下文表示，包括当前本身，编码成一个上下文表示
        # 对于一组概念，暂时使用取平均的方式
        # 输入尺寸为seqlen, padlen, embed_size和seqlen
        # 预期输出尺寸为dim_representation

        context_rep_final = list()
        for i in range(concepts_embed.size(0)):
            if concepts_length[i] > 0:
                context_rep_final.append(concepts_embed[i, :concepts_length[i]].mean(dim=0)) # embed_size
            else:
                context_rep_final.append(self.empty_tensor)

        if len(context_rep_final) > 0:
            # print(context_rep_final)
            # 这里后面考虑修改一下，使用别的上下文构建方案
            context_representation = torch.stack(context_rep_final, dim=0).mean(dim=0) # dim
        else:
            context_representation = self.empty_tensor

        return context_representation

    def forward(self, concepts_list, concepts_length, contrastive_mask=None):
        # 处理权重，范围一个等长度的seqlen, dim_representation的知识
        # contrastive_mask: (seq_len,vocab_size)
        # 输入尺寸seqlen, padlen和seqlen
        # if modal == "text":
        #     print(concepts_list)
        #     print(concepts_length)
        concepts_embedded = self.embed(concepts_list) # seqlen, pad_len, embed_size
        # get context representation
        context_representation = self.get_context_representation(concepts_embedded, concepts_length) # embed_size
        
        # get concept embedding
        cosine_similarity = torch.abs(torch.cosine_similarity(context_representation.unsqueeze(0), self.embed_kb.weight, dim=1)) # (vocab_size)

        # seqlen, padlen, vocab_size
        # print(self.edge_matrix.device, concepts_list.device, cosine_similarity.device)
        local_graph = self.edge_matrix[concepts_list] # seqlen, padlen, vocab_size
        # local_graph.requires_grad_()
        rel = local_graph * cosine_similarity # seqlen, padlen, vocab_size
        aff = (local_graph > 0).float() * self.affectiveness # seqlen, padlen, vocab_size
        concepts_weights = self._lambda * rel + (1 - self._lambda) * aff # seqlen, padlen, vocab_size

        local_nodes = self.embed_kb.weight.clone() # vocab_size, dim_representation
        local_nodes = local_nodes.view(1, self.vocab_size, self.embed_size).expand(concepts_embedded.size(0), -1, -1) # seq_len, vocab_size, dim_representation
        assert concepts_weights.size(0) == concepts_embedded.size(0) and concepts_weights.size(1) == concepts_embedded.size(1)
        
        if contrastive_mask is not None:
            # local_nodes = local_nodes * contrastive_mask.unsqueeze(2) # 
            concepts_weights = concepts_weights.masked_fill(contrastive_mask.unsqueeze(1) == 0, 1e-9)
            # local_nodes = local_nodes.masked_fill(contrastive_mask.unsqueeze(2) == 0, 1e-9)

        concepts_embedding = torch.matmul(torch.softmax(concepts_weights * self.concentration_factor, dim=2), local_nodes) # seqlen, padlen, dim_representation
        # concepts_embedding = self.mlp(concepts_embedding) # seqlen, padlen, dim_representation
        
        # 因为每一个片段只有一个特征，因此直接恢复回去，暂时不区分是否有效
        concepts_embedding_final = list()
        for i in range(concepts_list.size(0)):
            # 整合方法为平均
            if concepts_length[i] > 0:
                concepts_embedding_final.append(concepts_embedding[i, :concepts_length[i]].mean(dim=0)) # dim_representation
            else:
                concepts_embedding_final.append(self.empty_tensor)

        concepts_embedding = torch.stack(concepts_embedding_final, dim=0) # seqlen, dim_representation
        return concepts_embedding, local_nodes

class GraphAttention_2(nn.Module):
    def __init__(self, config, vocab_size):
        super(GraphAttention_2, self).__init__()
       
        self.vocab_size = vocab_size
        self.d_representation = config["knowledge"]["embedding_dim"] 
        self.embed_kb = nn.Embedding(self.vocab_size, self.d_representation) # 知识的内部表示统一为Glove的表示
        self.GAW = config["model"]["args"]["graph_attention_weight"] if config["model"]["args"]["graph_attention_weight"] >= 0 else None
        self.concentration_factor = config["model"]["args"]["concentrator_factor"]
        ####################
        # 由于我们的输入是多模态的标签，不是句子，因此将词嵌入的部分放在了模型里面
        self.embed = nn.Embedding(self.vocab_size, self.d_representation)
        # self.context_attn = MultiHeadAttention_2(4, self.d_representation, self.d_representation, self.d_representation)
        ####################

    def init_params(self, edge_matrix=None, affectiveness=None, embedding_concept=None, device=None):
        # conceptnet + senticnet
        edge_matrix_range = (edge_matrix.max(dim=1)[0] - edge_matrix.min(dim=1)[0]).unsqueeze(1)
        edge_matrix = edge_matrix/(edge_matrix_range + (edge_matrix_range == 0).float()) # 归一化
        self.edge_matrix = edge_matrix.to(device)
        
        # NRC_VAD
        # TODO: 希望通过调参的方式去掉这个部分
        self.affectiveness = affectiveness.to(device)
        
        if self.GAW is not None:
            self._lambda = nn.Parameter(torch.full((self.vocab_size,), self.GAW)).to(device)
        else:
            self._lambda = nn.Parameter(torch.full((self.vocab_size,), 0.5)).to(device)

        self.embed.weight.data.copy_(torch.from_numpy(embedding_concept))
        # self.embed_kb.weight.data.copy_(torch.from_numpy(embedding_concept))

        self.empty_tensor = torch.zeros(self.d_representation).to(device) # dim_representation
        self.empty_tensor.requires_grad = True

    def get_context_representation(self, concepts_embed, concepts_length, seg_len): 
        # concepts_embed: batch, seqlen, pad_len, embed_size
        # concepts_length: batch, seqlen, pad_len
        concepts_final_list = list()
        for i in range(concepts_embed.size(0)): # batch
            concepts_embedded_sample = concepts_embed[i, :seg_len[i]] # seq_len, pad_len, embed_size

            # 针对每一个样本，按照片段进行内部concept处理
            # TODO: 方案1：取平均
            concepts_embedded_list = list()
            for j in range(seg_len[i]): # seq_len
                if concepts_length[i, j] > 0:
                    concepts_embedded_list.append(concepts_embedded_sample[j, :concepts_length[i, j]].mean(dim=0)) # [embed_size] * seq_len
                else:
                    # 当前片段没有concept，使用全零向量替换
                    concepts_embedded_list.append(self.empty_tensor) # [embed_size] * seq_len

            # 对所有片段的表示，统合成一个表示
            # TODO：方案1：取平均
            concepts_embedded = torch.stack(concepts_embedded_list, dim=0) # seq_len. embed_size
            concepts_final_list.append(concepts_embedded.mean(dim=0)) # [embed_size] * batch    
            # concepts_embedded_processed, _ = self.context_attn(concepts_embedded.unsqueeze(0), concepts_embedded.unsqueeze(0), concepts_embedded.unsqueeze(0)) # seq_len, embed_size
            # concepts_embedded = concepts_embedded + concepts_embedded_processed.squeeze(0) # seq_len, embed_size
            # concepts_final_list.append(concepts_embedded.mean(dim=0)) # [embed_size] * batch
            # concepts_final_list.append(concepts_embedded) # [seq_len, embed_size] * batch

        context_representation = torch.stack(concepts_final_list, dim=0) # batch, embed_size

        return context_representation # batch, embed_size
        # return concepts_final_list # [seq_len, embed_size] * batch

    def forward(self, concepts, concepts_length, seg_len):
        '''
        concepts: batch, seqlen(longest), pad_len
        concepts_length: batch, seqlen(longest)
        seg_len: batch
        '''

        # 语义标签的预处理：嵌入
        concepts_embedded = self.embed(concepts) # batch, seq_len, pad_len, embed_size
        
        # 上下文知识构建：每个样本获得一个上下文信息
        context_representation = self.get_context_representation(concepts_embedded, concepts_length, seg_len) # batch, embed_size
        # context_list = self.get_context_representation(concepts_embedded, concepts_length, seg_len) # [seq_len, embed_size] * batch
        
        # final = list()
        # for index, context in enumerate(context_list): # batch
        #     # print(concepts[index].size(), context.size(), self.embed_kb.weight.size())
        #     cosine_similarity = torch.abs(torch.cosine_similarity(context.unsqueeze(1), self.embed_kb.weight.unsqueeze(0), dim=2)) # seq_len, vocab_size
        #     # print(cosine_similarity.size())
        #     rel = self.edge_matrix[concepts[index, :seg_len[index]]] * cosine_similarity.unsqueeze(1) # seq_len, pad_len, vocab_size

        #     aff = (self.edge_matrix[concepts[index, :seg_len[index]]] > 0).float() * self.affectiveness # seq_len, padlen, vocab_size

        #     concepts_weights = self._lambda * rel + (1 - self._lambda) * aff # seq_len, padlen, vocab_size
        #     concepts_embedding = torch.matmul(torch.softmax(concepts_weights * self.concentration_factor, dim=2), self.embed_kb.weight) # seq_len, padlen, dim_representation
            
        #     concepts_embedding_final_list = list()
        #     for j in range(seg_len[index]): # seq_len
        #         if concepts_length[index, j] > 0:
        #             concepts_embedding_final_list.append(concepts_embedding[j, :concepts_length[index, j]].mean(0)) # [embed_size] * seq_len
        #         else:
        #             # 当前片段没有concept，使用全零向量替换
        #             concepts_embedding_final_list.append(self.empty_tensor) # [embed_size] * seq_len
                
        #     concepts_embedded_final = torch.stack(concepts_embedding_final_list, dim=0) # seq_len, embed_size

        #     final.append(concepts_embedded_final) # [seq_len, embed_size] * batch

        # return final # [seq_len, embed_size] * batch

        # 面向所有词汇的相似度
        # print(context_representation.size(), self.embed_kb.weight.size())
        cosine_similarity = torch.abs(torch.cosine_similarity(context_representation.unsqueeze(1), self.embed_kb.weight.unsqueeze(0), dim=2)) # batch, vocab_size

        # KET的attention实现
        rel = self.edge_matrix[concepts] * cosine_similarity.view(concepts.size(0), 1, 1, -1) # batch, seq_len, pad_len, vocab_size
        aff = (self.edge_matrix[concepts] > 0).float() * self.affectiveness # batch, seqlen, padlen, vocab_size
        concepts_weights = self._lambda * rel + (1 - self._lambda) * aff # batch, seqlen, padlen, vocab_size
        concepts_embedding = torch.matmul(torch.softmax(concepts_weights * self.concentration_factor, dim=3), self.embed_kb.weight) # batch, seqlen, padlen, dim_representation
        
        # 特征整合：每一个片段保留一个特征
        # TODO: 方案1：取平均
        concepts_embedding_final_list = list()
        for i in range(concepts_embedding.size(0)): # batch
            concepts_embedding_sample = concepts_embedding[i, :seg_len[i]] # seq_len, pad_len, embed_size
            
            concepts_embedding_sample_list = list()
            for j in range(seg_len[i]): # seq_len
                if concepts_length[i, j] > 0:
                    concepts_embedding_sample_list.append(concepts_embedding_sample[j, :concepts_length[i, j]].mean(dim=0)) # [dim_representation] * seq_len
                else:
                    # 当前片段没有concept，使用全零向量替换
                    concepts_embedding_sample_list.append(self.empty_tensor)

            concepts_embedding_final_list.append(torch.stack(concepts_embedding_sample_list, dim=0)) # [seqlen, dim_representation] * batch
        
        return concepts_embedding_final_list # [seqlen, dim_representation] * batch

class PositionalWiseFeedForward(nn.Module):

    def __init__(self, d_model=512, d_ff=2048, dropout=0.1):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(d_model, d_ff, 1)
        self.w2 = nn.Conv1d(d_ff, d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, x_residual):
        x = x.unsqueeze(0) # 为了兼容
        output = x.transpose(1, 2)
        output = self.w2(torch.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))
        output = output.squeeze(0)
        # add residual and norm layer
        if x_residual is not None:
            output = self.layer_norm(x_residual + output)
        return output


class ModalityReinforcementUnit(nn.Module):
    def __init__(self, d_source, d_target, h=4, dropout=0.1):
        super().__init__()

        self.d_model = d_target
        self.source_ln = nn.LayerNorm(self.d_model)
        self.target_ln = nn.LayerNorm(self.d_model)
        # self.cross_att = MultiHeadAttention(h, d_target, d_source, d_source, dropout)
        # self.self_att = MultiHeadAttention(h, d_target, d_target, d_target, dropout)
        self.cross_att = MultiHeadAttention(h, self.d_model, self.d_model, self.d_model, dropout)
        self.self_att = MultiHeadAttention(h, self.d_model, self.d_model, self.d_model, dropout)
        
        self.source_linear = nn.Linear(d_source, self.d_model)
        self.target_linear = nn.Linear(d_target, self.d_model)

        self.source_att_linear = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout()
        )
        self.target_att_linear = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout()
        )

        # self.fuse_ln = nn.LayerNorm(d_model)
        # self.pff = PositionalWiseFeedForward(self.d_model, self.d_model)

    def forward(self, source, target, mask_source=None, mask_target=None, all_attention=False):
        # source: n_source, dim_source
        # target: n_target, dim_target

        # print(self.__class__.__name__, sys._getframe().f_code.co_name, source.size(), target.size())
        output_source = self.source_ln(self.source_linear(source)) # n_source, dim_model 
        output_target = self.target_ln(self.target_linear(target)) # n_target, dim_model
        # output_source = self.source_ln(source) # n_source, dim_model 
        # output_target = self.target_ln(target) # n_target, dim_model
        # output_source = self.source_linear(source) # n_source, dim_model 
        # output_target = self.target_linear(target) # n_target, dim_model
        # if mask_source is not None:
        #     print(output_source.size(), output_target.size())
        output_source_att, _ = self.cross_att(output_target, output_source, output_source, mask_source) # n_target, d_model
        output_target_att, _ = self.self_att(output_target, output_target, output_target, mask_target) # n_target, d_model

        output_source_att_add = output_source_att + output_source # n_target, d_model (n_source = 1)
        output_target_att_add = output_target_att + output_target # n_target, d_model

        # print(self.__class__.__name__, output_source_att.size(), output_target_att.size())
        output_source_lin = self.source_att_linear(output_source_att_add)
        output_target_lin = self.target_att_linear(output_target_att_add)
        # print(self.__class__.__name__, output_source_lin.size(), output_target_lin.size())
        output_fuse_coeff = torch.sigmoid(output_source_lin + output_target_lin) # n_target, d_model

        # print(self.__class__.__name__, output_fuse_coeff.size(), output_target_att.size())
        output = output_fuse_coeff * output_target_att_add + (1 - output_fuse_coeff) * output_source_att_add # n_target, d_model
        # output = self.pff(output, output) # n_target, d_model
        # output = output_source_lin + output_target_lin
        # print(self.__class__.__name__, sys._getframe().f_code.co_name, output.size())
        
        if all_attention:
            # 要用一下
            return output, output_source_att_add, output_target_att_add
        else:
            return output

class ModalityReinforcementUnit_3(nn.Module):
    def __init__(self, d_source, d_target, h=4, dropout=0.1):
        super().__init__()

        self.d_model = d_target
        self.source_ln = nn.LayerNorm(self.d_model)
        self.target_ln = nn.LayerNorm(self.d_model)

        self.cross_att = MultiHeadAttention_2(h, self.d_model, self.d_model, self.d_model, dropout)
        self.self_att = MultiHeadAttention_2(h, self.d_model, self.d_model, self.d_model, dropout)
        
        self.source_linear = nn.Linear(d_source, self.d_model)
        self.target_linear = nn.Linear(d_target, self.d_model)

        self.source_att_linear = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout()
        )
        self.target_att_linear = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout()
        )

        # self.fuse_ln = nn.LayerNorm(d_model)
        # self.pff = PositionalWiseFeedForward(self.d_model, self.d_model)

    def forward(self, source, target, mask_source=None, mask_target=None):
        # source: batch, n_source, dim_source
        # target: batch, n_target, dim_target
        # 二者的batch*seq_len是相同的，区别的内部的序列长度，源串知识，1长度；目标串用户之间，n_c长度

        # print(self.__class__.__name__, sys._getframe().f_code.co_name, source.size(), target.size())
        # print(self.__class__.__name__, sys._getframe().f_code.co_name, source.size(), target.size())
        output_source = self.source_ln(self.source_linear(source)) # batch, n_source, d_model 
        output_target = self.target_ln(self.target_linear(target)) # batch, n_target, d_model

        # print(output_source.size(), output_target.size())

        output_source_att, _ = self.cross_att(output_target, output_source, output_source, mask_source) # batch, n_target, d_model
        output_target_att, _ = self.self_att(output_target, output_target, output_target, mask_target) # batch, n_target, d_model

        # print(output_source_att.size(), output_target_att.size())

        output_source_att_add = output_source_att + output_source # batch, n_target, d_model (n_source = 1)
        output_target_att_add = output_target_att + output_target # batch, n_target, d_model

        # print(output_source_att_add.size(), output_target_att.size())

        output_source_lin = self.source_att_linear(output_source_att_add) # batch, n_target, d_model
        output_target_lin = self.target_att_linear(output_target_att_add) # batch, n_target, d_model
        # output_fuse_coeff = torch.sigmoid(output_source_lin + output_target_lin) # batch, n_target, d_model

        # print(output_fuse_coeff.size(), output_source_lin.size(), output_target_lin.size())

        # output = output_fuse_coeff * output_target_att_add + (1 - output_fuse_coeff) * output_source_att_add # batch, n_target, d_model
        
        output = output_source_lin + output_target_lin
        # print(output.size())

        return output


class ModalityReinforcementUnit_2(nn.Module):
    def __init__(self, d_source, d_target, h=4, dropout=0.1):
        super().__init__()

        self.d_model = d_target
        self.source_ln = nn.LayerNorm(self.d_model)
        self.target_ln = nn.LayerNorm(self.d_model)
        # self.cross_att = MultiHeadAttention(h, d_target, d_source, d_source, dropout)
        # self.self_att = MultiHeadAttention(h, d_target, d_target, d_target, dropout)
        self.cross_att = MultiHeadAttention(h, self.d_model, self.d_model, self.d_model, dropout)
        self.self_att = MultiHeadAttention(h, self.d_model, self.d_model, self.d_model, dropout)
        
        self.source_linear = nn.Linear(d_source, self.d_model)
        self.target_linear = nn.Linear(d_target, self.d_model)

        self.source_att_linear = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout()
        )
        self.target_att_linear = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Dropout()
        )

        # self.fuse_ln = nn.LayerNorm(d_model)
        # self.pff = PositionalWiseFeedForward(self.d_model, self.d_model)

    def forward(self, source, target, mask_source=None, mask_target=None):
        # source: n_source, dim_source
        # target: n_target, dim_target

        # print(self.__class__.__name__, sys._getframe().f_code.co_name, source.size(), target.size())
        # output_source = self.source_ln(self.source_linear(source)) # n_source, dim_model 
        # output_target = self.target_ln(self.target_linear(target)) # n_target, dim_model
        # output_source = self.source_ln(source) # n_source, dim_model 
        # output_target = self.target_ln(target) # n_target, dim_model
        output_source = self.source_linear(source) # n_source, dim_model 
        output_target = self.target_linear(target) # n_target, dim_model

        output_source_att, _ = self.cross_att(output_target, output_source, output_source, mask_source) # n_target, d_model
        output_target_att, _ = self.self_att(output_target, output_target, output_target, mask_target) # n_target, d_model

        # print(self.__class__.__name__, output_source_att.size(), output_target_att.size())
        output_source_lin = self.source_att_linear(output_source_att)
        output_target_lin = self.target_att_linear(output_target_att)
        # print(self.__class__.__name__, output_source_lin.size(), output_target_lin.size())
        output_fuse_coeff = torch.sigmoid(output_source_lin + output_target_lin) # n_target, d_model

        # print(self.__class__.__name__, output_fuse_coeff.size(), output_target_att.size())
        output = output_fuse_coeff * output_target_att + (1 - output_fuse_coeff) * output_source_att # n_target, d_model
        # output = self.pff(output, output) # n_target, d_model
        # print(self.__class__.__name__, sys._getframe().f_code.co_name, output.size())

        return output


class MessageUpdateModule(nn.Module):
    def __init__(self, d_model, d_knwoledge, h=4, dropout=0.1):
        super().__init__()

        self.mru_v = ModalityReinforcementUnit(d_model, d_knwoledge, h, dropout)
        self.mru_a = ModalityReinforcementUnit(d_model, d_knwoledge, h, dropout)
        self.mru_t = ModalityReinforcementUnit(d_model, d_knwoledge, h, dropout)

        self.attn_v = nn.Linear(d_knwoledge, d_knwoledge)
        self.attn_a = nn.Linear(d_knwoledge, d_knwoledge)
        self.attn_t = nn.Linear(d_knwoledge, d_knwoledge)
        self.attn_fuse = nn.Linear(d_knwoledge, 1)

        self.pff = PositionalWiseFeedForward(d_knwoledge, d_knwoledge * 4)


    def forward(self, visual, audio, text, mask_v, mask_a, mask_t, knowledge_pre, knowledge):
        # visual: n_c, dim_feature
        # audio: n_c, dim_feature
        # text: n_c, dim_feature
        # knowledge_pre: 1, dim_representation
        # knowledge: 1, dim_representation

        # print(self.__class__.__name__, sys._getframe().f_code.co_name, visual.size(), knowledge_pre.size())
        output_v = self.mru_v(visual, knowledge_pre, mask_v, None) # 1, d_knowledge
        output_a = self.mru_a(audio, knowledge_pre, mask_a, None) # 1, d_knowledge
        output_t = self.mru_t(text, knowledge_pre, mask_t, None) # 1, d_knowledge
        # print(self.__class__.__name__, sys._getframe().f_code.co_name, output_v.size())

        output = torch.cat([output_v, output_a, output_t], dim = 0) # 3, d_knowledge

        output_weight_v = self.attn_fuse(torch.tanh(self.attn_v(output_v))) # 1, 1
        output_weight_a = self.attn_fuse(torch.tanh(self.attn_a(output_a))) # 1, 1
        output_weight_t = self.attn_fuse(torch.tanh(self.attn_t(output_t))) # 1, 1

        nn.Dropout()
        output_weight = torch.softmax(torch.cat([output_weight_v, output_weight_a, output_weight_t], dim=1), dim=1) # 1, 3

        # print(self.__class__.__name__, sys._getframe().f_code.co_name, output_weight.size(), output.size())
        output = torch.matmul(output_weight, output) # 1, d_knowledge

        output = self.pff(output, knowledge)

        return output

        