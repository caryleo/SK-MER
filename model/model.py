import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from base import BaseModel
from model.attention import MultiHeadAttention_2
from model.attention import GraphAttention_CL
from model.attention import ModalityReinforcementUnit
from model.embedding import PositionalEncoding
class KEER_KE_CL_CK(BaseModel):
    '''
    并行化处理，并且使用了自己实现的基准模型
    '''
    def __init__(self, config):
        super().__init__()

    def initialize(self, config, device, vocab_size=None):
        n_classes = 9 if config['emo_type'] == 'primary' else 14
        D_e = config["model"]["args"]["D_e"] # dim_feature = 2 * D_e
        D_v = config["visual"]["dim_env"] + config["visual"]["dim_face"] + config["visual"]["dim_obj"]
        D_a = config["audio"]["feature_dim"]
        D_t = config["text"]["feature_dim"]
        D_p = config["personality"]["feature_dim"]
        D_knowledge = config["knowledge"]["embedding_dim"]

        # 知识图注意力，来自KET
        # TODO: 考虑调整内部片段之间表示的构成，增加一个MHSA？
        # self.g_att = GraphAttention_2(config, vocab_size)
        self.g_att = GraphAttention_CL(config, vocab_size)

        # 更加复杂的融合方式，来自PMR
        # self.attn_character_k2v = ModalityReinforcementUnit_3(D_knowledge, D_e * 4)
        # self.attn_character_k2a = ModalityReinforcementUnit_3(D_knowledge, D_e * 4)
        # self.attn_character_k2t = ModalityReinforcementUnit_3(D_knowledge, D_e * 4)
        self.mru_k2v = ModalityReinforcementUnit(D_knowledge, D_e * 4)
        self.mru_k2a = ModalityReinforcementUnit(D_knowledge, D_e * 4)
        self.mru_k2t = ModalityReinforcementUnit(D_knowledge, D_e * 4)
        # self.attn_character = ModalityReinforcementUnit_3(D_knowledge, D_e * 4)

        # 片段推理和最后的建模还是使用MHSA
        # TODO：这两个部分使用无参数SA使用试一下效果
        self.attn_segment = MultiHeadAttention_2(4, D_e * 4, D_e * 4, D_e * 4)
        self.attn_final = MultiHeadAttention_2(4, D_e * 4, D_e * 4, D_e * 4)

        self.enc_v = nn.Sequential(
            nn.Linear(D_v, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, D_e * 3),
            nn.ReLU(),
            nn.Linear(D_e * 3, 2 * D_e),
        )

        self.enc_a = nn.Sequential(
            nn.Linear(D_a, D_e * 8),
            nn.ReLU(),
            nn.Linear(D_e * 8, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, 2 * D_e),
        )
        self.enc_t = nn.Sequential(
            nn.Linear(D_t, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, 2 * D_e),
        )

        self.enc_p = nn.Sequential(
            nn.Linear(D_p, D_e * 4),
            nn.ReLU(),
            nn.Linear(D_e * 4, 2 * D_e),
        )

        self.out_layer = nn.Sequential(
            nn.Linear(4 * D_e, 2 * D_e), 
            nn.ReLU(), 
            nn.Linear(2 * D_e, n_classes)
        )

        unified_d = 14 * D_e
        # unified_d = 12 * D_e

        self.fusion_layer = nn.Linear(unified_d, 4 * D_e)

        self.cl_mlp = nn.Sequential(
            nn.Linear(12 * D_e, 4 * D_e),
            nn.ReLU(),
            nn.Linear(4 * D_e, D_knowledge),
        )

    def forward(self, U_v, U_a, U_t, U_p, M_v, M_a, M_t, C, concept_lengths, target_loc, seq_lengths, seg_len, n_c, contrastive_mask_list=None):
        '''
            U_v:    batch, seq_len, n_c, dim_visual_feature
            U_a:    batch, seq_len, n_c, dim_audio_feature
            U_t:    batch, seq_len, n_c, dim_text_feature
            U_q:    batch, seq_len, n_c, dim_personality_feature
            M_v:    batch, seq_len, n_c
            M_a:    batch, seq_len, n_c
            M_t:    batch, seq_len, n_c
            C:      batch, seg_len, concept_length
        '''
        # Encoders
        V_e, A_e, T_e, P_e = self.enc_v(U_v), self.enc_a(U_a), self.enc_t(U_t), self.enc_p(U_p) # (batch_size, padd_len, 2 * D_e)
        
        batch_size = M_v.shape[0]
        # 原始样本由n_c，seg_len构造，先转置一下，然后按照seg_len分割，
        V_list, A_list, T_list = [], [], []
        M_v_list, M_a_list, M_t_list = [], [], []
        target_moment_list, target_character_list = [], []

        feature_list = []
        for i in range(batch_size):
            target_moment, target_character = -1, -1
            for j in range(target_loc.shape[1]):
                if target_loc[i][j] == 1: # 因为原始的顺序从n_c, seqlen调成了seqlen, n_c，行对应序列，列对应角色
                    target_moment = int(j / n_c[i].cpu().numpy())
                    target_character = j % int(n_c[i].cpu().numpy())
                    break

            target_moment_list.append(target_moment)
            target_character_list.append(target_character)

            V_temp = torch.cat(
                [
                    V_e[i, : seq_lengths[i]].reshape(seg_len[i], n_c[i], -1),
                    P_e[i, : seq_lengths[i]].reshape(seg_len[i], n_c[i], -1)
                ], dim=2) # (seg_len, n_c, d_feature) * batch
            V_temp_tuple = torch.split(V_temp, 1, dim=0) # (1, n_c, d_feature) * batch x seg_len
            V_list.extend([V_temp.squeeze(0) for V_temp in V_temp_tuple]) # (n_c, d_feature) * batch x seg_len

            M_v_temp_tuple = torch.split(M_v[i, : seq_lengths[i]].reshape(seg_len[i], n_c[i], 1), 1, dim=0) # (1, n_c, 1) * batch_size x seg_len
            for M_v_temp in M_v_temp_tuple:
                M_v_list.append(M_v_temp.squeeze(0)) # (n_c, 1) * batch_size x seg_len

            A_temp = torch.cat(
                [
                    A_e[i, : seq_lengths[i]].reshape(seg_len[i], n_c[i], -1),
                    P_e[i, : seq_lengths[i]].reshape(seg_len[i], n_c[i], -1)
                ], dim=2) # (seg_len, n_c, d_feature) * batch
            A_temp_tuple = torch.split(A_temp, 1, dim=0) # (1, n_c, d_feature) * batch x seg_len
            A_list.extend([A_temp.squeeze(0) for A_temp in A_temp_tuple]) # (n_c, d_feature) * batch x seg_len

            M_a_temp_tuple = torch.split(M_a[i, : seq_lengths[i]].reshape(seg_len[i], n_c[i], 1), 1, dim=0) # (1, n_c, 1) * batch_size x seg_len
            for M_a_temp in M_a_temp_tuple:
                M_a_list.append(M_a_temp.squeeze(0)) # (n_c, 1) * batch_size x seg_len

            T_temp = torch.cat(
                [
                    T_e[i, : seq_lengths[i]].reshape(seg_len[i], n_c[i], -1),
                    P_e[i, : seq_lengths[i]].reshape(seg_len[i], n_c[i], -1)
                ], dim=2) # (seg_len, n_c, d_feature) * batch
            T_temp_tuple = torch.split(T_temp, 1, dim=0) # (1, n_c, d_feature) * batch x seg_len
            T_list.extend([T_temp.squeeze(0) for T_temp in T_temp_tuple]) # (n_c, d_feature) * batch x seg_len

            M_t_temp_tuple = torch.split(M_t[i, : seq_lengths[i]].reshape(seg_len[i], n_c[i], 1), 1, dim=0) # (1, n_c, 1) * batch_size x seg_len
            for M_t_temp in M_t_temp_tuple:
                M_t_list.append(M_t_temp.squeeze(0)) # (n_c, 1) * batch_size x seg_len

            feature_list.append(self.cl_mlp(torch.cat([V_temp.mean(dim=1), A_temp.mean(dim=1), T_temp.mean(dim=1)], dim=1))) # (seg_len, d_knowledge) * batch

        # 知识表示获取
        # concepts_representation_list = self.g_att(C, concept_lengths, seg_len) # [seg_len, dim_representation] * batch_size
        # concepts_representation = torch.cat(concepts_representation_list, dim=0).unsqueeze(1) # batch_size*seq_len, 1, dim_representation 

        ##############
        # TODO: 暂时换成原始的知识版本
        concepts_representation_list = list()
        local_nodes_list = list()
        for i in range(batch_size):
            if contrastive_mask_list is not None:
                concepts_representation_sample, local_nodes_sample = self.g_att(C[i][:seg_len[i]], concept_lengths[i][:seg_len[i]], contrastive_mask_list[i])
            else:
                concepts_representation_sample, local_nodes_sample = self.g_att(C[i][:seg_len[i]], concept_lengths[i][:seg_len[i]])
            concepts_representation_list.append(concepts_representation_sample) # [seg_len, dim_representation] * batch_size
            local_nodes_list.append(local_nodes_sample) # [seg_len, ped_len, dim_representation] * batch_size

        concepts_representation = torch.cat(concepts_representation_list, dim=0).unsqueeze(1) # batch_size*seq_len, 1, dim_representation 

        if contrastive_mask_list is None:
            # 角色间推理
            V_char = pad_sequence(V_list, batch_first=True) # (batch_size*seq_len, n_c(longest), 2 * D)
            V_char_pre = V_char.clone()
            M_v_char = pad_sequence(M_v_list, batch_first=True) # (batch_size*seq_len, n_c(longest), 1)
            M_v_char = M_v_char.transpose(-1, -2) # (batch_size*seq_len, 1, n_c(longest))
            

            A_char = pad_sequence(A_list, batch_first=True) # (batch_size*seq_len, n_c(longest), 2 * D)
            A_char_pre = A_char.clone()
            M_a_char = pad_sequence(M_a_list, batch_first=True) # (batch_size*seq_len, n_c(longest), 1)
            M_a_char = M_a_char.transpose(-1, -2) # (batch_size*seq_len, 1, n_c(longest))

            T_char = pad_sequence(T_list, batch_first=True) # (batch_size*seq_len, n_c(longest), 2 * D)
            T_char_pre = T_char.clone()
            M_t_char = pad_sequence(M_t_list, batch_first=True) # (batch_size*seq_len, n_c(longest), 1)
            M_t_char = M_t_char.transpose(-1, -2) # (batch_size*seq_len, 1, n_c(longest))

            #######################
            # TODO:替换成原始版本，看看是不是这里出问题
            assert concepts_representation.size(0) == V_char.size(0) == A_char.size(0) == T_char.size(0), print(concepts_representation.size(), V_char.size(), A_char.size(), T_char.size())
            V_char_processed_list, A_char_processed_list, T_char_processed_list = list(), list(), list()
            for i in range(V_char.size(0)):
                V_char_processed_list.append(self.mru_k2v(concepts_representation[i], V_char_pre[i], None, M_v_char[i])) # (n_c(longest), 2 * D) * batch_size x seq_len
                A_char_processed_list.append(self.mru_k2a(concepts_representation[i], A_char_pre[i], None, M_a_char[i])) # (n_c(longest), 2 * D) * batch_size x seq_len
                T_char_processed_list.append(self.mru_k2t(concepts_representation[i], T_char_pre[i], None, M_t_char[i])) # (n_c(longest), 2 * D) * batch_size x seq_len

            V_char_processed = torch.stack(V_char_processed_list, dim=0) # (batch_size*seq_len, n_c(longest), 2 * D)
            A_char_processed = torch.stack(A_char_processed_list, dim=0) # (batch_size*seq_len, n_c(longest), 2 * D)
            T_char_processed = torch.stack(T_char_processed_list, dim=0) # (batch_size*seq_len, n_c(longest), 2 * D)

            # V_char_processed = self.attn_character_k2v(concepts_representation, V_char_pre, None, M_v_char) # (batch_size*seq_len, n_c(longest), 2 * D)
            # A_char_processed = self.attn_character_k2a(concepts_representation, A_char_pre, None, M_a_char) # (batch_size*seq_len, n_c(longest), 2 * D)
            # T_char_processed = self.attn_character_k2t(concepts_representation, T_char_pre, None, M_t_char) # (batch_size*seq_len, n_c(longest), 2 * D)

            # V_char_processed = self.attn_character(concepts_representation, V_char_pre, None, M_v_char) # (batch_size*seq_len, n_c(longest), 2 * D)
            # A_char_processed = self.attn_character(concepts_representation, A_char_pre, None, M_a_char) # (batch_size*seq_len, n_c(longest), 2 * D)
            # T_char_processed = self.attn_character(concepts_representation, T_char_pre, None, M_t_char) # (batch_size*seq_len, n_c(longest), 2 * D)

            # residual
            V_char_post = V_char_processed + V_char # (batch_size*seq_len, n_c(longest), 2 * D)
            A_char_post = A_char_processed + A_char # (batch_size*seq_len, n_c(longest), 2 * D)
            T_char_post = T_char_processed + T_char # (batch_size*seq_len, n_c(longest), 2 * D)

            # 片段间推理前的处理，同样地，拆开后转置，然后按照n_c分割
            V_new_list, A_new_list, T_new_list = [], [], []
            M_v_new_list, M_a_new_list, M_t_new_list = [], [], []
            index_segment = 0
            for i in range(batch_size):
                V_new_temp_tuple = torch.split(V_char_post[index_segment: index_segment + seg_len[i], : n_c[i]].transpose(0, 1), 1, dim=0) # (1, seg_len, d_feature) * n_c
                V_new_list.extend([V_new_temp.squeeze(0) for V_new_temp in V_new_temp_tuple] ) # (seq_len, d_feature) * batch_size x n_c
                
                A_new_temp_tuple = torch.split(A_char_post[index_segment: index_segment + seg_len[i], : n_c[i]].transpose(0, 1), 1, dim=0) # (1, seg_len, d_feature) * n_c
                A_new_list.extend([A_new_temp.squeeze(0) for A_new_temp in A_new_temp_tuple] ) # (seq_len, d_feature) * batch_size x n_c

                T_new_temp_tuple = torch.split(T_char_post[index_segment: index_segment + seg_len[i], : n_c[i]].transpose(0, 1), 1, dim=0) # (1, seg_len, d_feature) * n_c
                T_new_list.extend([T_new_temp.squeeze(0) for T_new_temp in T_new_temp_tuple] ) # (seq_len, d_feature) * batch_size x n_c

                index_segment += seg_len[i]

                M_v_new_temp_tuple = torch.split(M_v[i, : seq_lengths[i]].reshape(seg_len[i], n_c[i], 1).transpose(0, 1), 1, dim=0) # (1, seg_len, 1) * n_c
                for M_v_new_temp in M_v_new_temp_tuple:
                    M_v_new_list.append(M_v_new_temp.squeeze(0)) # (seg_len, 1) * batch_size x n_c 

                M_a_new_temp_tuple = torch.split(M_a[i, : seq_lengths[i]].reshape(seg_len[i], n_c[i], 1).transpose(0, 1), 1, dim=0) # (1, seg_len, 1) * n_c
                for M_a_new_temp in M_a_new_temp_tuple:
                    M_a_new_list.append(M_a_new_temp.squeeze(0)) # (seg_len, 1) * batch_size x n_c

                M_t_new_temp_tuple = torch.split(M_t[i, : seq_lengths[i]].reshape(seg_len[i], n_c[i], 1).transpose(0, 1), 1, dim=0) # (1, seg_len, 1) * n_c
                for M_t_new_temp in M_t_new_temp_tuple:
                    M_t_new_list.append(M_t_new_temp.squeeze(0)) # (seg_len, 1) * batch_size x n_c

            # 片段间推理
            V_segment = pad_sequence(V_new_list, batch_first=True) # (batch_size*n_c, seqlen(longest), 2 * D) 
            V_segment_pre = V_segment.clone()
            M_v_segment = pad_sequence(M_v_new_list, batch_first=True) # (batch_size*n_c, seg_len(longest), 1)
            M_v_segment = M_v_segment.transpose(-1, -2) # (batch_size*n_c, 1, seg_len(longest))

            A_segment = pad_sequence(A_new_list, batch_first=True) # (batch_size*n_c, seqlen(longest), 2 * D)
            A_segment_pre = A_segment.clone()
            M_a_segment = pad_sequence(M_a_new_list, batch_first=True) # (batch_size*n_c, seg_len(longest), 1)
            M_a_segment = M_a_segment.transpose(-1, -2) # (batch_size*n_c, 1, seg_len(longest))

            T_segment = pad_sequence(T_new_list, batch_first=True) # (batch_size*n_c, seqlen(longest), 2 * D)
            T_segment_pre = T_segment.clone()
            M_t_segment = pad_sequence(M_t_new_list, batch_first=True) # (batch_size*n_c, seg_len(longest), 1)
            M_t_segment = M_t_segment.transpose(-1, -2) # (batch_size*n_c, 1, seg_len(longest))

            # print(V_segment_pre.size(), M_v_segment.size())
            V_segment_processed, V_segment_w = self.attn_segment(V_segment_pre, V_segment_pre, V_segment_pre, M_v_segment) # 
            A_segment_processed, A_segment_W = self.attn_segment(A_segment_pre, A_segment_pre, A_segment_pre, M_a_segment) # seg_len(longest), batch_size*n_c, 2 * D
            T_segment_processed, T_segment_W = self.attn_segment(T_segment_pre, T_segment_pre, T_segment_pre, M_t_segment) # seg_len(longest), batch_size*n_c, 2 * D

            # 再一次residual
            V_segment_post = V_segment_processed + V_segment # batch_size*n_c, seg_len(longest), 2 * D
            A_segment_post = A_segment_processed + A_segment # batch_size*n_c, seg_len(longest), 2 * D
            T_segment_post = T_segment_processed + T_segment # batch_size*n_c, seg_len(longest), 2 * D

            # 摘出target moment，并进行特征融合
            feature_final_list = []
            index_final = 0
            for i in range(batch_size):
                V_final_temp = V_segment_post[index_final: index_final + n_c[i], : seg_len[i]].transpose(0, 1)[target_moment_list[i]] # (n_c, d_feature)
                A_final_temp = A_segment_post[index_final: index_final + n_c[i], : seg_len[i]].transpose(0, 1)[target_moment_list[i]] # (n_c, d_feature)
                T_final_temp = T_segment_post[index_final: index_final + n_c[i], : seg_len[i]].transpose(0, 1)[target_moment_list[i]] # (n_c, d_feature)

                P_final_temp = P_e[i, : seq_lengths[i]].reshape(seg_len[i], n_c[i], -1)[target_moment_list[i]] # (n_c, d_feature)

                # feature_final_list.append(torch.cat([V_final_temp, A_final_temp, T_final_temp], dim=1)) # (n_c, 12 * D) * batch_size
                feature_final_list.append(torch.cat([V_final_temp, A_final_temp, T_final_temp, P_final_temp], dim=1)) # (n_c, 14 * D) * batch_size
                
                index_final += n_c[i]

            # 融合后的角色间图例
            feature_final = pad_sequence(feature_final_list, batch_first=True) # (batch_size, n_c(longest), 14 * D)
            feature_final = self.fusion_layer(feature_final) # (batch_size, n_c(longest), d_feature)
            feature_final_pre = feature_final.clone()

            feature_final_processed, feature_final_w = self.attn_final(feature_final_pre, feature_final_pre, feature_final_pre) # (batch_size, n_c(longest), 4*D)

            feature_final_post = feature_final_processed + feature_final # (batch_size, n_c(longest), 4*D)

            output_list = []
            for i in range(batch_size):
                output_list.append(feature_final_post[i, target_character_list[i]]) # (4*D) * batch_size

            output = torch.stack(output_list, dim=0) # (batch_size, 4*D)
            log_prob = F.log_softmax(self.out_layer(output), dim=1) # (batch_size, n_class)

            return log_prob, concepts_representation_list, feature_list, local_nodes_list
        
        else:
            return concepts_representation_list, feature_list
