'''
Description: 
Autor: Gary Liu
Date: 2021-07-02 11:56:31
LastEditors: Gary Liu
LastEditTime: 2022-09-07 22:04:23
'''
import torch
from utils import read_json, convert_examples_to_ids, build_vocab
from base import BaseFeatureExtractor


class AudioFeatureExtractor_KE_ALL(BaseFeatureExtractor):
    """音频特征提取类
    """    
    def __init__(self, config):
        super().__init__()
        print("Initializing Extracor", '[' + self.__class__.__name__ + ']', "...", end=' ')
        self.feature_dim = config["audio"]["feature_dim"]
        self.feature_file = config["audio"]["feature_file"]
        self.data = read_json(config["data_file"])
        self.features = read_json(self.feature_file)
        ########
        # 音频概念的部分
        self.concepts = read_json(config["audio"]['concepts_file'])
        # self.concept2id, _ = build_vocab(config, 'audio')
        ########
        self.missing_tensor = torch.zeros((self.feature_dim))
        print("Done!")

    def get_feature(self, clip, target_character):
        """音频特征获取

        Arguments:
            clip {Str} -- 欲提取的影片名称 \\
            target_character {int} -- 目标角色的index

        Returns:
            Tensor (n_seg * n_character, dim_feature_a) -- 音频特征，每个片段每个角色，如果没有，则置零 \\
            Tensor (n_seg * n_character) -- 音频特征掩膜，对应上述有或者没有 \\
            Tensor (n_seg, some_len) -- 每个片段对应的音频语义标签（音频情绪），音频语义标签每个片段都是1个
        """        
        on_characters = self.data[clip]["on_character"]
        if target_character not in on_characters:
            on_characters.append(target_character)
        on_characters = sorted(on_characters)
        speakers = self.data[clip]["speakers"]
        seg_ori_ind = self.data[clip]["seg_ori_ind"]
        ret = []
        ret_valid = []
        ##############
        ret_concepts = list()
        ##############

        for ii, speaker in enumerate(speakers): # seq_len, n_c
            index = "{}+{}".format(clip, seg_ori_ind[ii])
            for character in on_characters:
                if character == speaker:
                    if index in self.features:
                        ret.append(torch.tensor(self.features[index]))
                        ret_valid.append(1)
                    else:
                        ret.append(self.missing_tensor)
                        ret_valid.append(0)
                else:
                    ret.append(self.missing_tensor)
                    ret_valid.append(0)

            # 注意这里的概念，全部改成原始的字符串，音频因为原始输入是一个字符串，这里处理一下
            ret_concepts.append([self.concepts[index]] if index in self.concepts else []) # seqlen，
            # ret_concepts.append(torch.LongTensor(convert_examples_to_ids([], self.concept2id)))

        ret = torch.stack(ret, dim=0) # seqlen * n_c, dim_feature_a
        ret_valid = torch.tensor(ret_valid, dtype=torch.int8) # seqlen * n_c
        # ret_concepts = convert_examples_to_ids(ret_concepts, self.concept2id) # seqlen
        return ret, ret_valid, ret_concepts 