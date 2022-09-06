'''
Description: 
Autor: Gary Liu
Date: 2021-09-07 19:02:07
LastEditors: Gary Liu
LastEditTime: 2021-09-11 23:06:32
'''
'''
Description: 
Autor: Gary Liu
Date: 2021-07-02 11:56:31
LastEditors: Gary Liu
LastEditTime: 2021-07-26 23:11:56
'''
import torch
from utils import read_json, convert_examples_to_ids, build_vocab, debug_print_dims
from base import BaseFeatureExtractor


class TextFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, config):
        super().__init__()
        print("Initializing TextFeatureExtractor...")
        self.feature_file = config["text"]["feature_file"]
        self.feature_dim = config["text"]["feature_dim"]
        self.features = read_json(self.feature_file)
        self.data = read_json(config["data_file"])
        self.missing_tensor = torch.zeros((self.feature_dim))

    def get_feature(self, clip, target_character):
        on_characters = self.data[clip]["on_character"]
        if target_character not in on_characters:
            on_characters.append(target_character)
        on_characters = sorted(on_characters)
        speakers = self.data[clip]["speakers"]
        seg_ori_ind = self.data[clip]["seg_ori_ind"]
        ret = []
        ret_valid = []
        for character in on_characters:
            for ii, speaker in enumerate(speakers):
                if character == speaker:
                    index = "{}+{}".format(clip, ii)
                    ret.append(torch.tensor(self.features[index]))
                    ret_valid.append(1)
                else:
                    ret.append(self.missing_tensor)
                    ret_valid.append(0)
        ret = torch.stack(ret, dim=0)
        ret_valid = torch.tensor(ret_valid, dtype=torch.int8)
        return ret, ret_valid

class TextFeatureExtractor_KE(BaseFeatureExtractor):
    def __init__(self, config):
        super().__init__()
        print("Initializing TextFeatureExtractor...")
        self.feature_file = config["text"]["feature_file"]
        self.feature_dim = config["text"]["feature_dim"]
        self.features = read_json(self.feature_file)
        self.data = read_json(config["data_file"])
        ########
        self.concepts = read_json(config["text"]['concepts_file'])
        self.concept2id, _ =  build_vocab(config, 'text')
        ########
        self.missing_tensor = torch.zeros((self.feature_dim))

    def get_feature(self, clip, target_character):
        on_characters = self.data[clip]["on_character"]
        if target_character not in on_characters:
            on_characters.append(target_character)
        on_characters = sorted(on_characters)
        speakers = self.data[clip]["speakers"]
        seg_ori_ind = self.data[clip]["seg_ori_ind"]
        ##############
        ret_concepts = list()
        ##############
        ret = []
        ret_valid = []
        
        for ii, speaker in enumerate(speakers):
            index = "{}+{}".format(clip, ii) # 妈的原作者实现有问题
            index_concept = "{}+{}".format(clip, seg_ori_ind[ii])
            for character in on_characters:
                if character == speaker:
                    ret.append(torch.tensor(self.features[index]))
                    ret_valid.append(1)
                else:
                    ret.append(self.missing_tensor)
                    ret_valid.append(0)

            ret_concepts.append(torch.LongTensor(convert_examples_to_ids(self.concepts[index_concept] if index_concept in self.concepts else [], self.concept2id)) ) # seqlen
            # ret_concepts.append(torch.LongTensor(convert_examples_to_ids([], self.concept2id)))

        ret = torch.stack(ret, dim=0) # seqlen * n_c, dim_feature_t
        ret_valid = torch.tensor(ret_valid, dtype=torch.int8) # seqlen * n_c
        # ret_concepts = convert_examples_to_ids(ret_concepts, self.concept2id) # seqlen
        return ret, ret_valid, ret_concepts

    @property
    def get_concept2id(self):
        return self.concept2id

class TextFeatureExtractor_KE_ALL(BaseFeatureExtractor):
    """文本特征提取类
    """    
    def __init__(self, config):
        super().__init__()
        print("Initializing Extracor", '[' + self.__class__.__name__ + ']', "...", end=' ')
        self.feature_file = config["text"]["feature_file"]
        self.feature_dim = config["text"]["feature_dim"]
        self.features = read_json(self.feature_file)
        self.data = read_json(config["data_file"])
        ########
        self.concepts = read_json(config["text"]['concepts_file'])
        # self.concept2id, _ =  build_vocab(config, 'text')
        ########
        self.missing_tensor = torch.zeros((self.feature_dim))
        print("Done!")

    def get_feature(self, clip, target_character):
        """文本特征获取

        Arguments:
            clip {Str} -- 欲提取的影片名称 \\
            target_character {int} -- 目标角色的index

        Returns:
            Tensor (n_seg * n_character, dim_feature_t) -- 文本特征，每个片段每个角色，如果没有，则置零 \\
            Tensor (n_seg * n_character) -- 文本特征掩膜，对应上述有或者没有 \\
            Tensor (n_seg, some_len) -- 每个片段对应的文本语义标签（非终止词），每个片段不定长
        """   
        on_characters = self.data[clip]["on_character"]
        if target_character not in on_characters:
            on_characters.append(target_character)
        on_characters = sorted(on_characters)
        speakers = self.data[clip]["speakers"]
        seg_ori_ind = self.data[clip]["seg_ori_ind"]
        ##############
        ret_concepts = list()
        ##############
        ret = []
        ret_valid = []
        
        for ii, speaker in enumerate(speakers):
            index = "{}+{}".format(clip, ii) # 妈的原作者实现有问题
            index_concept = "{}+{}".format(clip, seg_ori_ind[ii])
            for character in on_characters:
                if character == speaker:
                    ret.append(torch.tensor(self.features[index]))
                    ret_valid.append(1)
                else:
                    ret.append(self.missing_tensor)
                    ret_valid.append(0)

            ret_concepts.append(self.concepts[index_concept] if index_concept in self.concepts else []) # seqlen
            # ret_concepts.append(torch.LongTensor(convert_examples_to_ids(self.concepts[index_concept] if index_concept in self.concepts else [], self.concept2id)) ) # seqlen
            # ret_concepts.append(torch.LongTensor(convert_examples_to_ids([], self.concept2id)))

        ret = torch.stack(ret, dim=0) # seqlen * n_c, dim_feature_t
        ret_valid = torch.tensor(ret_valid, dtype=torch.int8) # seqlen * n_c
        # ret_concepts = convert_examples_to_ids(ret_concepts, self.concept2id) # seqlen
        return ret, ret_valid, ret_concepts