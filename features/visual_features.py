'''
Description: 
Autor: Gary Liu
Date: 2021-07-02 11:56:31
LastEditors: Gary Liu
LastEditTime: 2022-02-13 17:37:59
'''
import torch
import os
from base import BaseFeatureExtractor
from utils import read_json, convert_examples_to_ids, build_vocab, debug_print_dims


class VisualFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, config):
        super().__init__()
        print('Initializing VisualFeatureExtractor...')
        self.speakers = config['speakers']
        self.faces_features_dir = config['visual']['faces_feature_dir']
        self.faces_names_dir = config['visual']['faces_names_dir']
        self.obj_feature_dir = config['visual']['obj_feature_dir']
        self.env_features_dir = config['visual']['env_feature_dir']
        self.data = read_json(config['data_file'])
        self.feature_dim = config['visual']['dim_env'] + config['visual']['dim_obj'] + config['visual']['dim_face']
        self.missing_tensor = torch.zeros((self.feature_dim))

    def get_feature(self, clip, target_character):
        fps = 23.976
        ret = []
        ret_valid = []
        on_characters = self.data[clip]['on_character']
        if target_character not in on_characters:
            on_characters.append(target_character)
        on_characters = sorted(on_characters)
        seg_start = self.data[clip]['seg_start']
        seg_end = self.data[clip]['seg_end']
        overall_start = self.data[clip]['start']
        with open(os.path.join(self.faces_names_dir, clip+'.txt')) as fin:
            faces_image_names = fin.readline().strip().split('\t')
        
        threshold = 10

        if len(faces_image_names) > threshold:
            face_features = torch.load(os.path.join(self.faces_features_dir, clip+'.pt'), map_location='cpu')
            obj_features = torch.load(os.path.join(self.obj_feature_dir, clip+'.pt'), map_location='cpu')
            env_features = torch.load(os.path.join(self.env_features_dir, clip+'.pt'), map_location='cpu')
            for character in on_characters:
                for ii in range(len(seg_start)):
                    begin_sec, end_sec = seg_start[ii] - overall_start, seg_end[ii] - overall_start
                    begin_idx, end_idx = int(begin_sec * fps), int(end_sec * fps)
                    character_face_feature = []
                    for jj, image_name in enumerate(faces_image_names):
                        idx, person = tuple(image_name[:-4].split('_'))
                        if begin_idx <= int(idx) <= end_idx and person.lower() == self.speakers[character]:
                            character_face_feature.append(face_features[jj])
                    face_num = len(character_face_feature)
                    
                    if face_num > threshold:
                        ret_in = []
                        ret_in.append(torch.mean(torch.stack(character_face_feature), dim=0))
                        ret_in.append(torch.mean(obj_features[begin_idx:end_idx, :], dim=0))
                        ret_in.append(torch.mean(env_features[begin_idx:end_idx, :], dim=0))
                        # print(torch.cat(ret_in).shape)
                        ret.append(torch.cat(ret_in)) 
                        ret_valid.append(1)
                    else:
                        ret.append(self.missing_tensor) 
                        ret_valid.append(0)                    
            ret = torch.stack(ret, dim=0)
            ret_valid = torch.tensor(ret_valid, dtype=torch.int8)
            return ret, ret_valid
        else:
            return torch.zeros((len(on_characters) * len(seg_start), self.feature_dim)), torch.zeros(len(on_characters) * len(seg_start), dtype=torch.int8)

class VisualFeatureExtractor_KE(BaseFeatureExtractor):
    def __init__(self, config):
        super().__init__()
        print('Initializing VisualFeatureExtractor...')
        self.speakers = config['speakers']
        self.faces_features_dir = config['visual']['faces_feature_dir']
        self.faces_names_dir = config['visual']['faces_names_dir']
        self.obj_feature_dir = config['visual']['obj_feature_dir']
        self.env_features_dir = config['visual']['env_feature_dir']
        self.data = read_json(config['data_file'])
        ######################
        self.faces_concepts = read_json(config['visual']['faces_concepts_file'])
        self.obj_concepts = read_json(config['visual']['obj_concepts_file'])
        self.action_concepts = read_json(config['visual']['action_concepts_file'])
        self.concept2id, _ =  build_vocab(config, 'visual')
        ##################
        self.feature_dim = config['visual']['dim_env'] + config['visual']['dim_obj'] + config['visual']['dim_face']
        self.missing_tensor = torch.zeros((self.feature_dim))

    def get_feature(self, clip, target_character):
        fps = 23.976
        ret = []
        ret_valid = []
        on_characters = self.data[clip]['on_character']
        if target_character not in on_characters:
            on_characters.append(target_character)
        on_characters = sorted(on_characters)
        seg_start = self.data[clip]['seg_start']
        seg_end = self.data[clip]['seg_end']
        overall_start = self.data[clip]['start']
        with open(os.path.join(self.faces_names_dir, clip+'.txt')) as fin:
            faces_image_names = fin.readline().strip().split('\t')
        
        ######
        ret_concepts = list()
        ######

        threshold = 10

        if len(faces_image_names) > threshold:
            face_features = torch.load(os.path.join(self.faces_features_dir, clip+'.pt'), map_location='cpu')
            obj_features = torch.load(os.path.join(self.obj_feature_dir, clip+'.pt'), map_location='cpu')
            env_features = torch.load(os.path.join(self.env_features_dir, clip+'.pt'), map_location='cpu')

            for ii in range(len(seg_start)):
                index = "{}+{}".format(clip, ii)
                for character in on_characters:
                    begin_sec, end_sec = seg_start[ii] - overall_start, seg_end[ii] - overall_start
                    begin_idx, end_idx = int(begin_sec * fps), int(end_sec * fps)
                    character_face_feature = []
                    for jj, image_name in enumerate(faces_image_names):
                        idx, person = tuple(image_name[:-4].split('_'))
                        if begin_idx <= int(idx) <= end_idx and person.lower() == self.speakers[character]:
                            character_face_feature.append(face_features[jj])
                    face_num = len(character_face_feature)

                    if face_num > threshold:
                        ret_in = []
                        ret_in.append(torch.mean(torch.stack(character_face_feature), dim=0))
                        ret_in.append(torch.mean(obj_features[begin_idx:end_idx, :], dim=0))
                        ret_in.append(torch.mean(env_features[begin_idx:end_idx, :], dim=0))
                        # print(torch.cat(ret_in).shape)
                        ret.append(torch.cat(ret_in))
                        ret_valid.append(1)
                        
                    else:
                        ret.append(self.missing_tensor) 
                        ret_valid.append(0) 
                concepts = sorted(list(set(self.faces_concepts[index] + self.action_concepts[index] + self.obj_concepts[index]))) if index in self.faces_concepts else []
                # if "drawing" in concepts: print(index)
                ret_concepts.append(torch.LongTensor(convert_examples_to_ids(concepts, self.concept2id))) # seqlen      
                # ret_concepts.append(torch.LongTensor(convert_examples_to_ids([], self.concept2id)))
            
            ret = torch.stack(ret, dim=0) # seqlen * n_c, dim_feature_v
            ret_valid = torch.tensor(ret_valid, dtype=torch.int8) # seqlen * n_c
            # ret_concepts = convert_examples_to_ids(ret_concepts, self.concept2id) # seqlen

            return ret, ret_valid, ret_concepts
        else:

            return torch.zeros((len(on_characters) * len(seg_start), self.feature_dim)),\
                torch.zeros(len(on_characters) * len(seg_start), dtype=torch.int8),\
                [torch.LongTensor(0) for _ in range(len(seg_start))]

    @property
    def get_concept2id(self):
        return self.concept2id

class VisualFeatureExtractor_KE_ALL(BaseFeatureExtractor):
    """视频特征提取类
    """   
    def __init__(self, config):
        super().__init__()
        print("Initializing Extracor", '[' + self.__class__.__name__ + ']', "...", end=' ')
        self.speakers = config['speakers']
        self.faces_features_dir = config['visual']['faces_feature_dir']
        self.faces_names_dir = config['visual']['faces_names_dir']
        self.obj_feature_dir = config['visual']['obj_feature_dir']
        self.env_features_dir = config['visual']['env_feature_dir']
        self.data = read_json(config['data_file'])
        ######################
        self.faces_concepts = read_json(config['visual']['faces_concepts_file'])
        self.obj_concepts = read_json(config['visual']['obj_concepts_file'])
        self.action_concepts = read_json(config['visual']['action_concepts_file'])
        # self.concept2id, _ =  build_vocab(config, 'visual')
        ##################
        self.feature_dim = config['visual']['dim_env'] + config['visual']['dim_obj'] + config['visual']['dim_face']
        self.missing_tensor = torch.zeros((self.feature_dim))
        print("Done!")

    def get_feature(self, clip, target_character):
        """视频特征获取

        Arguments:
            clip {Str} -- 欲提取的影片名称 \\
            target_character {int} -- 目标角色的index

        Returns:
            Tensor (n_seg * n_character, dim_feature_v) -- 视频特征，每个片段每个角色，如果没有，则置零 \\
            Tensor (n_seg * n_character) -- 视频特征掩膜，对应上述有或者没有 \\
            Tensor (n_seg, some_len) -- 每个片段对应的视频语义标签（人脸表情，场景目标，人物动作），三个标签列表混合，每个片段不定长
        """  
        fps = 23.976
        ret = []
        ret_valid = []
        on_characters = self.data[clip]['on_character']
        if target_character not in on_characters:
            on_characters.append(target_character)
        on_characters = sorted(on_characters)
        seg_start = self.data[clip]['seg_start']
        seg_end = self.data[clip]['seg_end']
        overall_start = self.data[clip]['start']
        with open(os.path.join(self.faces_names_dir, clip+'.txt')) as fin:
            faces_image_names = fin.readline().strip().split('\t')
        
        ######
        ret_concepts = list()
        ######

        threshold = 10

        if len(faces_image_names) > threshold:
            face_features = torch.load(os.path.join(self.faces_features_dir, clip+'.pt'), map_location='cpu')
            obj_features = torch.load(os.path.join(self.obj_feature_dir, clip+'.pt'), map_location='cpu')
            env_features = torch.load(os.path.join(self.env_features_dir, clip+'.pt'), map_location='cpu')

            for ii in range(len(seg_start)):
                index = "{}+{}".format(clip, ii)
                for character in on_characters:
                    begin_sec, end_sec = seg_start[ii] - overall_start, seg_end[ii] - overall_start
                    begin_idx, end_idx = int(begin_sec * fps), int(end_sec * fps)
                    character_face_feature = []
                    for jj, image_name in enumerate(faces_image_names):
                        idx, person = tuple(image_name[:-4].split('_'))
                        if begin_idx <= int(idx) <= end_idx and person.lower() == self.speakers[character]:
                            character_face_feature.append(face_features[jj])
                    face_num = len(character_face_feature)

                    if face_num > threshold:
                        ret_in = []
                        ret_in.append(torch.mean(torch.stack(character_face_feature), dim=0))
                        ret_in.append(torch.mean(obj_features[begin_idx:end_idx, :], dim=0))
                        ret_in.append(torch.mean(env_features[begin_idx:end_idx, :], dim=0))
                        # print(torch.cat(ret_in).shape)
                        ret.append(torch.cat(ret_in))
                        ret_valid.append(1)
                        
                    else:
                        ret.append(self.missing_tensor) 
                        ret_valid.append(0)

                # 暂时按照去重处理，
                # concepts = sorted(list(set(self.faces_concepts[index] + self.action_concepts[index] + self.obj_concepts[index]))) if index in self.faces_concepts else []
                concepts = sorted(list(set(self.action_concepts[index] + self.obj_concepts[index]))) if index in self.faces_concepts else []
                ret_concepts.append(concepts) # seqlen      
                # ret_concepts.append(torch.LongTensor(convert_examples_to_ids([], self.concept2id)))
            
            ret = torch.stack(ret, dim=0) # seqlen * n_c, dim_feature_v
            ret_valid = torch.tensor(ret_valid, dtype=torch.int8) # seqlen * n_c
            # ret_concepts = convert_examples_to_ids(ret_concepts, self.concept2id) # seqlen

            return ret, ret_valid, ret_concepts
        else:

            return torch.zeros((len(on_characters) * len(seg_start), self.feature_dim)),\
                torch.zeros(len(on_characters) * len(seg_start), dtype=torch.int8),\
                [[] for _ in range(len(seg_start))]

