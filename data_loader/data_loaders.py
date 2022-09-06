import torch.utils.data as data
from torch.nn.utils.rnn import pad_sequence
import torch
from base import BaseDataLoader
from utils import read_json, find_model_using_name, debug_print_dims, build_vocab, convert_cocepts_to_ids
import numpy as np
import pandas as pd
from tqdm import tqdm
from features import AudioFeatureExtractor, TextFeatureExtractor, VisualFeatureExtractor, PersonalityFeatureExtractor
from features import AudioFeatureExtractor_KE, TextFeatureExtractor_KE, VisualFeatureExtractor_KE
from features import AudioFeatureExtractor_KE_ALL, TextFeatureExtractor_KE_ALL, VisualFeatureExtractor_KE_ALL

EMOTIONS = ["neutral","joy","anger","disgust","sadness","surprise","fear","anticipation","trust","serenity","interest","annoyance","boredom","distraction"]

class MEmoRDataset_KE_CL(data.Dataset):
    """数据集实例类，所有模态标签混合版本
    """    
    def __init__(self, config):
        super().__init__()
        self.config = config
        annos = read_json(config['anno_file'])[config['emo_type']]

        ##################################################
        # 作者将这部分代码注释掉了，就是用是否只用训练数据集进行训练
        # 在train_test模式下，所有的数据都应该预先处理，然后按照给定的train和test的id文件划分训练集和验证集
        if not config["val_file"]:
            print("!!! Caution! Loading Samples from {}".format(config['id_file']))
            ids = []
            tmp_annos = []
            with open(config['id_file']) as fin:
                for line in fin.readlines():
                    ids.append(int(line.strip()))
            
            for jj, anno in enumerate(annos):
                if jj in ids:
                    tmp_annos.append(anno)
            annos = tmp_annos
        ##################################################
            
        emo_num = 9 if config['emo_type'] == 'primary' else 14
        self.emotion_classes = EMOTIONS[:emo_num]
        
        data = read_json(config['data_file'])
        self.visual_features, self.audio_features, self.text_features = [], [], []
        self.visual_valids, self.audio_valids, self.text_valids = [], [], []
        ################################
        # 所有模态的标签，按照片段归到一块
        self.concept2id, self.id2concept, self.useful_words, self.useful_ids = build_vocab(config, "all", contrast_flag=True)
        self.concepts = list()
        self.concepts_length = list()
        ################################

        self.labels = []
        self.charcaters_seq = []
        self.time_seq = []
        self.target_loc = []
        self.seg_len = [] 
        self.n_character = []
        vfe = VisualFeatureExtractor_KE_ALL(config)
        afe = AudioFeatureExtractor_KE_ALL(config)
        tfe = TextFeatureExtractor_KE_ALL(config)
        pfe = PersonalityFeatureExtractor(config)
        self.personality_list = pfe.get_features() # n_c
        self.personality_features = []
        
        print('Processing Samples...')
        for jj, anno in enumerate(tqdm(annos)):
            # if jj >= 300: break # 测试用，加载全部数据有点浪费时间
            clip = anno['clip']
            target_character = anno['character']
            target_moment = anno['moment']
            on_characters = data[clip]['on_character']
            if target_character not in on_characters:
                on_characters.append(target_character)
            on_characters = sorted(on_characters)
            
            charcaters_seq, time_seq, target_loc, personality_seq = [], [], [], []
            

            for ii in range(len(data[clip]['seg_start'])):
                for character in on_characters:
                    charcaters_seq.append([0 if character != i else 1 for i in range(len(config['speakers']))])
                    time_seq.append(ii)
                    personality_seq.append(self.personality_list[character])
                    if character == target_character and data[clip]['seg_start'][ii] <= target_moment < data[clip]['seg_end'][ii]:
                        target_loc.append(1)
                    else:
                        target_loc.append(0)
            
            ####################################################
            # xf: seqlen * n_c, dim_features_x XX模态特征
            # x_valid: seqlen * n_c XX模态特征掩膜，表示有效性
            # xc：seqlen, somelen XX模态语义标签，每个片段一个list
            v_feature, v_valid, v_concepts = vfe.get_feature(anno['clip'], target_character) # seqlen * n_c, dim_features_v
            a_feature, a_valid, a_concepts = afe.get_feature(anno['clip'], target_character) # seqlen * n_c, dim_features_a
            t_feature, t_valid, t_concepts = tfe.get_feature(anno['clip'], target_character) # seqlen * n_c, dim_features_t
            ####################################################
            
            self.n_character.append(len(on_characters))
            self.seg_len.append(len(data[clip]['seg_start']))
    
            self.personality_features.append(torch.stack(personality_seq)) # num_anno, seqlen * n_c, dim_features_p
            self.charcaters_seq.append(torch.tensor(charcaters_seq)) # num_anno, seqlen * n_c, some
            self.time_seq.append(torch.tensor(time_seq)) # num_anno, seqlen * n_c, some
            self.target_loc.append(torch.tensor(target_loc, dtype=torch.int8)) # num_anno, seqlen * n_c
            self.visual_features.append(v_feature) # num_anno, seqlen * n_c, dim_features_v
            self.audio_features.append(a_feature) # num_anno, seqlen * n_c, dim_features_a
            self.text_features.append(t_feature) # num_anno, seqlen * n_c, dim_features_t
            self.visual_valids.append(v_valid) # num_anno, seqlen * n_c
            self.audio_valids.append(a_valid) # num_anno, seqlen * n_c
            self.text_valids.append(t_valid) # num_anno, seqlen * n_c

            #######################################################
            # 对应的保存，按照样本对应
            # 每个片段进行合并，然后组合保存起来，合并顺序：V+A+T
            concepts_all = list()
            for v_c, a_c, t_c in zip(v_concepts, a_concepts, t_concepts):
                # concepts_all.append(v_c + a_c + t_c)
                concepts_all.append(v_c + t_c)
            
            ids, ids_lengths = convert_cocepts_to_ids(concepts_all, self.concept2id)

            # if torch.any(ids_lengths < 0):
            #     print(ids_lengths, ids)

            self.concepts.append(ids)
            self.concepts_length.append(ids_lengths)
            #######################################################

            self.labels.append(self.emotion_classes.index(anno['emotion']))


    def __getitem__(self, index):
        """按照index获取样本的覆盖实现

        Arguments:
            index {Int} -- 保存样本的index

        Returns:
            Tensor (1) -- 当前样本的标签，数值对应情绪索引，参考文件开头的EMOTIONS数组 \\
            Tensor (n_seg * n_character, dim_feature_v) -- 当前样本所在影片的视觉特征 \\
            Tensor (n_seg * n_character, dim_feature_a) -- 当前样本所在影片的音频特征 \\
            Tensor (n_seg * n_character, dim_feature_t) -- 当前样本所在影片的文本特征 \\
            Tensor (n_seg * n_character, dim_feature_p) -- 当前样本所在影片的人格特征 \\
            Tensor (n_seg * n_character) -- 当前样本所在影片的视觉特征掩膜 \\
            Tensor (n_seg * n_character) -- 当前样本所在影片的音频特征掩膜 \\
            Tensor (n_seg * n_character) -- 当前样本所在影片的文本特征掩膜 \\
            Tensor (n_seg, some_len) -- 当前样本所在影片各个片段的语义标签（全部模态），已编码 \\
            Tensor (n_seg,) -- 当前样本所在影片各个片段的语义标签（全部模态）的长度 \\
            Tensor (n_seg * n_character) -- 当前样本所在影片的考察位置掩膜 \\
            Tensor (n_seg * n_character) -- 当前样本实际长度掩膜 \\
            Tensor (n_seg) -- 当前样本所在影片的所有分段的起始时间戳 \\
            Tensor (n_character) -- 当前样本所在影片的角色出场情况，顺序决定了后续计算的时候的列 \\
        """        
        return torch.tensor([self.labels[index]]), \
            self.visual_features[index], \
            self.audio_features[index], \
            self.text_features[index], \
            self.personality_features[index], \
            self.visual_valids[index], \
            self.audio_valids[index], \
            self.text_valids[index], \
            self.concepts[index], \
            self.concepts_length[index], \
            self.target_loc[index], \
            torch.tensor([1] * len(self.time_seq[index]), dtype=torch.int8), \
            torch.tensor([self.seg_len[index]], dtype=torch.int8), \
            torch.tensor([self.n_character[index]], dtype=torch.int8)
            

    def __len__(self):
        return len(self.visual_features)

    def collate_fn(self, data):
        """batch合并函数

        Arguments:
            data {Any} -- DataLoader从Dataset调用__getitem__方法的实例，

        Returns:
            List[Any] -- 对Tensor进行合并，其余进行保留（只剩下语义标签了）的列表
        """        
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i], True)  if type(dat[i][0]) == torch.Tensor else dat[i] for i in dat]

    def statistics(self):
        """统计样本中的类别分布情况

        Returns:
            List[Int] -- 保存类别数目的列表，索引位置对应情绪
        """        
        all_emotion = [0] * len(self.emotion_classes)
        for emotion in self.labels:
            all_emotion[emotion] += 1
        return all_emotion

    def get_concept2ids(self):
        return self.concept2id

    def get_useful_ids(self):
        return self.useful_ids

class MEmoRDataLoader_KE(BaseDataLoader):
    """数据加载实例类
    """    
    def __init__(self, config, training=True):
        data_loader_config = config['data_loader']['args']
        self.seed = data_loader_config['seed']
        ####################
        # 动态查找
        dataset_cls = find_model_using_name("data_loader.data_loaders", config['data_loader']['dataset'])
        self.dataset = dataset_cls(config)
        #####################
        self.emotion_nums = self.dataset.statistics()

        #####################################
        ## 修改：将验证集强制为指定文件
        self.val_file = False
        if config["val_file"]:
            self.val_file = True
            print('!!! Caution!  Loading', config['val_id_file'], 'for Validation')
            test_list = list()
            with open(config['val_id_file']) as val_file:
                for line in val_file.readlines():
                    test_list.append(int(line))
            self.valid_idx = np.array(test_list)
        #######################################

        super().__init__(self.dataset, data_loader_config['batch_size'], data_loader_config['shuffle'], data_loader_config['validation_split'], data_loader_config['num_workers'], collate_fn=self.dataset.collate_fn)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(self.seed)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        ##################
        # 制定了测试文件（valtest）就将其作为验证集，否则将训练集分拆出测试集
        if self.val_file:
            valid_idx = self.valid_idx
            train_idx = np.array([idx for idx in idx_full if idx not in valid_idx])
        else:
            valid_idx = idx_full[0:len_valid]
            train_idx = np.delete(idx_full, np.arange(0, len_valid))
        #######################
        weights_per_class = 1. / torch.tensor(self.emotion_nums, dtype=torch.float)
        weights = [0] * self.n_samples
        for idx in range(self.n_samples):
            if idx in valid_idx:
                weights[idx] = 0.
            else:
                label = self.dataset[idx][0]
                weights[idx] = weights_per_class[label]
        weights = torch.tensor(weights)
        train_sampler = data.WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    
        valid_sampler = data.SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def get_concept2ids(self):
        return self.dataset.get_concept2ids()

    def get_useful_ids(self):
        return self.dataset.useful_ids
