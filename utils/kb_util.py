import json
import pickle
from collections import Counter

import numpy as np
import torch
from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords
from senticnet.senticnet import SenticNet
from pymagnitude import Magnitude

from torch.utils.data import dataloader
from torch.nn.utils.rnn import pad_sequence

nltk_stopwords = stopwords.words('english')
# spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS # older version of spacy
stopwords = set(nltk_stopwords).union(spacy_stopwords)
# porter = PorterStemmer()
stemmer = SnowballStemmer("english")
sn = SenticNet()


sn_labels = ["polarity", "pleasantness", "attention", "sensitivity", "aptitude"]

def to_pickle(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

def convert_examples_to_ids(concepts_list, concept2id):
    """将输入的标签按照给定的词汇表处理成index列表

    Arguments:
        concepts_list {List[Str] 或者 Str} -- 输入的标签列表，音频的原始数据因为每个片段只有一个标签，所以保存的只是字符串 \\
        concept2id {Dict[Str, Int]} -- 给定的词汇表

    Returns:
        List[index] -- 根据给定词汇表，输出index列表，输出长度根据输入确定
    """    
    # print(len(concepts_list))
    concepts_ids_list = []
    if type(concepts_list) is list:
        for concept in concepts_list:
            concepts_ids_list.append(concept2id[concept])
            # print(len(concepts_ids_list), concepts_ids_list)
    else:
        concepts_ids_list.append(concept2id[concepts_list])
    return concepts_ids_list

def convert_cocepts_to_ids(concepts_list, concept2id):
    """将给定batch规格的concept列表转换为对应的索引

    Arguments:
        concepts_list_batch {List[List[List[Str]]]} -- 传入的标签列表，尺寸为batch, som_num_seg, some_len \\
        concept2id {Dict[Str, Int]} -- 词汇表，单词到索引

    Returns:
        Tensor n_seg, some_len -- 单个样本的标签的编码向量，每个分段的个数（长度不同）
        Tensor n_seg -- 记录每个片段的标签的个数
    """    
    ids_list = list()
    ids_list_length = list()

    for concepts in concepts_list:
        ids = list()
        for concept in concepts:
            ids.append(concept2id[concept])

        temp = torch.zeros(512, dtype=torch.long)
        ids_tensor = torch.LongTensor(ids)
        temp[:ids_tensor.size(0)] = ids_tensor
        ids_list.append(temp)
        ids_list_length.append(len(ids))

        if len(ids) < 0:
            print("HA", len(ids))

    return torch.stack(ids_list, dim=0), torch.tensor(ids_list_length, dtype=torch.long)


def build_vocab(config, modal, contrast_flag=False):
    """根据给定的模态信息，构造指定的词汇表

    Arguments:
        config {Dict} -- 配置文件，包含所有必要的配置信息 \\
        modal {Str ['visual'|'audio'|'text'|'all']} -- 模态，其中all为所有模态混合在一起

    Returns:
        Dict[Str, Int] -- 词汇表，从word到index \\
        Dict[Int, Str] -- 词汇表，从index到word
    """    
    # 针对给定的标签信息，汇总成词汇表，用于标注使用
    print("Building Vocabulary for [%s] Concepts ..." % (modal.upper()), end=' ')
    
    word2id = {'<unk>': 0}
    id2word = {0: '<unk>'}
    
    assert modal in ['visual', 'audio', 'text', 'all'], "Wrong Argument: MODAL"
    if modal == 'visual':
        faces_concepts = json.load(open(config['visual']['faces_concepts_file']))
        obj_concepts = json.load(open(config['visual']['obj_concepts_file']))
        action_concepts = json.load(open(config['visual']['action_concepts_file']))
        visual_concepts = dict()
        for key in faces_concepts.keys():
            visual_concepts[key] = faces_concepts[key] + obj_concepts[key] + action_concepts[key]
    elif modal == 'audio':
        visual_concepts = json.load(open(config["audio"]['concepts_file']))
    elif modal == 'text':
        visual_concepts = json.load(open(config["text"]['concepts_file']))
    else:
        # faces_concepts = json.load(open(config['visual']['faces_concepts_file'])) # 去掉视频标签中的人脸标签
        obj_concepts = json.load(open(config['visual']['obj_concepts_file']))
        action_concepts = json.load(open(config['visual']['action_concepts_file']))
        # audio_concepts = json.load(open(config["audio"]['concepts_file'])) # 去掉其中的音频标签
        text_concepts = json.load(open(config["text"]['concepts_file']))
        visual_concepts = dict()
        
        visual_concepts_pure = dict()

        # 因为视觉的片段和文本、音频的片段的序号编码不同，需要额外进行处理
        for key in obj_concepts.keys():
            # visual_concepts[key] = faces_concepts[key] + obj_concepts[key] + action_concepts[key]
            visual_concepts[key] = obj_concepts[key] + action_concepts[key]
            visual_concepts_pure[key] = obj_concepts[key] + action_concepts[key]

    # 这里之所以用counter是参考了原始实现，实际上使用dict也能达到同样的效果
    # counter最后没用上，留着备用
    # id2word因为不需要采样也没用上，留着备用

    words_frequency = Counter()
    for _, items in visual_concepts.items():
        words_frequency.update(items)

    # for _, items in audio_concepts.items(): # 去掉音频标签
    #     words_frequency.update([items])

    for _, items in text_concepts.items():
        words_frequency.update(items)

    words_pure = Counter()
    for _, items in visual_concepts_pure.items():
        words_pure.update(items)
    for _, items in text_concepts.items():
        words_pure.update(items)

    # SenticNet标签加入
    words_frequency.update(sn_labels)
    
    # words = [(word, counter) for word, counter in words_frequency.items() if counter >= int(config["knowledge"]["min_frequency"])]
    words = [(word, counter) for word, counter in words_frequency.items()]
    # words = sorted(list(words_frequency.keys()))


    
    useful_words = list()
    useful_ids = list()

    # 构建词汇表的过程中顺便词频过滤，这里的词频过滤存在问题，暂时不考虑启用
    for index, (word, counter) in enumerate(words, start=1):
        if counter >= int(config["knowledge"]["min_frequency"]):
            word2id[word] = index
            id2word[index] = word

            if word in words_pure and word not in stopwords:
                # 由于上述过程有序，因此当前过程也是有序的
                useful_words.append(word)
                useful_ids.append(index)
        else:
            # 如果这个词出现的太少，替换成UNK
            word2id[word] = 0

    for word, counter in words_pure.items():
        if counter >= int(config["knowledge"]["min_frequency"]):
            if word not in stopwords:
                useful_words.append(word)
                useful_ids.append(word2id[word])

    print("Done!")
    if contrast_flag:
        return word2id, id2word, useful_words, useful_ids
    else:
        return word2id, id2word

def build_kb(concept2id, config, modal):
    """创建给定模态内所有标签之间的知识图与情感值（后面可能去掉），知识图实体联系权重来自ConceptNet(O2O)和SenticNet(O2E)，情感值来自NRC_VAD

    Arguments:
        concept2id {Dict[Str, Int]} -- 词汇表，单词对应索引 \\
        config {Dict} -- 配置 \\
        modal {Str ['visual'|'audio'|'text'|'all']} -- 指定构建的模态，其中'all'表示所有模态融合在一起构建

    Returns:
        np.ndarray -- 知识图，对应位置表示两个索引对应实体的连接权重 \\
        
    """ 
    print("Creating KB Graph of [%s], [%s] and [%s] ..." % ("ConceptNet", "NRC_VAD", "SenticNet"), end=' ')   
    conceptnet = load_pickle(config["knowledge"]["conceptnet_file"])
    filtered_conceptnet = filter_conceptnet(conceptnet, concept2id)
    filtered_conceptnet = remove_KB_duplicates(filtered_conceptnet)

    vocab_size = len(concept2id)
    edge_matrix = np.zeros((vocab_size, vocab_size))

    # 基于ConceptNet的连接
    for k in filtered_conceptnet:
        for c,w in filtered_conceptnet[k]:
            edge_matrix[concept2id[k], concept2id[c]] = w

    # 基于SenticNet的连接
    # sn = SenticNet()
    for word in concept2id.keys():
        word_pre = word
        if word not in sn.data and stemmer.stem(word) in sn.data:
            word = stemmer.stem(word)

        if word in sn.data:
            sn_concept = sn.concept(word)
            edge_matrix[concept2id[word_pre], concept2id['polarity']] = sn_concept['polarity_intense']
            edge_matrix[concept2id[word_pre], concept2id['pleasantness']] = sn_concept['sentics']['pleasantness']
            edge_matrix[concept2id[word_pre], concept2id['attention']] = sn_concept['sentics']['attention']
            edge_matrix[concept2id[word_pre], concept2id['sensitivity']] = sn_concept['sentics']['sensitivity']
            edge_matrix[concept2id[word_pre], concept2id['aptitude']] = sn_concept['sentics']['aptitude']
            edge_matrix[concept2id['polarity'], concept2id[word_pre]] = sn_concept['polarity_intense']
            edge_matrix[concept2id['pleasantness'], concept2id[word_pre]] = sn_concept['sentics']['pleasantness']
            edge_matrix[concept2id['attention'], concept2id[word_pre]] = sn_concept['sentics']['attention']
            edge_matrix[concept2id['sensitivity'], concept2id[word_pre]] = sn_concept['sentics']['sensitivity']
            edge_matrix[concept2id['aptitude'], concept2id[word_pre]] = sn_concept['sentics']['aptitude']
            # 保留方案1：全置成1
            # edge_matrix[concept2id[word_pre], concept2id['polarity']] = 1
            # edge_matrix[concept2id[word_pre], concept2id['pleasantness']] = 1
            # edge_matrix[concept2id[word_pre], concept2id['attention']] = 1
            # edge_matrix[concept2id[word_pre], concept2id['sensitivity']] = 1
            # edge_matrix[concept2id[word_pre], concept2id['aptitude']] = 1
            # edge_matrix[concept2id['polarity'], concept2id[word_pre]] = 1
            # edge_matrix[concept2id['pleasantness'], concept2id[word_pre]] = 1
            # edge_matrix[concept2id['attention'], concept2id[word_pre]] = 1
            # edge_matrix[concept2id['sensitivity'], concept2id[word_pre]] = 1
            # edge_matrix[concept2id['aptitude'], concept2id[word_pre]] = 1

    # 保留方案1：全置成1
    # for word in concept2id.keys():
    #     word_pre = word
    #     if word in sn.data:
    #         sn_concept = sn.concept(word)
    #         edge_matrix[concept2id[word_pre], concept2id['polarity']] = 1
    #         edge_matrix[concept2id[word_pre], concept2id['pleasantness']] = 1
    #         edge_matrix[concept2id[word_pre], concept2id['attention']] = 1
    #         edge_matrix[concept2id[word_pre], concept2id['sensitivity']] = 1
    #         edge_matrix[concept2id[word_pre], concept2id['aptitude']] = 1
    #         edge_matrix[concept2id['polarity'], concept2id[word]_pre] = 1
    #         edge_matrix[concept2id['pleasantness'], concept2id[word]_pre] = 1
    #         edge_matrix[concept2id['attention'], concept2id[word]_pre] = 1
    #         edge_matrix[concept2id['sensitivity'], concept2id[word]_pre] = 1
    #         edge_matrix[concept2id['aptitude'], concept2id[word]_pre] = 1
    
    # senticnet的label之间全连接
    for word in sn_labels:
        edge_matrix[concept2id[word], concept2id['polarity']] = 1
        edge_matrix[concept2id[word], concept2id['pleasantness']] = 1
        edge_matrix[concept2id[word], concept2id['attention']] = 1
        edge_matrix[concept2id[word], concept2id['sensitivity']] = 1
        edge_matrix[concept2id[word], concept2id['aptitude']] = 1

    # 本任务暂时不考虑随机删除
    # kb_percentage = config["knowledge"]["kb_percentage"]
    # if kb_percentage > 0: # 参考原始实现，给定一定的随机采样比例
    #         print("Keeping {0}% KB {1} concepts...".format(kb_percentage * 100, modal))
    #         edge_matrix = edge_matrix * (np.random.random((vocab_size,vocab_size)) < kb_percentage).astype(float)


    edge_matrix = torch.FloatTensor(edge_matrix)
    edge_matrix[torch.arange(vocab_size), torch.arange(vocab_size)] = 1
    # print(edge_matrix)
    # incorporate NRC VAD intensity
    
    NRC = load_pickle(config["knowledge"]["affectiveness_file"])
    affectiveness = np.zeros(vocab_size)
    for w, id in concept2id.items():
        VAD = get_emotion_intensity(NRC, w)
        affectiveness[id] = VAD
        # affectiveness[id] = 0.5
    affectiveness = torch.FloatTensor(affectiveness)
    # print(affectiveness)

    print("Done!")
    return edge_matrix, affectiveness

# conceptnet
def get_concept_embedding(concept2id, config, vectors=None):
    """获取所有标签的嵌入表示，初始化为Glove向量，对于标签为短语的，使用所有单词的Glove加和

    Arguments:
        concept2id {Dict[Str, Int]} -- 词汇表，从单词映射到索引 \\
        config {Dict} -- 配置文件示例

    Keyword Arguments:
        vectors {Magnitude} -- 使用pyMagnitude库构建的Magnitude实例，用于获得Glove信息 (default: {None})

    Returns:
        Dict[Int, np.ndarray] -- 所有标签的嵌入表示，索引为键，来自Glove的嵌入表示为值
    """    
    if not vectors:
        vectors = Magnitude(config["knowledge"]["embedding_file"])

    # 这里的实现和原始实现稍许不一样，如果对应不上会直接炸
    pretrained_word_embedding = np.zeros((len(concept2id), config["knowledge"]["embedding_dim"]))
    for word, index in concept2id.items():
        # 对于包含下划线的短语进行处理：二者加和
        if '_' in word:
            words = word.split('_')
            vector_phrase = np.zeros((1, config["knowledge"]["embedding_dim"]))
            for word_item in words:
                vector_phrase += vectors.query(word_item)
            pretrained_word_embedding[index] = vector_phrase
        elif word != '<unk>' and index == 0:
            # 词频过滤情形，直接替换为UNK(index是0)，就不更新embedding了
            pass
        else:
            pretrained_word_embedding[index] = vectors.query(word)

        assert pretrained_word_embedding[index].shape[0] == config["knowledge"]["embedding_dim"]

    # print(pretrained_word_embedding)
    return pretrained_word_embedding


def get_emotion_intensity(emotionKB, word, type='NRC'):
    """获取所有标签的情绪数据库权重，结果的计算来自三个权值的关系连接（KET实现）

    Arguments:
        emotionKB {Dict[List]} -- 情绪数据库
        word {Str} -- 取出情绪值的单词
        type {Str ['senticnet'|'NRC']} -- 使用的情绪知识库

    Returns:
        Int -- 计算好的情绪值
    """    
    
    if type == 'NRC':
        # 从NRC_VAD中取值，按照KET原始实现改造
        if word not in emotionKB and stemmer.stem(word) in emotionKB:
            word = stemmer.stem(word)

        if word in emotionKB:
            v, a, d = emotionKB[word]
            a = a / 2
            affectiveness = (np.linalg.norm(np.array([v, a]) - np.array([0.5, 0])) - 0.06467)/0.607468 # 后面的归一化处理时省略了文章里的min-max
        else:
            affectiveness = 0.5

    else:
        # 从SenticNet取值
        if word not in emotionKB and stemmer.stem(word) in emotionKB:
                word = stemmer.stem(word)

        if word in emotionKB.data:
            item = sn.concept(word)
            po, pl, at, se, ap = float(item["polarity_intense"]), float(item["sentics"]["pleasantness"]), float(item["sentics"]["attention"]), float(item["sentics"]["sensitivity"]), float(item["sentics"]["aptitude"])
            # v_max = 1.9421369673635276
            # v_min = 0.048948953002081665
            affectiveness = (np.linalg.norm(np.array([po, pl, at, se, ap])) - 0.04894)  / (1.9422 - 0.04894)
        else:
            affectiveness = 0.5

    return affectiveness


def filter_conceptnet(conceptnet, concept2id):
    """将原始处理好的concept根据给定的标签进行过滤，没出现过的实体去掉

    Arguments:
        conceptnet {Dict[Str, List[Str, Int]]} -- 已经处理过的ConceptNet实例，键为实体名，值为一系列元组列表，包含了当前实体与所包含实体的联系权重 \\
        concept2id {Dict[Str, Int]} -- 词汇表，单词到索引，这里词汇表只用于提供词汇表

    Returns:
        Dict[Str, List[Str, Int]] -- 进一步过滤的ConceptNet实体，这里仅保留了词汇表中出现的实体与词汇表中出现实体的连接
    """    
    filtered_conceptnet = {}
    for key_concept in conceptnet:
        if key_concept in concept2id and key_concept not in stopwords:
            filtered_conceptnet[key_concept] = set()
            for connect_concept, connect_weight in conceptnet[key_concept]:
                if connect_concept in concept2id and connect_concept not in stopwords and connect_weight>=1:
                    filtered_conceptnet[key_concept].add((connect_concept, connect_weight))
    return filtered_conceptnet


def remove_KB_duplicates(conceptnet):
    """在过滤的concept实例基础上进一步处理重复实体的问题，该问题由于多义词问题引入，为了简化实现，只提取所选项的最大一个

    Arguments:
        conceptnet {Dict[Str, List[Str, Int]]} -- ConceptNet实例，键为实体名，值为一系列元组列表，包含了当前实体与所包含实体的联系权重

    Returns:
        Dict[Str, List[Str, Int]] -- 进一步过滤的ConceptNet实体，去掉了同一个实体对其他实体的重复连接权重信息
    """    
    filtered_conceptnet = {}
    for key_concept in conceptnet:
        filtered_conceptnet[key_concept] = set()
        concepts = set()
        filtered_concepts = sorted(conceptnet[key_concept], key=lambda x: x[1], reverse=True)
        for connect_concept, connect_weight in filtered_concepts:
            if connect_concept not in concepts:
                filtered_conceptnet[key_concept].add((connect_concept, connect_weight))
                concepts.add(connect_concept)
    return filtered_conceptnet