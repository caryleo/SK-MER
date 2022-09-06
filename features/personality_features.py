import torch
import torch.nn.functional as F
from base import BaseFeatureExtractor

class PersonalityFeatureExtractor(BaseFeatureExtractor):
    """人格特征提取类
    """    
    def __init__(self, config):
        print("Initializing Extracor", '[' + self.__class__.__name__ + ']', "...", end=' ')
        self.characters = config['speakers']
        self.features = []
        with open(config['personality']['anno_file']) as fin:
            for ii, line in enumerate(fin.readlines()):
                features = [float(i) for i in line.strip().split(',')]
                self.features.append(features)
        self.features = torch.tensor(self.features)
        self.features = F.normalize(self.features, dim=0)
        print("Done!")
    
    def get_features(self):
        """提取人格信息

        Returns:
            Tensor n_character_all, dim_feature_p -- 全部角色的数值化人格特征
        """        
        return self.features


