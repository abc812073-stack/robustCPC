import torch
import numpy as np

'''
Thanks to https://github.com/Spijkervet/SimCLR/blob/master/modules/nt_xent.py.
'''

class NTXentLoss(torch.nn.Module):
    # 是否在计算时使用余弦相似度
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    # 生成掩码张量,屏蔽具有相同表示的样本对
    # 生成一个掩码张量，用于在训练时屏蔽具有相同表示的样本对，需要屏蔽的样本对值为1，不需要屏蔽的样本对值为0
    def _get_correlated_mask(self):
        # 生成一个2 * batch_size大小的单位矩阵
        diag = np.eye(2 * self.batch_size)

        # 创建一个(2 * batch_size),(2 * batch_size)的单位矩阵，将对角线下移动batch_size
        # k>0:对角线为主对角线，向右移动    k<0：副对角线，向下移动
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)     # 下对角线矩阵
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)      # 上对角线矩阵
        # 将单位矩阵与上下三角矩阵相加并转换为Tensor类型
        mask = torch.from_numpy((diag + l1 + l2))
        # 将矩阵中所有的值取反，使0和1位置互调，转化为bool类型
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    # 点积
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    # 求余弦，内积
    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        # 将张量zjs和zis进行拼接
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
