import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import classification_report, cohen_kappa_score
from models import Model_r18, Model_Rs_50, Model_r50_TM
from tqdm import tqdm
from torch.distributions.beta import Beta
from torch.optim.lr_scheduler import StepLR
from losses import loss_structrue, NTXentLoss, loss_structrue_t

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class robustCPC(nn.Module):  # 继承 nn.Module
    def __init__(self, config: dict = None, input_channel: int = 3, num_classes: int = 3, col=192):
        super(robustCPC, self).__init__()  # 调用父类构造函数
        # 生成 transition matrix 为可学习参数
        self.transition_matrix = nn.Parameter(torch.randn(num_classes, num_classes).to(device))

        self.batch_size = config['batch_size']
        self.lr = config['lr']
        self.num_classes = config['num_classes']
        self.feature_extractor = Model_r50_TM(feature_dim=self.num_classes * col, is_linear=True,
                                               num_classes=num_classes).to(device)

        # 增大正则化权重
        self.lambda_reg = config.get('lambda_reg', 0.5)  # 调整正则化权重
        self.k1 = config.get('k1', 5)  # 内部近邻数
        self.k2 = config.get('k2', 5)  # 外部近邻数
        self.sigma = config.get('sigma', 1.0)  # 高斯核参数

        mom1 = 0.9
        mom2 = 0.1
        self.alpha_plan = [self.lr] * config['epochs']
        self.beta1_plan = [mom1] * config['epochs']

        for i in range(config['epoch_decay_start'], config['epochs']):
            self.alpha_plan[i] = float(config['epochs'] - i) / (
                        config['epochs'] - config['epoch_decay_start']) * self.lr
            self.beta1_plan[i] = mom2

        self.device = device
        self.epochs = config['epochs']

        # scratch
        self.model_scratch = Model_Rs_50(feature_dim=config['feature_dim'], is_linear=True, num_classes=num_classes).to(
            device)
        self.optimizer1 = torch.optim.Adam(self.model_scratch.parameters(), lr=self.lr)
        self.optimizer2 = torch.optim.Adam(list(self.model_scratch.fc.parameters()), lr=self.lr / 5)
        self.scheduler = StepLR(self.optimizer1, step_size=150, gamma=0.3)  # 每 150 个 epoch 学习率减小 0.5 倍
        self.adjust_lr = config['adjust_lr']
        self.ntxent = NTXentLoss(self.device, self.batch_size, temperature=0.5, use_cosine_similarity=True)
        self.param_v = config.get('distribution_t', None)

    def mixup_data(self, x, y, alpha=5.0):  # 调整 alpha 的值
        lam = Beta(torch.tensor(alpha), torch.tensor(alpha)).sample() if alpha > 0 else 1
        index = torch.randperm(x.size()[0]).to(self.device)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam=0.5):
        return (lam * F.cross_entropy(pred, y_a, reduction='none') + (1 - lam) * F.cross_entropy(pred, y_b,
                                                                                                 reduction='none')).mean()

    def compute_affinity_matrix(self, X, k, sigma, same_label=True):
        dist_matrix = torch.cdist(X, X, p=2)
        S = torch.exp(-dist_matrix ** 2 / (2 * sigma ** 2))

        for i in range(X.size(0)):
            sorted_idx = torch.argsort(S[i], descending=True)
            if same_label:
                S[i, sorted_idx[k:]] = 0
            else:
                S[i, sorted_idx[:k]] = 0
        return S

    def manifold_regularization(self, feature_space, S_I, S_B):
        S = S_I - S_B
        D = torch.diag(S.sum(dim=1))
        L = D - S
        T_flat = feature_space.view(feature_space.size(0), -1)
        M = torch.trace(torch.mm(T_flat.T, torch.mm(L, T_flat)))
        return M

    def evaluate(self, test_loader):
        # 抑制CuDNN警告
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning, message="Plan failed with a cudnnException*")
        
        self.model_scratch.eval()

        correct2 = 0
        total2 = 0
        all_true_labels = []
        all_predicted_labels = []

        for images, labels in test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                _, _, logits2 = self.model_scratch(images)
                outputs2 = F.softmax(logits2, dim=1)
                _, pred2 = torch.max(outputs2, dim=1)

                total2 += labels.size(0)
                correct2 += (pred2 == labels).sum().item()

                all_true_labels.extend(labels.cpu().numpy())
                all_predicted_labels.extend(pred2.cpu().numpy())

        acc2 = 100 * float(correct2) / float(total2)
        kappa = cohen_kappa_score(all_true_labels, all_predicted_labels)
        # 添加zero_division参数，避免Precision计算警告
        report = classification_report(all_true_labels, all_predicted_labels, output_dict=True, zero_division=0)

        f1_score = report['macro avg']['f1-score'] if 'macro avg' in report else 0.0
        recall = report['macro avg']['recall'] if 'macro avg' in report else 0.0

        # 移除打印语句
        # print("Classification Report:", report)
        # print(f"Kappa: {kappa:.4f}")

        return acc2, kappa, f1_score, recall, all_true_labels, all_predicted_labels

    def train(self, train_loader, epoch):
        # 直接打印当前epoch信息，避免tqdm缓存问题
        print(f'Training Epoch [{epoch}/{self.epochs}]...')
        self.model_scratch.train()

        if self.adjust_lr:
            self.adjust_learning_rate(self.optimizer1, epoch)
            self.adjust_learning_rate(self.optimizer2, epoch)

        # 使用更慢的非线性方式引入 transition_matrix 的影响
        alpha = min(1.0, (epoch / self.epochs) ** 5)

        # 使用更简单的进度条描述，避免epoch显示混淆
        pbar = tqdm(train_loader, desc='Training batch', leave=False)
        for item in pbar:
            raw, pos_1, pos_2, labels = item[0:4]
            pos_1 = pos_1.to(self.device, non_blocking=True)
            pos_2 = pos_2.to(self.device, non_blocking=True)
            labels = labels.to(self.device)
            raw = raw.to(self.device, non_blocking=True)

            feat, outs, logits = self.model_scratch(raw)
            if self.param_v is None:
                loss_feat = loss_structrue(outs.detach(), logits)
            else:
                loss_feat = loss_structrue_t(outs.detach(), logits, self.param_v)
            self.optimizer2.zero_grad()
            loss_feat.backward()
            self.optimizer2.step()

            # Self-learning
            out_1 = self.model_scratch(pos_1, ignore_feat=True, forward_fc=False)
            out_2 = self.model_scratch(pos_2, ignore_feat=True, forward_fc=False)
            loss_con = self.ntxent(out_1, out_2)

            # Supervised-learning
            feat, outs, logits = self.model_scratch(raw)
            logits = F.softmax(logits, dim=1)
            inputs, targets_a, targets_b, lam = self.mixup_data(raw, labels, alpha=3.0)  # 调整 mixup 强度

            # 逐渐引入 transition_matrix 的影响
            transition_matrix_normalized = F.softmax(self.transition_matrix, dim=-1)  # 对 transition matrix 进行归一化
            logits_with_transition = torch.matmul(logits, transition_matrix_normalized)

            # 使用 Residual 方式引入 transition_matrix 的影响
            logits = (1 - alpha) * logits + alpha * logits_with_transition

            # 计算监督学习损失
            loss_sup = self.mixup_criterion(logits, targets_a, targets_b, lam)

            # 流形正则化
            with torch.no_grad():
                feature_space = feat.detach()
                S_I = self.compute_affinity_matrix(feature_space, self.k1, self.sigma, same_label=True)
                S_B = self.compute_affinity_matrix(feature_space, self.k2, self.sigma, same_label=False)

            # 添加流形正则化项
            M_reg = self.manifold_regularization(feature_space, S_I, S_B)
            loss = loss_sup + loss_con + self.lambda_reg * M_reg

            # 优化
            self.optimizer1.zero_grad()
            loss.backward()
            self.optimizer1.step()

            pbar.set_description(
                'Epoch [%d/%d], loss_sup: %.4f, M_reg: %.4f'
                % (epoch + 1, self.epochs, loss_sup.item(), M_reg.item())
            )
        pbar.close()

        # 更新学习率
        self.scheduler.step()

    def adjust_learning_rate(self, optimizer, epoch):
        # 使用epoch-1作为索引，因为epoch从1开始，而列表索引从0开始
        epoch_idx = epoch - 1
        # 确保索引不超出范围
        epoch_idx = min(epoch_idx, len(self.alpha_plan) - 1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.alpha_plan[epoch_idx]
            param_group['betas'] = (self.beta1_plan[epoch_idx], 0.999)
