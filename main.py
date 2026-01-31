import argparse
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
from utils import load_config, get_log_name, set_seed, save_results, \
    plot_results, get_test_acc, print_config
from datasets import cifar_dataloader
import algorithms
import numpy as np
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', type=str, default='./configs/cpc.py',
                    help='The path of config file.')
parser.add_argument('--mode', '-m', type=str, choices=['train', 'eval'], default='eval',
                    help='Mode to run the script. Options are "train" for training or "eval" for evaluation.')

args = parser.parse_args()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def plot_accuracy(accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', linestyle='-')
    plt.title('Test Accuracy During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(true_labels, predicted_labels, class_names=None, title="Confusion Matrix"):
    """
    绘制并显示混淆矩阵
    :param true_labels: 真实标签列表
    :param predicted_labels: 预测标签列表
    :param class_names: 类别名称（可选）
    :param title: 图标题
    """
    cm = confusion_matrix(true_labels, predicted_labels)
    # 创建figure并获取当前axes
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    # 在指定的axes上绘制，避免创建新的figure
    disp.plot(cmap=plt.cm.Blues, colorbar=True, ax=ax)
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()


def evaluate_model(model, testloader):
    # 确保模型处于评估模式
    if hasattr(model, 'model_scratch'):
        model.model_scratch.eval()
    else:
        model.eval()
    
    num_test_images = len(testloader.dataset)
    acc2, kappa, f1_score, recall, all_true_labels, all_predicted_labels = model.evaluate(testloader)
    # print('Test Accuracy on the %s test images: %.4f %%' % (num_test_images, acc2))
    # print('Kappa: %.4f, F1-score: %.4f, Recall: %.4f' % (kappa, f1_score, recall))
    return acc2, kappa, f1_score, recall, all_true_labels, all_predicted_labels


def train_model(model, trainloader, testloader, config):
    acc_list, acc_all_list = [], []
    best_acc, best_epoch = 0.0, 0
    epoch_accs = []
    best_kappa, best_f1, best_recall = 0.0, 0.0, 0.0
    best_true_labels, best_predicted_labels = None, None  # ✅ 新增：保存最佳混淆矩阵对应标签

    # 保存最佳模型权重的路径（使用配置文件中的路径）
    save_path = config.get('save_path', 'weight\best_model.pth')
    
    # 处理相对路径，确保基于项目根目录
    if save_path.startswith('/'):
        # 移除开头的 '/'，并基于项目根目录构建绝对路径
        project_root = os.path.abspath(os.path.dirname(__file__))
        save_path = os.path.join(project_root, save_path.lstrip('/'))
        # 统一路径分隔符为Windows格式
        save_path = save_path.replace('/', '\\')
    
    save_dir = os.path.dirname(save_path)
    os.makedirs(save_dir, exist_ok=True)
    best_model_path = save_path
    print(f"Model weights will be saved to: {best_model_path}")

    # 抑制CuDNN警告
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, message="Plan failed with a cudnnException*")

    for epoch in range(1, config['epochs'] + 1):
        model.train(trainloader, epoch)

        # 评估模型
        acc2, kappa, f1_score, recall, all_true_labels, all_predicted_labels = model.evaluate(testloader)
        epoch_accs.append(acc2)

        # 更新最佳指标并保存模型
        if best_acc < acc2:
            best_acc, best_epoch = acc2, epoch
            best_kappa, best_f1, best_recall = kappa, f1_score, recall
            best_true_labels, best_predicted_labels = all_true_labels, all_predicted_labels  # ✅ 保存最佳结果对应标签

            # 保存最佳模型权重，确保保存完整状态
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model updated and saved at epoch {epoch} with accuracy: {best_acc:.4f}%")
            
            # 同时保存最佳指标，方便后续比较
            best_metrics = {
                'acc': best_acc,
                'kappa': best_kappa,
                'f1': best_f1,
                'recall': best_recall,
                'epoch': best_epoch
            }
            metrics_path = os.path.join(os.path.dirname(best_model_path), 'best_metrics.pth')
            torch.save(best_metrics, metrics_path)
            print(f"Best metrics saved to: {metrics_path}")
        else:
            # 即使没有超过最佳值，也打印当前epoch的信息
            print(f"Current model at epoch {epoch} with accuracy: {acc2:.4f}%")
            print(f"Current metrics: Kappa: {kappa:.4f}, F1: {f1_score:.4f}, Recall: {recall:.4f}")
            print(f"Best so far: epoch {best_epoch} with accuracy: {best_acc:.4f}%")

        print(f'Epoch [{epoch:3d}/{config["epochs"]:3d}] | '
              f'Images: {len(testloader.dataset):6d} | '
              f'Acc: {acc2:.4f}% | '
              f'Kappa: {kappa:.4f} | '
              f'F1: {f1_score:.4f} | '
              f'Recall: {recall:.4f}')

        # 保存最后10个epoch的准确率
        if epoch >= config['epochs'] - 10:
            acc_list.append(acc2)
        acc_all_list.append(acc2)

    # 打印最佳模型的所有指标
    print('\n=====================================')
    print('Best Model Performance Summary')
    print('=====================================')
    print(f'Epoch: {best_epoch}')
    print(f'Accuracy: {best_acc:.4f}%')
    print(f'Kappa: {best_kappa:.4f}')
    print(f'F1-score: {best_f1:.4f}')
    print(f'Recall: {best_recall:.4f}')
    print('=====================================')

    # ✅ 在训练全部结束后再绘制一次最佳模型的混淆矩阵
    if best_true_labels is not None and best_predicted_labels is not None:
        plot_confusion_matrix(best_true_labels, best_predicted_labels,
                              title=f"Best Model Confusion Matrix (Epoch {best_epoch})")

    plot_accuracy(epoch_accs)


def main():
    # 加载配置文件
    config = load_config(args.config, _print=False)
    print_config(config)
    
    # 固定随机种子，确保结果可重现
    set_seed(config['seed'])
    # 额外固定 PyTorch 和 NumPy 的随机种子
    import random
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed fixed to: {config['seed']}")

    # 创建模型
    if config['algorithm'] == 'colearning':
        model = algorithms.robustCPC(config, input_channel=config['input_channel'], num_classes=config['num_classes'])
    else:
        model = algorithms.__dict__[config['algorithm']](config, input_channel=config['input_channel'],
                                                         num_classes=config['num_classes'])

    # 加载保存的最佳模型权重（如果存在）
    # best_model_path = None
    best_model_path = 'IDN_Weight\\best_30%.pth'  # 指定权重路径
    print(f"Current mode: {args.mode}")
    print(f"Looking for model at: {best_model_path}")
    print(f"Model exists: {os.path.exists(best_model_path) if best_model_path else False}")
    
    if args.mode == 'eval':
        print("Entering evaluation mode...")
        if best_model_path and os.path.exists(best_model_path):
            print(f"Loading model from: {best_model_path}")
            try:
                # 加载模型权重时指定设备，确保与训练时一致
                model.load_state_dict(torch.load(best_model_path, map_location=device))
                print(f"Pretrained model loaded from {best_model_path}. Running only evaluation.")
                model.to(device)
                
                # 确保模型处于评估模式
                if hasattr(model, 'model_scratch'):
                    model.model_scratch.eval()
                    print("Set model_scratch to eval mode")
                else:
                    model.eval()
                    print("Set model to eval mode")

                dataloaders = cifar_dataloader(cifar_type=config['dataset'], root=config['root'],
                                               batch_size=config['batch_size'], num_workers=0,  # 设为0避免多进程问题
                                               noise_type=config['noise_type'], percent=config['percent'])
                _, testloader = dataloaders.run(mode='train'), dataloaders.run(mode='test')
                print(f"Test loader created with {len(testloader.dataset)} samples")
                
                # 加载最佳指标，方便比较
                metrics_path = os.path.join(os.path.dirname(best_model_path), 'best_metrics.pth')
                if os.path.exists(metrics_path):
                    best_metrics = torch.load(metrics_path)
                    print(f"Best metrics: Accuracy={best_metrics['acc']:.4f}, Kappa={best_metrics['kappa']:.4f}, F1={best_metrics['f1']:.4f}, Recall={best_metrics['recall']:.4f}")
                
                # 评估模型并获取返回值
                eval_acc, eval_kappa, eval_f1, eval_recall, _, _ = evaluate_model(model, testloader)
                
                # 比较结果
                # print(f"Evaluation results: Accuracy={eval_acc:.4f}, Kappa={eval_kappa:.4f}")
                # if os.path.exists(metrics_path):
                #     print(f"Difference: Accuracy={abs(eval_acc - best_metrics['acc']):.4f}, Kappa={abs(eval_kappa - best_metrics['kappa']):.4f}")
                #     if abs(eval_acc - best_metrics['acc']) < 0.1:
                #         print("✅ Results match training best accuracy!")
                #     else:
                #         print("❌ Results do not match training best accuracy!")
                
                # print("Evaluation completed. Exiting...")
                return  # 仅在找到模型时退出
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Falling back to training mode...")
        else:
            # ✅ 自动切换为训练
            print(f"No pretrained model found at {best_model_path}. Starting training from scratch instead.")

    # 如果没有加载预训练模型，开始训练
    print("No saved model found, starting training from scratch.")
    model.to(device)  # 将模型转移到计算设备（GPU或CPU）

    # 加载数据，确保与测试时参数一致
    dataloaders = cifar_dataloader(cifar_type=config['dataset'], root=config['root'], batch_size=config['batch_size'],
                                   num_workers=0,  # 设为0避免多进程随机性
                                   noise_type=config['noise_type'], percent=config['percent'])
    trainloader, testloader = dataloaders.run(mode='train'), dataloaders.run(mode='test')
    print(f"Train loader created with {len(trainloader.dataset)} samples")
    print(f"Test loader created with {len(testloader.dataset)} samples")

    # 训练模型
    train_model(model, trainloader, testloader, config)


if __name__ == '__main__':
    main()
