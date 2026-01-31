algorithm = 'colearning'
# dataset param
dataset = 'mydataset'
input_channel = 3
num_classes = 3
root = 'data\\data_csv\\train_data.csv'
root_test = 'data\\data_csv\\test_data.csv'
# noise_type = 'Sym'
# noise_type = 'pair'
noise_type = 'ins_dep'
percent = 0.3
seed = 1
loss_type = 'sce'
# model param
model1_type = 'resnest50'
model2_type = 'none'
# train param
batch_size = 64
lr = 0.0001
epochs = 300
num_workers = 4
adjust_lr = 1
epoch_decay_start = 80
# result param
save_result = True
save_path = '/IDN_Weight/best_50%.pth'
save_freq = 1
feature_dim = 32
