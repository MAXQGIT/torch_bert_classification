from importlib import import_module
import argparse
import torch
from utils import build_data

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
args = parser.parse_args()


# 读取原始数据
def read_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as read:
        text = read.readlines()
        for line in text:
            yield line.strip('\n')


# 读取数据标签数据整理成字典
def label_data():
    label_dict = {}
    index = 0
    with open('THUCNews/data/class.txt', 'r', encoding='utf-8') as read:
        label_list = read.readlines()
        for label in label_list:
            label_dict[index] = label.strip('\n')
            index += 1
    return label_dict


if __name__ == '__main__':
    dataset = 'THUCNews'
    # model_name = args.model
    model_name = 'bert'
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    model = x.Model(config)
    # GPU训练cpu部署
    model.load_state_dict(torch.load(config.save_path, map_location='cpu'))
    model = model.to('cpu')
    data_path = 'THUCNews/data/predict.txt'
    pre_data = build_data(config, data_path)
    label_dict = label_data()

    for text, line in zip(pre_data, read_data(data_path)):
        outputs = model(text)
        predic = torch.max(outputs.data, 1)[1].cpu().numpy()
        print(line)
        print(label_dict[int(predic)])
        print('------------------')
