import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
import sys
sys.path.append("../")

from dataset.dataloader_merge_r0_r1 import CLSDataset
from models.bert_molecule_prediction_new import *
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec

import math
import tqdm
import pandas as pd
import numpy as np
import os
import pickle
import argparse

# 配置bert模型参数


class Trainer:
    def __init__(self, max_seq_len,
                 batch_size,
                 lr, # 学习率
                 config,
                 with_cuda=True, # 是否使用GPU, 如未找到GPU, 则自动切换CPU
                 ):

        self.config = config
        self.vocab_size = int(self.config["vocab_size"])
        self.batch_size = batch_size
        self.lr = lr

        ident = open(self.config["word2idx_path"], 'rb')
        self.word2idx = pickle.load(ident)
        self.vocab_size = len(self.word2idx)
        # 判断是否有可用GPU
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:"+str(config["cudaId"]) if cuda_condition else "cpu")
        # 允许的最大序列长度
        self.max_seq_len = max_seq_len
        #self.mol2vec_model = word2vec.Word2Vec.load('model_300dim.pkl')
        #self.keys = set(self.mol2vec_model.wv.vocab.keys())
        # 定义模型超参数
        bertconfig=BertConfig(
                vocab_size=int(self.config["vocab_size"]),
                hidden_size=int(self.config["hidden_size"]),  # 隐藏层维度也就是字向量维度
                num_hidden_layers=int(self.config["num_hidden_layers"]),  # transformer block 的个数
                num_attention_heads=int(self.config["num_attention_heads"]),  # 注意力机制"头"的个数
                intermediate_size=int(self.config["vocab_size"]),  # feedforward层线性映射的维度
                work_nums=config["work_nums"],
                hidden_dropout_prob=float(config["hidden_dropout_prob"]),
                attention_probs_dropout_prob=float(config["attention_probs_dropout_prob"])
            )
        # 初始化bert模型
        self.bert_model = Bert_Smiles_Analysis(config=bertconfig,deviceId=config["cudaId"],
                                               isRegression=config["isRegression"],base_config=config)
        # 将模型发送到计算设备(GPU或CPU)
        self.bert_model.to(self.device)
        # 声明训练数据集, 按照pytorch的要求定义数据集class
        train_dataset = CLSDataset(corpus_path=self.config["train_path"],
                                   word2idx=self.word2idx,
                                   max_seq_len=self.max_seq_len,
                                   dataset_name=self.config["dataset_name"],
                                   data_regularization=False,
                                   #model=self.mol2vec_model,
                                   #keys=self.keys
                                   )
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=self.batch_size,
                                           num_workers=0,
                                           collate_fn=lambda x: x # 这里为了动态padding
                                           )
        # 声明测试数据集
        test_dataset = CLSDataset(corpus_path=self.config["test_path"],
                                  word2idx=self.word2idx,
                                  max_seq_len=self.max_seq_len,
                                  dataset_name=self.config["dataset_name"],
                                  data_regularization=False,
                                  #model=self.mol2vec_model,
                                  #keys=self.keys
                                  )
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=self.batch_size,
                                          num_workers=0,
                                          collate_fn=lambda x: x)

        valid_dataset = CLSDataset(corpus_path=self.config["validation_path"],
                                  word2idx=self.word2idx,
                                  max_seq_len=self.max_seq_len,
                                  dataset_name=self.config["dataset_name"],
                                  data_regularization=False,
                                   #model=self.mol2vec_model,
                                   #keys=self.keys
                                  )
        self.valid_dataloader = DataLoader(valid_dataset,
                                          batch_size=self.batch_size,
                                          num_workers=0,
                                          collate_fn=lambda x: x)
        # 初始化位置编码
        self.hidden_dim = bertconfig.hidden_size
        self.positional_enc = self.init_positional_encoding()
        # 扩展位置编码的维度, 留出batch维度,
        # 即positional_enc: [batch_size, embedding_dimension]
        self.positional_enc = torch.unsqueeze(self.positional_enc, dim=0)

        # 声明需要优化的参数, 并传入Adam优化器
        self.optim_parameters = list(self.bert_model.parameters())
        # print(len(self.optim_parameters))
        # all_parameters = list(self.bert_model.named_parameters())
        # lis_ = ["dense.weight", "dense.bias", "final_dense.weight", "final_dense.bias"]
        # # self.optim_parameters = [i[1] for i in all_parameters if i[0] in lis_]
        # self.optim_parameters = list(self.bert_model.parameters())

        self.init_optimizer(lr=self.lr)
        if not os.path.exists(self.config["state_dict_dir"]):
            if not os.path.exists("../finetu_output"):
                os.mkdir("../finetu_output")
            os.mkdir(self.config["state_dict_dir"])

    def init_optimizer(self, lr, weight_decay=5e-3):
        # 用指定的学习率初始化优化器
        self.optimizer = torch.optim.Adam(self.optim_parameters, lr=lr, weight_decay=weight_decay)
        # self.optimizer = torch.optim.SGD(self.optim_parameters, lr=lr, momentum=0, dampening=0, weight_decay=1e-3)

    def init_positional_encoding(self):
        position_enc = np.array([
            [pos / np.power(10000, 2 * i / self.hidden_dim) for i in range(self.hidden_dim)]
            if pos != 0 else np.zeros(self.hidden_dim) for pos in range(self.max_seq_len)])

        position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
        position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
        denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
        # 归一化
        position_enc = position_enc / (denominator + 1e-8)
        position_enc = torch.from_numpy(position_enc).type(torch.FloatTensor)
        return position_enc

    def load_model(self, model, dir_path="../output", load_bert=False, specialModelId=None):
        if specialModelId == None:
            checkpoint_dir = self.find_most_recent_state_dict(dir_path)
        else:
            if load_bert == False:
                checkpoint_dir = dir_path + "/model_path" + "/bert.model.epoch." + str(specialModelId) + ".pth"
            else:
                checkpoint_dir = dir_path + "/bert.model.epoch." + str(specialModelId)
        print('model name', checkpoint_dir)
        checkpoint = torch.load(checkpoint_dir)
        if load_bert:
            checkpoint["model_state_dict"] = {k[5:]: v for k, v in checkpoint["model_state_dict"].items()
                                              if k[:4] == "bert" and "pooler" not in k}
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        torch.cuda.empty_cache()
        model.to(self.device)
        print("{} loaded!".format(checkpoint_dir))

    def train(self, epoch):
        # 一个epoch的训练
        self.bert_model.train()
        return self.iteration(epoch, self.train_dataloader, train="train")

    def validation(self, epoch):
        # 一个epoch的测试, 并返回验证集的auc
        self.bert_model.eval()
        with torch.no_grad():
            return self.iteration(epoch, self.valid_dataloader, train="validation")
            # return self.iteration(epoch, self.test_dataloader, train="test")


    def test(self, epoch):
        # 一个epoch的测试, 并返回测试集的auc
        self.bert_model.eval()
        with torch.no_grad():
            # return self.iteration(epoch, self.valid_dataloader, train="validation")

            return self.iteration(epoch, self.test_dataloader, train="test")

    def padding(self, output_dic_lis):
        """动态padding, 以当前mini batch内最大的句长进行补齐长度"""
        text_input = [i["text_input"] for i in output_dic_lis]
        #text_0 = torch.cat([i["text_0"] for i in output_dic_lis])
        #fp = torch.cat([i["fp"] for i in output_dic_lis])
        # text_0 =
        text_input = torch.nn.utils.rnn.pad_sequence(text_input, batch_first=True)
        label = torch.cat([i["label"] for i in output_dic_lis])
        return {"text_input": text_input,
                #"text_0" : text_0,
                "label": label,}
        #       "fp": fp}

    def iteration(self, epoch, data_loader, train=True):


        # 进度条显示
        str_code = "train" if train == "train" else "validation"
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        total_loss = 0.0
        # 存储所有预测的结果和标记, 用来计算auc
        all_predictions, all_labels,all_loss = [], [], []
        final_loss = 0.0
        for i, data in data_iter:
            # padding
            data = self.padding(data)
            # 将数据发送到计算设备
            data = {key: value.to(self.device) for key, value in data.items()}
            # print(data.keys())
            # 根据padding之后文本序列的长度截取相应长度的位置编码,
            # 并发送到计算设备
            positional_enc = self.positional_enc[:, :data["text_input"].size()[-1], :].to(self.device)

            # 正向传播, 得到预测结果和loss
            #print(data["label"].shape,data["text_input"].shape)
            predictions, loss, loss2 = self.bert_model.forward(text_input=data["text_input"],
                                                        positional_enc=positional_enc,
                                                        labels=data["label"],
                                                        #text_input1=data["text_0"]      
                                                  )
            if self.config["isRegression"] == True:
                total_loss += loss.item()
                final_loss = total_loss / (i + 1)
                # 反向传播
                if train == "train":
                    # 清空之前的梯度
                    self.optimizer.zero_grad()
                    # 反向传播, 获取新的梯度
                    if loss2 == None:
                        loss.backward()
                    else:
                        loss2.backward()
                    # 用获取的梯度更新模型参数
                    self.optimizer.step()

            else:
                if self.config["work_nums"] == 1:

                    # 提取预测的结果和标记, 并存到all_predictions, all_labels里
                    # 用来计算auc
                    # print("predict: ", predictions)
                    # print("label--: ", data["label"])

                    predictions = predictions.detach().cpu().numpy().reshape(-1).tolist()
                    labels = data["label"].cpu().numpy().reshape(-1).tolist()
                    # predictions = np.array([1, 1, 1, 1, 1, 1, 0, 0])w
                    all_predictions.extend(predictions)
                    all_labels.extend(labels)
                    # total_loss += loss.item()
                    # print(all_predictions)
                    # 计算auc
                    # fpr, tpr, thresholds = metrics.roc_curve(y_true=all_labels,
                    #                                          y_score=all_predictions)
                    # auc = metrics.auc(fpr, tpr)

                else:
                    # predictions = predictions.detach().cpu().numpy()
                    # labels = data["label"].cpu().numpy()

                    all_predictions.append(predictions)
                    all_labels.append(data["label"])
                    # print(all_predictions[-1].shape)
                    # print(all_labels[-1].shape)
                    #print(len(predictions),len(labels))
                    # 计算auc
                    # print(all_labels)
                    # print(all_predictions)
                    # fpr, tpr, thresholds = metrics.roc_curve(y_true=all_labels,
                    #                                          y_score=all_predictions,pos_label=self.config["work_nums"])
                    # auc = metrics.auc(fpr, tpr)
                #auc = 0.6
                # 反向传播
                if train == "train":
                    # 清空之前的梯度
                    self.optimizer.zero_grad()
                    # 反向传播, 获取新的梯度
                    loss.backward()
                    # 用获取的梯度更新模型参数
                    self.optimizer.step()

                # 为计算当前epoch的平均loss
                total_loss += loss.item()

                # threshold_ = find_best_threshold(all_predictions, all_labels)
                # print(str_code + " best threshold: " + str(threshold_))

        # 在最后统计
        # 返回auc, 作为early
        if self.config["isRegression"] == True:
            return final_loss
        elif self.config["work_nums"] == 1:
            # 二分类统计auc值
            fpr, tpr, thresholds = metrics.roc_curve(y_true=all_labels,
                                                     y_score=all_predictions)
            auc = metrics.auc(fpr, tpr)
            print(total_loss / (i+1))
        else:
            # 多分类auc值
            # all_labels = np.array(all_labels)
            # all_predictions = np.array(all_predictions)
            all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
            all_predictions = torch.cat(all_predictions, dim=0).detach().cpu().numpy()

            roc_list = []
            for i in range(all_labels.shape[1]):
                # AUC is only defined when there is at least one positive data.
                if np.sum(all_labels[:, i] == 1) > 0 and np.sum(all_labels[:, i] == -1) > 0:
                    is_valid = all_labels[:, i] ** 2 > 0
                    roc_list.append(roc_auc_score((all_labels[is_valid, i] + 1) / 2, all_predictions[is_valid, i]))

            if len(roc_list) < all_labels.shape[1]:
                print("Some target is missing!")
                print("Missing ratio: %f" % (1 - float(len(roc_list)) / all_labels.shape[1]))
            print(total_loss / (i+1))
            return sum(roc_list) / len(roc_list)  # y_true.shape[1]
        return auc

    def find_most_recent_state_dict(self, dir_path):
        """
        :param dir_path: 存储所有模型文件的目录
        :return: 返回最新的模型文件路径, 按模型名称最后一位数进行排序
        """
        dic_lis = [i for i in os.listdir(dir_path)]
        if len(dic_lis) == 0:
            raise FileNotFoundError("can not find any state dict in {}!".format(dir_path))
        dic_lis = [i for i in dic_lis if "model" in i]
        dic_lis = sorted(dic_lis, key=lambda k: int(k.split(".")[-1]))
        return dir_path + "/" + dic_lis[-1]

    def save_state_dict(self, model, epoch, state_dict_dir="../output", file_path="bert.model"):
        """存储当前模型参数"""
        if not os.path.exists(state_dict_dir):
            os.mkdir(state_dict_dir)

        save_path = state_dict_dir + file_path + ".epoch.{}".format(str(epoch)) + ".pth"
        if os.path.exists(save_path):  # 覆盖之前的文件
            # print(save_path)
            os.remove(save_path)

        model.to("cpu")
        torch.save({"model_state_dict": model.state_dict()}, save_path)
        print("{} saved!".format(save_path))
        model.to(self.device)




def init_trainer(dynamic_lr, config, batch_size=8,):
    trainer = Trainer(max_seq_len=200,
                                batch_size=batch_size,
                                lr=dynamic_lr,
                                config = config,
                                with_cuda=True,)
    return trainer, dynamic_lr

def main():
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of BERT')
    parser.add_argument('--device', type=int, default=3,
                    help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=8,
                    help='input batch size for training (default: 8)')
    parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-5,
                    help='learning rate (default: 2e-5)')
    parser.add_argument('--dataset_name', type=str, default="bbbp",
                        help='dataset_name (default: BBBP)')
    parser.add_argument('--hidden_size', type=int, default=300,
                        help='hidden_size (default: 300)')
    parser.add_argument('--num_hidden_layers', type=int, default=6,
                        help='the number of hidden layers (default: 6)')
    parser.add_argument('--num_attention_heads', type=int, default=6,
                        help='the number of attention heads (default: 6)')
    parser.add_argument('--root_path', type=str, default="../dataset",
                        help='downstream_dataset (default: ../dataset)')
    parser.add_argument('--pretraining_model_path', type=str, default="../output_model",
                        help='downstream_dataset (default: ../output_model)')
    parser.add_argument('--model_Id', type=int, default=3,
                        help='pretraining_model_Id (default: 3)')
    parser.add_argument('--cudaId', type=int, default=2,
                        help='avaliable cudaId (default:  2)')
    parser.add_argument('--vocab_size', type=int, default=64711,
                        help='avaliable cudaId (default:  2)')
    # hidden_dropout_prob = 0.4,  # dropout的概率
    # attention_probs_dropout_prob = 0.4,
    parser.add_argument('--hidden_dropout_prob',type=float,default=0.4,
                        help='dropout_rate')
    parser.add_argument('--attention_probs_dropout_prob', type=float, default=0.4,
                        help='dropout_rate')
    parser.add_argument('--dr', type=float, default=1.0,
                        help='dalymic learning rate')
    args = parser.parse_args()

    # if args.dr == 1.0:
    #     lr_list = [1e-4, 2e-4, 5e-4, 5e-5, 2e-5, 1e-5, 5e-6, 1e-6, 5e-7]
    # else:
    #     lr_list = [5e-2, 2e-2,1e-3, 5e-3, 1e-4, 2e-4, 5e-4, 5e-5, 2e-5, 1e-5, 5e-6, 1e-6, 5e-7]
    # lr_list = [2e-5, 5e-5]
    lr_list = [0.0001, 2e-5, 5e-5]

    config = {}
    config["hidden_dropout_prob"] = args.hidden_dropout_prob
    config["attention_probs_dropout_prob"] = args.attention_probs_dropout_prob
    config["word2idx_path"] = "../croups/ident_base.pickle"
    config["dataset_name"] = args.dataset_name
    config["isRegression"] = False
    config["lr"] = args.lr
    config["vocab_size"] = args.vocab_size
    config["batch_size"] = args.batch_size
    config["epochs"] = args.epochs
    config["max_seq_len"] = 100
    config["state_dict_dir"] = "../finetu_output/" + args.dataset_name + "/small_hidden_" + str(args.hidden_size) + \
                               "_layers_" + str(args.num_hidden_layers) + "_heads_" + str(args.num_attention_heads)+"/seq100"
    config["num_workers"] = 0
    config["hidden_size"] = args.hidden_size
    config["num_hidden_layers"] = args.num_hidden_layers
    config["num_attention_heads"] = args.num_attention_heads
    config["intermediate_size"] = config["vocab_size"]
    config["root_path"] = args.root_path
    config["cudaId"] = args.cudaId
    config["pretraining_model_path"] = args.pretraining_model_path + "/small_hidden_" + str(config["hidden_size"]) \
                                       + "_layers_" + str(config["num_hidden_layers"]) + "_heads_" + str(
        config["num_attention_heads"]) + "/seq100"
    config["train_path"] = config["root_path"] + "/" + config["dataset_name"] + "/"
    config["test_path"] = config["root_path"] + "/" + config["dataset_name"] + "/"
    config["validation_path"] = config["root_path"] + "/" + config["dataset_name"] + "/"
    config["pretraining_model_Id"] = args.model_Id

    print(config["state_dict_dir"], config["pretraining_model_path"])
    if not os.path.exists("../finetu_output/" + args.dataset_name):
        os.mkdir("../finetu_output/" + args.dataset_name)

    if config["dataset_name"] == 'hiv':
        # print("hiv---")
        config["work_nums"] = 1
    elif config["dataset_name"] == 'bbbp':
        config["work_nums"] = 1
    elif config["dataset_name"] == 'bace':
        config["work_nums"] = 1
    elif config["dataset_name"] == 'estrogen-alpha':
        config["work_nums"] = 1
    elif config["dataset_name"] == 'estrogen-beta':
        config["work_nums"] = 1
    elif config["dataset_name"] == 'mesta-high':
        config["work_nums"] = 1
    elif config["dataset_name"] == 'mesta-low':
        config["work_nums"] = 1
    elif config["dataset_name"]== "tox21":
        config["work_nums"] = 12

    elif config["dataset_name"] == "pcba":
        config["work_nums"] = 128
    elif config["dataset_name"] == "muv":
        config["work_nums"] = 17
    elif config["dataset_name"] == "toxcast":
        config["work_nums"] = 617
    elif config["dataset_name"] == "sider":
        config["work_nums"] = 27
    elif config["dataset_name"] == "clintox":
        config["work_nums"] = 2
    elif config["dataset_name"] == 'esol' or config["dataset_name"] == 'freesolv' or \
            config["dataset_name"] == 'lipophilicity':
        config["work_nums"] = 1
        config["isRegression"] = True
        if config["dataset_name"] == "esol":
            config["k"] = 0.7309941520467836*2
            # config["k"] = 0.7309941520467836 # 20
            config["Min"] = -11.6
        elif config["dataset_name"] == "freesolv":
            config["k"] = 1.2474112397767172*2
            config["Min"] = -5.635369496027946
        elif config["dataset_name"] == 'lipophilicity':
            config["k"] = 1.5384615384615385*2
            config["Min"] = -1.5
    else:
        raise ValueError("Invalid dataset name.")

    # 还需记录每次epoc测量值

    all_lr_auc = []  # 每次学习率对应的平均auc
    all_lr_loss = []
    all_lr_max_auc = []  # 每次学习率对应的最大auc
    all_lr_max_loss = []
    all_lr_min_auc = []  # 每次学习率对应的最小auc
    all_lr_min_loss = []
    all_lr_variance = []
    for lr in lr_list:
        config["lr"] = lr
        six_auc = []  # 记录每次迭代测试集的auc
        six_loss = []

        print("-----------lr = " + str(lr) + "------------");
        l = 5
        if config["dataset_name"] == "hiv":
            l = 3
        for iter in range(l):
            itr = iter
            if config["dataset_name"] == "hiv" or config["dataset_name"] == "bace":
                itr = "0"
            config["train_path"] = config["root_path"] + "/" + config["dataset_name"] + "/" + \
                                   "train_"+ str(itr+1) + ".txt"
            config["test_path"] = config["root_path"] + "/" + config["dataset_name"] + "/" + \
                                  "test_" + str(itr+1) + ".txt"
            config["validation_path"] = config["root_path"] + "/" + config["dataset_name"] + "/" \
                                        + "validation_" + str(itr+1) + ".txt"
            start_epoch = 0
            train_epoches = 9999
            trainer, dynamic_lr = init_trainer(dynamic_lr=lr, config=config, batch_size=config["batch_size"])
            all_auc = []  # 记录每次迭代所有的auc
            all_loss = []
            threshold = 0
            patient = 6  # 防止过拟合
            best_auc = 0
            best_loss = 100000000

            for epoch in range(start_epoch, start_epoch + train_epoches):
                if epoch >= 30:
                    break
                if epoch == start_epoch and epoch == 0:
                    # 第一个epoch的训练需要加载预训练的BERT模型
                    trainer.load_model(trainer.bert_model, dir_path=config["pretraining_model_path"], load_bert=True,
                                       specialModelId=config["pretraining_model_Id"])
                elif epoch == start_epoch:
                    trainer.load_model(trainer.bert_model, dir_path=trainer.config["state_dict_dir"], specialModelId=0)
                print("train with learning rate {}".format(str(dynamic_lr)))
                # 训练一个epoch

                print('epoc', epoch, 'start training !!!')
                auc = trainer.train(epoch)
                print("train_values: "+str(auc))
                # 保存当前epoch模型参数

                # 测试阶段无需保存模型
                # print('saving model !!!')
                # trainer.save_state_dict(trainer.bert_model, epoch,
                #                         state_dict_dir=trainer.config["state_dict_dir"],
                #                         file_path=".model")

                print('start validation !!!')
                auc = trainer.validation(epoch)


                if config["isRegression"] == True:
                    RMSE_loss = math.sqrt(auc)
                    print('epoc :', epoch, 'validation_RMSE_loss :', RMSE_loss)

                    all_loss.append(RMSE_loss)

                    # best_auc = max(all_auc)
                    if all_loss[-1] >= best_loss:
                        threshold += 1
                        dynamic_lr *= args.dr
                        trainer.init_optimizer(lr=dynamic_lr)
                    else:
                        # 如果
                        threshold = 0
                        best_loss = all_loss[-1]
                        trainer.save_state_dict(trainer.bert_model, 0,
                                                state_dict_dir=trainer.config["state_dict_dir"] + "/model_path/",
                                                file_path="bert.model")

                    if threshold >= patient:
                        print("epoch {} has the lowest loss".format(start_epoch + np.argmax(np.array(all_loss))))
                        print("early stop!")
                        print('start test !!!')
                        auc = trainer.test(epoch)
                        print("test", math.sqrt(auc))
                        break
                        print('epoc :', epoch, 'validation_RMSE_loss :', best_loss)
                    print('start test !!!')
                    auc = trainer.test(epoch)
                    print("test", math.sqrt(auc))
                else:
                    print('epoc :', epoch, 'validation_auc :', auc)
                    all_auc.append(auc)

                    # best_auc = max(all_auc)
                    if all_auc[-1] <= best_auc:
                        threshold += 1
                        dynamic_lr *= args.dr
                        trainer.init_optimizer(lr=dynamic_lr)
                    else:
                        # 如果
                        threshold = 0
                        # if best_auc >= max(all_auc):
                        best_auc = all_auc[-1]
                        trainer.save_state_dict(trainer.bert_model, 0,
                                                state_dict_dir=trainer.config["state_dict_dir"] + "/model_path/",
                                                file_path="bert.model")
                    if threshold >= patient:
                        print("epoch {} has the lowest loss".format(start_epoch + np.argmax(np.array(all_auc))))
                        print("early stop!")
                        print('start test !!!')
                        auc = trainer.test(epoch)
                        print("test", auc)
                        break
                    print('start test !!!')
                    auc = trainer.test(epoch)
                    print("test",auc)

            if config["isRegression"]:
                print("best_validation_loss " + str(iter) + str(":"), best_loss)
            else:
                print("best_validation_auc " + str(iter) + str(":"), best_auc)

            trainer.load_model(trainer.bert_model, dir_path=config["state_dict_dir"], load_bert=False, specialModelId=0)
            # model_name = config["state_dict_dir"] + "/.model.epoch." + str(0)
            # trainer, _ = init_trainer(dynamic_lr=lr, config=config, batch_size=config["batch_size"])
            # trainer.load_best_model(trainer.bert_model, model_name=model_name)

            if config["isRegression"]:
                six_loss.append(math.sqrt(trainer.validation(epoch)))
                print("test_loss",six_loss[-1])
            else:
                six_auc.append(trainer.validation(epoch))
                print("test_auc",six_auc[-1])

        if config["isRegression"]:
            all_lr_loss.append(sum(six_loss) / l)
            all_lr_max_loss.append(max(six_loss))
            all_lr_min_loss.append(min(six_loss))
            # 求方差
            average_loss = sum(six_loss) / l
            sum_loss = 0.0
            for i in six_loss:
                sum_loss += (i - average_loss) * (i - average_loss)
            all_lr_variance.append(sum_loss / l)
        else:
            all_lr_auc.append(sum(six_auc) / l)
            all_lr_max_auc.append(max(six_auc))
            all_lr_min_auc.append(min(six_auc))
            # 求方差
            average_auc = sum(six_auc) / l
            sum_auc = 0.0
            for i in six_auc:
                sum_auc += (i - average_auc) * (i - average_auc)
            all_lr_variance.append(sum_auc / l)



    if config["isRegression"]:
        print('best loss : ', max(all_lr_min_loss))
        with open(config["state_dict_dir"] + "/test_output_quick_mmm_final_r0_r1_seq100_5fold_" + str(args.dr) + "_" + \
            str(args.model_Id) + "_"+".txt", "w") as f:
            for i in range(len(lr_list)):
                lr = lr_list[i]
                f.write("lr: " + str(lr) + "," + "average loss: " + str(all_lr_loss[i]) + ",margin: " + str(
                    all_lr_max_loss[i] - all_lr_min_loss[i]) +",std: " + str(
                    math.sqrt(all_lr_variance[i])) + "\n")
    else:
        print('best auc : ', max(all_lr_max_auc))
        with open(config["state_dict_dir"] + "/test_output_quick_mmm_final_r0_r1_seq100_5fold_" + str(args.dr) + "_" + \
            str(args.model_Id) + "_" + ".txt", "w") as f:
            for i in range(len(lr_list)):
                lr = lr_list[i]
                f.write("lr: " + str(lr) + "," + "average auc: " + str(all_lr_auc[i]) + ",margin: " + str(
                    all_lr_max_auc[i] - all_lr_min_auc[i]) + ",std: "+
                        str(math.sqrt(all_lr_variance[i])) + "\n")


if __name__ == '__main__':
    main()
