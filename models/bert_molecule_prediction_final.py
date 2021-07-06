from torch import nn
from models.bert_model_v2 import *
import numpy as np
"""使用mean max pool"""

class Bert_Smiles_Analysis(nn.Module):
    def __init__(self, config, deviceId=2, isRegression=False, base_config=None):
        super(Bert_Smiles_Analysis, self).__init__()
        self.bert = BertModel(config)
        # if config.work_nums == 1:
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.dense1 = nn.Linear(config.hidden_size, 100)
        self.final_dense = nn.Linear(100, config.work_nums)
        # self.final_dense = nn.Linear(config.hidden_size, config.work_nums)

        self.work_nums = config.work_nums
        self.deviceId = deviceId
        self.isRegression = isRegression
          # work_nums 分类的数目
        # self.final_dense = nn.Linear(config.hidden_size, config.work_nums)  # work_nums 分类的数目
        if self.isRegression == False:
            self.activation_final = nn.Sigmoid()
            self.activation = nn.ReLU()

        else: # 是回归任务
            # 外接一个线性回归
            self.loss_fn = nn.MSELoss()
            self.activation = nn.ReLU()
            self.k = base_config["k"]
            self.Min = base_config["Min"]

    def compute_loss(self, predictions, labels):
        # 将预测和标记的维度展平, 防止出现维度不一致
        if predictions.shape[1] == 1:
            # criterion = nn.BCEWithLogitsLoss()
            # predictions = predictions.view(-1)
            # labels = labels.float().view(-1)
            import numpy as np
            predictions = predictions.reshape((predictions.shape[0], -1))
            labels = labels.reshape((predictions.shape[0], -1))
            # epsilon = 1e-8
            torch.cuda.set_device(device=self.deviceId)
            # loss = -labels * torch.log(predictions + epsilon).cuda() - (torch.tensor(1.0) - labels).cuda() * \
            #        torch.log(torch.tensor(1.0) - predictions + epsilon).cuda()
            pos_weight = [2 for i in range(self.work_nums)]
            pos_weight = np.array(pos_weight)
            pos_weight = torch.from_numpy(pos_weight).float().cuda()
            loss = torch.nn.functional.binary_cross_entropy(predictions,labels.float().cuda(), weight=pos_weight.float().cuda())
            # loss = criterion(labels.float().reshape([-1,1]), predictions)
            loss = torch.mean(loss) # scalar值才可以反向传播

        # # 采用交叉熵统计损失函数
        else:
            import numpy as np
            torch.cuda.set_device(device=self.deviceId)
            pos_weight = [2 for i in range(predictions.shape[1])]
            pos_weight = np.array(pos_weight)
            pos_weight = torch.from_numpy(pos_weight).double().cuda()
            # pos_weight = torch.ones([27])
            criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
            # criterion = nn.BCEWithLogitsLoss(reduction="none")

            y = labels.view(predictions.shape).to(torch.float64)
            # Whether y is non-null or not.
            is_valid = y ** 2 > 0
            # Loss matrix
            loss_mat = criterion(predictions.double(), (y + 1) / 2)
            # loss matrix after removing null target
            loss_mat = torch.where(is_valid, loss_mat,
                                   torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat) / torch.sum(is_valid)

        return loss



    def forward(self, text_input, positional_enc, labels=None, text_input1=None, fp = None):
        # print(text_input.shape,labels.shape)
        encoded_layers, _ = self.bert(text_input, positional_enc,
                                      output_all_encoded_layers=True)
        sequence_output = encoded_layers[-1]  # 为了避免过拟合
        # # sequence_output的维度是[batch_size, seq_len, embed_dim]
        # print(sequence_output.shape)
        avg_pooled = sequence_output.mean(1) # [batch_size, embed_dim]
        max_pooled = torch.max(sequence_output, dim=1)  # [batch_size, embed_dim]
        pooled = torch.cat((avg_pooled, max_pooled[0]), dim=1)  # [batch_size, embed_dim * 2]
        #pooled = torch.cat((pooled,text_input1),dim=1)
        #pooled = torch.cat((pooled, fp), dim=1)
        pooled = self.dense(pooled)
        pooled = self.dense1(pooled)


        # 下面是[batch_size, hidden_dim * 2] 到 [batch_size, 1]的映射
        if self.isRegression == True:

            predictions = self.final_dense(pooled)
            # predictions = self.activation(predictions)
            # print(predictions.shape,labels.shape)
            torch.cuda.set_device(device=self.deviceId)
            # k = float(self.k)
            # Min = float(self.Min)
            # loss = self.loss_fn((predictions / k + Min).cuda(), (labels.view(-1, 1) / k + Min).cuda())
            loss = self.loss_fn(predictions.cuda(),labels.view(-1,1).cuda())
            loss = torch.mean(loss)
            return predictions, loss
        else:

            predictions = self.final_dense(pooled)
            if self.work_nums == 1:
                # pooled = self.dense(pooled)
                # predictions = self.final_dense(pooled)
                predictions = self.activation_final(predictions)

            if labels is not None:
                # 计算loss
                # self.activation = nn.ReLU()
                # predictions = self.activation(predictions)
                loss = self.compute_loss(predictions, labels)
                return predictions, loss
            else:
                return predictions
