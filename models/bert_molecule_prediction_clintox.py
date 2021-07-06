from torch import nn
from models.bert_model_v2 import *
import numpy as np
"""使用mean max pool"""

class Bert_Smiles_Analysis(nn.Module):
    def __init__(self, config, deviceId=2, isRegression=False):
        super(Bert_Smiles_Analysis, self).__init__()
        self.bert = BertModel(config)
        self.dense = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.dense1 = nn.Linear(config.hidden_size, 100)
        self.final_dense1 = nn.Linear(100, 1)  # work_nums 分类的数目
        self.final_dense2 = nn.Linear(100, 1)  # work_nums 分类的数目
        self.activation = nn.Sigmoid()
        self.work_nums = config.work_nums
        self.deviceId = deviceId
        self.isRegression = isRegression
        self.class_nums = config.work_nums


    def compute_loss(self, predictions, labels):
        # 将预测和标记的维度展平, 防止出现维度不一致
        if predictions.shape[1] == 1:
            torch.cuda.set_device(device=self.deviceId)
            predictions = predictions.view(-1)
            y = labels.float().view(-1)
            epsilon = 1e-8

            # 交叉熵
            loss = -(y+1)/2 * torch.log(predictions + epsilon).cuda() - (torch.tensor(1.0) - ((y+1)/2)).cuda() * \
                   torch.log(torch.tensor(1.0) - predictions + epsilon).cuda()

            # loss matrix after removing null target
            is_valid = y ** 2 > 0
            loss = torch.where(is_valid, loss, torch.zeros(loss.shape).to(loss.device).to(loss.dtype))
            loss = torch.mean(loss) # scalar值才可以反向传播

            # predictions = predictions.reshape((predictions.shape[0], -1))
            # labels = labels.reshape((predictions.shape[0], -1))
            # # epsilon = 1e-8
            # torch.cuda.set_device(device=self.deviceId)
            # # loss = -labels * torch.log(predictions + epsilon).cuda() - (torch.tensor(1.0) - labels).cuda() * \
            # #        torch.log(torch.tensor(1.0) - predictions + epsilon).cuda()
            # pos_weight = [2 for i in range(1)]
            # pos_weight = np.array(pos_weight)
            # pos_weight = torch.from_numpy(pos_weight).float().cuda()
            # # print(pos_weight.shape, predictions.shape)
            # # loss = torch.nn.functional.binary_cross_entropy(predictions, labels.float().cuda(),
            # #                                                 weight=pos_weight.float().cuda())
            # # loss = criterion(labels.float().reshape([-1,1]), predictions)
            # loss = torch.mean(loss)  # scalar值才可以反向传播
        return loss

    def forward(self, text_input, positional_enc, labels=None,text_input1=None, fp = None):
        # print(text_input.shape,labels.shape)
        encoded_layers, _ = self.bert(text_input, positional_enc,
                                      output_all_encoded_layers=True)
        sequence_output = encoded_layers[-1]  # 为了避免过拟合
        # # sequence_output的维度是[batch_size, seq_len, embed_dim]
        avg_pooled = sequence_output.mean(1)  # [batch_size, embed_dim]
        max_pooled = torch.max(sequence_output, dim=1)  # [batch_size, embed_dim]
        pooled = torch.cat((avg_pooled, max_pooled[0]), dim=1)  # [batch_size, embed_dim * 2]
        pooled = torch.cat((pooled, text_input1), dim=1)
        pooled = torch.cat((pooled, fp), dim=1)
        pooled = self.dense(pooled)  # [batch_size, embed]
        pooled = self.dense1(pooled)
        # 我们在这里要解决的是多分类问题
        # predictions = self.final_dense(pooled)
        # print(pooled,".......")
        pred1 = self.activation(self.final_dense1(pooled))
        pred2 = self.activation(self.final_dense2(pooled))

        loss1 = self.compute_loss(pred1, labels[:, 0])
        loss2 = self.compute_loss(pred2, labels[:, 1])

        pred = [pred1, pred2,]
        loss = [loss1, loss2,]


        if labels is not None:
            # 计算loss
            # loss = self.compute_loss(predictions, labels)
            return pred, loss
        else:
            return pred