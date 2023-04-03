import os
import torch
from torch.utils.data import Dataset, DataLoader
import jieba


class MyDataset(Dataset):
    def __init__(self, folder_path):
        self.file_list = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                self.file_list.append(os.path.join(root, file))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        with open(self.file_list[idx], "r", encoding="utf-8") as f:
            text = f.read().strip()
            words = jieba.lcut(text)
            inputs = []
            labels = []
            for word in words:
                if "/DS" in word:
                    inputs.extend(list(word[:-3]))
                    labels.extend(["O"] * (len(word) - 3) + ["B-TIME"])
                elif "/DO" in word:
                    inputs.extend(list(word[:-3]))
                    labels.extend(["O"] * (len(word) - 3) + ["I-TIME"])
                elif "/TS" in word:
                    inputs.extend(list(word[:-3]))
                    labels.extend(["O"] * (len(word) - 3) + ["B-TIME"])
                elif "/TO" in word:
                    inputs.extend(list(word[:-3]))
                    labels.extend(["O"] * (len(word) - 3) + ["I-TIME"])
                elif "/LOC" in word:
                    inputs.extend(list(word[:-4]))
                    labels.extend(["O"] * (len(word) - 4) + ["B-LOC"])
                else:
                    inputs.extend(list(word))
                    labels.extend(["O"] * len(word))
            inputs = [vocab[word] if word in vocab else vocab["<UNK>"] for word in inputs]
            return inputs, labels


def build_vocab(file_list, min_count=5):
    word_count = {}
    for file_path in file_list:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            words = jieba.lcut(text)
            for word in words:
                if word in word_count:
                    word_count[word] += 1
                else:
                    word_count[word] = 1
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, count in word_count.items():
        if count >= min_count:
            vocab[word] = len(vocab)
    return vocab


# 指定数据集路径
folder_path = "实验2语料"

# 构建词表
file_list = []
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_list.append(os.path.join(root, file))
vocab = build_vocab(file_list)

# 构建数据集迭代器
dataset = MyDataset(folder_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

import torch
import torch.nn as nn
from transformers import BertModel


class BertForSequenceTagging(nn.Module):

    def __init__(self, num_labels):
        super().__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_fct(active_logits, active_labels)

        if labels is not None:
            return loss
        else:
            return logits

model = BertForSequenceTagging(num_labels=5)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
metric = nn.CrossEntropyLoss(ignore_index=-100)
