"""
@Author: yanzx
@Date: 2023/7/13 11:44
@Description:
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

train_data = [
    ('entity1', 'relation1', 'entity2'),
    ('entity3', 'relation2', 'entity4'),
    ('entity3', 'relation2', 'entity5'),
]

entity_dict = {'entity1': 0, 'entity2': 1, 'entity3': 2, 'entity4': 3, "entity5": 4}
relation_dict = {'relation1': 0, 'relation2': 1}

embedding_dim = 50
num_entities = 5
num_relations = 2
margin = 1.0


class TransH(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, margin):
        super(TransH, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.margin = margin

        # 实体和关系的嵌入向量
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # 实体和关系的法向量
        self.entity_normal_vectors = nn.Embedding(num_entities, embedding_dim)
        self.relation_normal_vectors = nn.Embedding(num_relations, embedding_dim)

        # 初始化参数
        nn.init.xavier_uniform_(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform_(self.relation_embeddings.weight.data)
        nn.init.xavier_uniform_(self.entity_normal_vectors.weight.data)
        nn.init.xavier_uniform_(self.relation_normal_vectors.weight.data)

    def forward(self, head_entities, relations, tail_entities):
        # 获取实体和关系的嵌入向量
        head_embeddings = self.entity_embeddings(head_entities)
        relation_embeddings = self.relation_embeddings(relations)
        tail_embeddings = self.entity_embeddings(tail_entities)

        # 获取实体和关系的法向量
        head_normal_vectors = self.entity_normal_vectors(head_entities)
        relation_normal_vectors = self.relation_normal_vectors(relations)
        tail_normal_vectors = self.entity_normal_vectors(tail_entities)

        # TransH的投影操作
        self.head_proj = head_embeddings - torch.sum(
            head_embeddings * head_normal_vectors, dim=-1, keepdim=True
        ) * head_normal_vectors
        self.tail_proj = tail_embeddings - torch.sum(
            tail_embeddings * tail_normal_vectors, dim=-1, keepdim=True
        ) * tail_normal_vectors
        self.relation_proj = relation_embeddings - torch.sum(
            relation_embeddings * relation_normal_vectors, dim=-1, keepdim=True
        ) * relation_normal_vectors

        # 计算头尾实体之间的关系得分
        scores = torch.norm(self.head_proj + self.relation_proj - self.tail_proj, p=2, dim=-1)

        return scores

    def predict(self, head_entities, relations, tail_entities):
        """
        模型训练完成后进行预测
        :param head_entities:
        :param relations:
        :param tail_entities:
        :return:
        """
        self.forward(head_entities, relations, tail_entities)
        return self.head_proj, self.relation_proj, self.tail_proj

    def recommend_tail(self, head_entity_id, relation_id, top_k=5):
        # 获取head和relation的嵌入向量
        head_embedding = self.entity_embeddings(head_entity_id)
        relation_embedding = self.relation_embeddings(relation_id)

        # 获取所有tail实体的嵌入向量
        all_tail_embeddings = self.entity_embeddings.weight

        # 计算关系转移向量
        relation_transfer_vector = relation_embedding - torch.sum(
            relation_embedding * self.relation_normal_vectors(relation_id), dim=-1,
            keepdim=True) * self.relation_normal_vectors(relation_id)

        # 计算预测的tail向量
        predicted_tail_vector = head_embedding + relation_transfer_vector

        # 计算与所有tail实体的余弦相似度
        cosine_similarities = torch.nn.functional.cosine_similarity(predicted_tail_vector, all_tail_embeddings)

        # 找到与预测向量最相似的top_k个tail实体
        _, top_k_indices = torch.topk(cosine_similarities, top_k)
        recommended_tail_entity_ids = top_k_indices.tolist()

        return recommended_tail_entity_ids


class TransHDataset(Dataset):
    def __init__(self, train_data):
        self.train_data = train_data

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, index):
        head_entity, relation, tail_entity = self.train_data[index]
        return torch.tensor(head_entity), torch.tensor(relation), torch.tensor(tail_entity)


def train():
    """
    模型入口函数
    :return:
    """
    model = TransH(len(entity_dict), len(relation_dict), 50, 1.0)

    criterion = nn.MarginRankingLoss(margin=1.0, reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    train_dataset = TransHDataset([(entity_dict[h], relation_dict[r], entity_dict[t]) for h, r, t in train_data])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    num_epochs = 10
    for epoch in range(num_epochs):
        for batch_idx, (head_entities, relations, tail_entities) in enumerate(train_loader):
            # 清零梯度
            optimizer.zero_grad()

            # 正例样本
            pos_scores = model(head_entities, relations, tail_entities)

            # 随机生成负例样本
            neg_head_entities = torch.randint(0, len(entity_dict), size=head_entities.shape)
            neg_tail_entities = torch.randint(0, len(entity_dict), size=tail_entities.shape)
            neg_scores = model(neg_head_entities, relations, neg_tail_entities)

            # 计算损失
            targets = torch.ones_like(pos_scores)
            loss = criterion(pos_scores, neg_scores, targets)

            # 反向传播和参数更新
            loss.backward()
            optimizer.step()

            # 打印损失
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item()}")

    return model


def save_embedding(model: TransH):
    """
    保存模型训练好的entity embedding和 relation embedding
    :param model:
    :return:
    """
    # 获取实体和关系的嵌入向量
    entity_embeddings = model.entity_embeddings.weight.data.numpy()
    relation_embeddings = model.relation_embeddings.weight.data.numpy()

    # 假设存在实体和关系的标识符列表
    entity_ids = list(entity_dict.values())
    relation_ids = list(relation_dict.values())

    # 创建包含实体嵌入向量的DataFrame
    entity_df = pd.DataFrame(data=entity_embeddings, index=entity_ids)

    # 创建包含关系嵌入向量的DataFrame
    relation_df = pd.DataFrame(data=relation_embeddings, index=relation_ids)

    # 保存实体和关系的嵌入向量为CSV文件
    entity_df.to_csv('data/entity_embeddings.csv')
    relation_df.to_csv('data/relation_embeddings.csv')


def load_model(path: str):
    """
    加载模型
    :param path:
    :return:
    """
    # 加载已训练好的模型参数
    model = TransH(num_entities, num_relations, embedding_dim, margin)
    model.load_state_dict(torch.load(path))

    return model


def extend_model(model: TransH, num_entities: int, num_new_entities: int):
    """
    扩展模型，当index小于原来时，需要进行扩展
    :param model:
    :param num_entities:
    :param num_new_entities:
    :return:
    """
    # 扩展模型的实体嵌入矩阵
    extended_entity_embeddings = nn.Embedding(num_entities + num_new_entities, embedding_dim)
    extended_relation_embeddings = nn.Embedding(num_relations, embedding_dim)
    extended_entity_embeddings.weight.data[:num_entities] = model.entity_embeddings.weight.data
    extended_relation_embeddings.weight.data = model.relation_embeddings.weight.data

    # 创建扩展后的模型实例
    extended_model = TransH(num_entities + num_new_entities, num_relations, embedding_dim, margin)
    extended_model.entity_embeddings = extended_entity_embeddings
    extended_model.relation_embeddings = extended_relation_embeddings

    # 加载训练好的模型参数到扩展后的模型
    loaded_state_dict = torch.load('model/transH.pth')
    extended_state_dict = extended_model.state_dict()

    for name, param in loaded_state_dict.items():
        if name in extended_state_dict:
            if param.shape == extended_state_dict[name].shape:
                extended_state_dict[name] = param

    extended_model.load_state_dict(extended_state_dict)
    return extended_model


def main():
    model = train()
    save_embedding(model)
    torch.save(model.state_dict(), "model/transH.pth")  # 保存torch
    model = load_model("model/transH.pth")

    loaded_model = extend_model(model, num_entities, 100)

    # 构造输入数据（新的三元组）
    new_head_entity = torch.tensor([100])
    new_relation = torch.tensor([1])
    new_tail_entity = torch.tensor([2])

    # 获取实体和关系的嵌入向量
    with torch.no_grad():
        head_embeddings, relation_embeddings, tail_embeddings = loaded_model.predict(new_head_entity, new_relation,
                                                                                     new_tail_entity)

    # 打印实体和关系的嵌入向量
    print("Head Entity Embedding:", head_embeddings)
    print("Relation Embedding:", relation_embeddings)
    print("Tail Entity Embedding:", tail_embeddings)

    res = model.recommend_tail(torch.tensor(1), torch.tensor(0))
    print(res)


if __name__ == '__main__':
    main()
