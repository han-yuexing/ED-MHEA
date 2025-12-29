from torch.utils.data import DataLoader
import gensim
from package.nn import ConditionalRandomField
from transformers import BertModel, PreTrainedTokenizerFast, BertConfig
import torch
import csv
import json
import time
import torch
from torch.utils.data import DataLoader
from package.model import SciBert_FastText_BiLSTM_CRF
from torch.optim import Adam
from package.dataset import SLSDataset, id2tag, decode_tags_from_ids


def process_sample(sample_tokens, sample_predictions, sample_mapping):
    word_predictions = {}
    for token, pred, map_index in zip(sample_tokens, sample_predictions, sample_mapping):
        if map_index == -1:  # 忽略-1的mapping，因为它代表特殊token，例如[CLS]、[SEP]
            continue
        if map_index not in word_predictions:
            word_predictions[map_index] = {"tokens": [], "preds": []}
        word_predictions[map_index]["tokens"].append(token)
        word_predictions[map_index]["preds"].append(pred)

    # 合并tokens并选择最佳预测
    original_words = []
    predictions_for_words = []
    for index, info in sorted(word_predictions.items()):
        original_word = "".join([t.replace("##", "") for t in info["tokens"]])
        word_pred = id2tag[info["preds"][0]]
        original_words.append(original_word)
        predictions_for_words.append(word_pred)
    return original_words, predictions_for_words

def extract_entities(original_words, predictions):
    entities = []
    current_entity = []
    current_type = None

    for word, pred in zip(original_words, predictions):
        if pred.startswith("B-"):
            if current_entity:
                entities.append((" ".join(current_entity)))
            current_entity = [word]
            current_type = pred[2:]  # Remove the "B-" prefix
        elif pred.startswith("I-") and current_type == pred[2:]:
            current_entity.append(word)
        else:
            if current_entity:
                entities.append((" ".join(current_entity)))
                current_entity = []
                current_type = None

    # Catch any remaining entity
    if current_entity:
        entities.append((" ".join(current_entity)))

    return entities


# 推理用的文本数据


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
bert_path = "/root/autodl-tmp/NER-SciBERT/resource/matscibert"
lstm_dropout_rate = 0.4
lr = 2e-3
batch_size =32
model_for_inference = SciBert_FastText_BiLSTM_CRF(bert_path, len(id2tag), bert_lstm_hidden_dim=1536, w2v_lstm_hidden_dim=100, lstm_dropout_rate=lstm_dropout_rate).to(device)
model_for_inference.load_state_dict(torch.load('/root/autodl-tmp/NER-SciBERT/logs-MaSciBERT--MaterialsBert-FastText-ep200-bs32model_weights.pth'))
# 假设你已经有tokenizer和fasttext模型
#tokenizer = PreTrainedTokenizerFast.from_pretrained("/root/autodl-tmp/NER-SciBERT/resource/matscibert")  # 你的tokenizer实例

fasttext = "fasttext/fasttext_embeddings-MINIFIED.model"  # 你的fasttext模型实例
optimizer = Adam(filter(lambda p: p.requires_grad, model_for_inference.parameters()), lr=lr)
# 创建DataLoader

dataset = SLSDataset('sls/ner_passage_HEA.json', model_for_inference.tokenizer, fasttext)

dataloader_inference = DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=32, shuffle=False, drop_last=False)




model_for_inference.eval()

with torch.no_grad():
    for bert_inputs, bert_masks, bert_labels, w2v_inputs, w2v_masks, w2v_labels, mappings in dataloader_inference:
        bert_inputs, bert_masks, bert_labels, w2v_inputs, w2v_masks, w2v_labels, mappings = [_.to(device) for _ in (
        bert_inputs, bert_masks, bert_labels, w2v_inputs, w2v_masks, w2v_labels, mappings)]
        # 这里使用假设的batch_size，应与实际DataLoader中的batch_size一致
        predictions = model_for_inference(batch_size,bert_inputs, bert_masks, bert_labels, w2v_inputs, w2v_masks, w2v_labels, mappings)
        # 假设`predictions`是模型输出的预测标签序列
        # 假设`token_ids`是输入模型的token IDs序列
        # `tokenizer`是用于将token IDs转换回tokens的tokenizer
        # `label_map`是将标签ID映射到B-I-O标记的字典

        # 假设 bert_inputs 是一个包含多个样本的batch，其中每个样本都是一个token ID的列表
        #tokens = [model_for_inference.tokenizer.convert_ids_to_tokens(ids) for ids in bert_inputs]
        # 假设 bert_inputs 是一个包含多个样本的batch，其中每个样本都是一个token ID的列表
        # PAD_token_id 是用于填充的特殊token的ID，通常是0
        PAD_token_id = model_for_inference.tokenizer.pad_token_id

        # 假设 bert_inputs 是一个包含多个样本的batch，其中每个样本都是一个token ID的列表
        # 假设 PAD_token_id 是用于填充的特殊token的ID
        # 对于batch中的每个样本，转换token IDs到tokens，停止在PAD token
        tokens = []
        token_mapps=[]# 用于存储所有样本的tokens
        PAD_token_id = model_for_inference.tokenizer.pad_token_id

        # 逐个样本处理
        for sample in bert_inputs:
            token = []  # 用于存储当前样本的tokens
            for token_id in sample:
                # 检查是否为PAD token
                if token_id.item() == PAD_token_id:
                    break  # 遇到PAD token，停止处理当前样本
                # 将token ID转换为token并添加到当前样本的token列表中
                token.append(model_for_inference.tokenizer.convert_ids_to_tokens(token_id.item()))
            # 将当前样本的token列表添加到tokens中
            tokens.append(token)

        # tokens_batch 现在包含了每个样本转换后的tokens，PAD之后的tokens被忽略了

        for map in mappings:
            mapps = [-1]  # 表示cls
            for map_id in map:
                # 检查是否为PAD token
                if map_id.item() == -1:
                    break  # 遇到PAD token，停止处理当前样本
                # 将token ID转换为token并添加到当前样本的token列表中
                mapps.append(map_id.item())
            # 将当前样本的token列表添加到tokens中
            mapps.append(-2)  #表示sep
            token_mapps.append(mapps)



        for sample_tokens, sample_predictions, sample_mapping in zip(tokens, predictions, token_mapps):
            original_words, predictions = process_sample(sample_tokens, sample_predictions, sample_mapping)
            all_entities = []

            with open('output.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csvwriter = csv.writer(csvfile)
                # 合并每个样本的实体
                all_entities = extract_entities(original_words, predictions)
                # 检查并保存最后一个实体
                if all_entities:
                    csvwriter.writerow([all_entities])
                    all_entities = []



