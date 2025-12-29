from torch import nn
from package.nn import ConditionalRandomField
from transformers import BertModel, PreTrainedTokenizerFast, BertConfig
import torch

class SciBert_FastText_BiLSTM_CRF(nn.Module):

    def __init__(self, path_to_bert, tag_set_size,  bert_lstm_hidden_dim=768, w2v_lstm_hidden_dim=100, lstm_dropout_rate=0.1, freeze_bert=True):
        super(SciBert_FastText_BiLSTM_CRF, self).__init__()
        model1 = BertModel.from_pretrained("/root/autodl-tmp/NER-SciBERT/resource/MaterialsBert")
        #model2 = BertModel.from_pretrained("/root/autodl-tmp/NER-SciBERT/resource/MaterialsBert")
        if freeze_bert:
            for param in model1.parameters():
                param.requires_grad = False
            # for param in model2.parameters():
            #     param.requires_grad = False

        self.bert1 = model1
        #self.bert2 = model2

        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(path_to_bert)
        # self.bert_biLstm = nn.LSTM(bert_lstm_hidden_dim + w2v_lstm_hidden_dim,
        #                            (bert_lstm_hidden_dim + w2v_lstm_hidden_dim) // 2,
        #                            num_layers=2,
        #                            bidirectional=True,
        #                            dropout=lstm_dropout_rate,
        #                            batch_first=True)
        #self.bert_hidden2tag = nn.Linear(bert_lstm_hidden_dim + w2v_lstm_hidden_dim, tag_set_size)
        self.bert_biLstm = nn.LSTM(bert_lstm_hidden_dim,
                                   bert_lstm_hidden_dim // 2,
                                   num_layers=2,
                                   bidirectional=True,
                                   dropout=lstm_dropout_rate,
                                   batch_first=True)
        self.bert_hidden2tag = nn.Linear(bert_lstm_hidden_dim, tag_set_size)

        # transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=tag_set_size, nhead=1)
        # self.transformer_decoder = nn.TransformerDecoder(transformer_decoder_layer, num_layers=2)


        self.crf = ConditionalRandomField(tag_set_size)

    def reset_parameters(self):
        self.crf.reset_parameters()


    def forward(self, batch_size, bert_input: torch.LongTensor, bert_mask: torch.ByteTensor, bert_target: torch.LongTensor,
                 mappings: torch.IntTensor):

        bert_x1 = self.bert1(input_ids=bert_input, attention_mask=bert_mask)[0]
        #bert_x2 = self.bert2(input_ids=bert_input, attention_mask=bert_mask)[0]
        #bert_x = torch.cat((bert_x1, bert_x2), dim=-1)
        # bert_w2v_emb = torch.zeros(bert_x1.shape[0], bert_x1.shape[1], bert_x1.shape[2] + w2v_input.shape[2])
        bert_w2v_emb = torch.zeros(bert_x1.shape[0], bert_x1.shape[1], bert_x1.shape[2])
        for batch_num in range(min(batch_size, bert_input.shape[0])):
            single_bert_x1 = bert_x1[batch_num]
            #single_bert_x2 = bert_x2[batch_num]
            # single_w2v_x = w2v_input[batch_num]
            single_mappings = mappings[batch_num]
            # for i, mapping in enumerate(single_mappings):
            #     if mapping == -1:
            #         break
            #     else:
            #         if i < 509:
            #             bert_w2v_emb[batch_num][i + 1] = torch.cat(
            #                 #(single_bert_x1[i + 1], single_bert_x2[i + 1], single_w2v_x[mapping]), dim=0)
            #                 (single_bert_x1[i + 1], single_w2v_x[mapping]), dim=0)

        bert_w2v_emb = bert_w2v_emb.to('cuda')
        bert_w2v_emb, _ = self.bert_biLstm(bert_w2v_emb)
        bert_w2v_emb = self.bert_hidden2tag(bert_w2v_emb)

        #bert_w2v_emt = self.transformer_decoder(bert_w2v_emb, bert_w2v_emb)

        return self.crf(bert_w2v_emb, bert_mask)

    def loss(self, batch_size, bert_input: torch.LongTensor, bert_mask: torch.ByteTensor, bert_target: torch.LongTensor, mappings: torch.IntTensor):
        bert_x1 = self.bert1(input_ids=bert_input, attention_mask=bert_mask)[0]
        #bert_x2 = self.bert2(input_ids=bert_input, attention_mask=bert_mask)[0]
        #bert_x = torch.cat((bert_x1, bert_x2), dim=-1)
        #bert_w2v_emb = torch.zeros(bert_x1.shape[0], bert_x1.shape[1], bert_x1.shape[2] + w2v_input.shape[2])
        bert_w2v_emb = torch.zeros(bert_x1.shape[0], bert_x1.shape[1], bert_x1.shape[2])
        #bert_w2v_emb = torch.zeros(bert_x.shape[0], bert_x.shape[1], bert_x.shape[2] + w2v_input.shape[2])
        for batch_num in range(batch_size):
            single_bert_x1 = bert_x1[batch_num]
            #single_bert_x2 = bert_x2[batch_num]
            #single_w2v_x = w2v_input[batch_num]
            single_mappings = mappings[batch_num]
            # for i, mapping in enumerate(single_mappings):
            #     if mapping == -1:
            #         break
            #     else:
            #         if i < 509:
            #             #bert_w2v_emb[batch_num][i+1] = torch.cat((single_bert_x1[i+1],single_bert_x2[i+1], single_w2v_x[mapping]), dim=0)
            #             bert_w2v_emb[batch_num][i + 1] = torch.cat(
            #                 (single_bert_x1[i + 1], single_w2v_x[mapping]), dim=0)
        bert_w2v_emb = bert_w2v_emb.to('cuda')
        bert_w2v_emb, _ = self.bert_biLstm(bert_w2v_emb)
        bert_w2v_emb = self.bert_hidden2tag(bert_w2v_emb)

        #bert_w2v_emb = self.transformer_decoder(bert_w2v_emb, bert_w2v_emb)

        return self.crf.neg_log_likelihood_loss(bert_w2v_emb, bert_mask, bert_target)
