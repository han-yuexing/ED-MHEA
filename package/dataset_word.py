# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from package.utils import read_train_json
import numpy as np

# tag2id = {'O': 0,
#           'B-Preparation': 1, 'I-Preparation': 2,
#           'B-HV': 3, 'I-HV': 4,
#           'B-ST': 5, 'I-ST': 6,
#           'B-Name': 7, 'I-Name': 8,
#           }
#
# id2tag = {0: 'O',
#           1: 'B-Preparation', 2: 'I-Preparation',
#           3: 'B-HV', 4: 'I-HV',
#           5: 'B-ST', 6: 'I-ST',
#           7: 'B-Name', 8: 'I-Name',
#          }



tag2id = {'O': 0,
          'B-Preparation': 1, 'I-Preparation': 2,
          'B-Temperature': 3, 'I-Temperature': 4,
          'B-HV': 5, 'I-HV': 6,
          'B-UTS': 7, 'I-UTS': 8,
          'B-TE': 9, 'I-TE': 10,
          'B-FT': 11, 'I-FT': 12,
          'B-Element': 13, 'I-Element': 14,
          'B-Name': 15, 'I-Name': 16,
          }

id2tag = {0: 'O',
          1: 'B-Preparation', 2: 'I-Preparation',
          3: 'B-Temperature', 4: 'I-Temperature',
          5: 'B-HV', 6: 'I-HV',
          7: 'B-UTS', 8: 'I-UTS',
          9: 'B-TE', 10: 'I-TE',
          11: 'B-FT', 12: 'I-FT',
          13: 'B-Element', 14: 'I-Element',
          15: 'B-Name', 16: 'I-Name',
         }

#

#
# tag2id = {'O': 0,
#           'B-Preparation': 1, 'I-Preparation': 2,
#           'B-HV': 3, 'I-HV': 4,
#           'B-UTS': 5, 'I-UTS': 6,
#           'B-YS': 7, 'I-YS': 8,
#           'B-TE': 9, 'I-TE': 10,
#           'B-FT': 11, 'I-FT': 12,
#           'B-Temperature': 13, 'I-Temperature': 14,
#           'B-Name': 15, 'I-Name': 16,
#           'B-Property': 17, 'I-Property': 18,
#           'B-Element': 19, 'I-Element': 20
#           }
#
# id2tag = {0: 'O',
#           1: 'B-Preparation', 2: 'I-Preparation',
#           3: 'B-HV', 4: 'I-HV',
#           5: 'B-UTS', 6: 'I-UTS',
#           7: 'B-YS', 8: 'I-YS',
#           9: 'B-TE', 10: 'I-TE',
#           11: 'B-FT', 12: 'I-FT',
#           13: 'B-Temperature', 14: 'I-Temperature',
#           15: 'B-Name', 16: 'I-Name',
#           17: 'B-Property', 18: 'I-Property',
#           19: 'B-Element', 20: 'I-Element',
#          }


#
# tag2id = {'O': 0,
#           'B-CMT': 1, 'I-CMT': 2,
#           'B-MAT': 3, 'I-MAT': 4,
#           'B-DSC': 5, 'I-DSC': 6,
#           'B-PRO': 7, 'I-PRO': 8,
#           'B-SMT': 9, 'I-SMT': 10,
#           'B-APL': 11, 'I-APL': 12,
#           'B-SPL': 13, 'I-SPL': 14,
#           }
#
# id2tag = {0: 'O',
#           1: 'B-CMT', 2: 'I-CMT',
#           3: 'B-MAT', 4: 'I-MAT',
#           5: 'B-DSC', 6: 'I-DSC',
#           7: 'B-PRO', 8: 'I-PRO',
#           9: 'B-SMT', 10: 'I-SMT',
#           11: 'B-APL', 12: 'I-APL',
#           13: 'B-SPL', 14: 'I-SPL',
#          }
#
#





# tag2id = {'O': 0,
#           'B-Material Name': 1, 'I-Material Name': 2,
#           'B-Research Aspect': 3, 'I-Research Aspect': 4,
#           'B-Technology': 5, 'I-Technology': 6,
#           'B-Method': 7, 'I-Method': 8,
#           'B-Performance': 9, 'I-Performance': 10,
#           'B-Performance Values': 11, 'I-Performance Values': 12,
#           'B-Experiment Name': 13, 'I-Experiment Name': 14,
#           'B-Experimental Conditions': 15, 'I-Experimental Conditions': 16,
#           'B-Condition Value': 17, 'I-Condition Value': 18,
#           'B-Equipment Used': 19, 'I-Equipment Used': 20,
#           'B-Adding Elements': 21, 'I-Adding Elements': 22,
#           'B-Application': 23, 'I-Application': 24,
#           'B-Experiment Output': 25, 'I-Experiment Output': 26,}
#
# id2tag = {0: 'O',
#           1: 'B-Material Name', 2: 'I-Material Name',
#           3: 'B-Research Aspect', 4: 'I-Research Aspect',
#           5: 'B-Technology', 6: 'I-Technology',
#           7: 'B-Method', 8: 'I-Method',
#           9: 'B-Performance', 10: 'I-Performance',
#           11: 'B-Performance Values', 12: 'I-Performance Values',
#           13: 'B-Experiment Name', 14: 'I-Experiment Name',
#           15: 'B-Experimental Conditions', 16: 'I-Experimental Conditions',
#           17: 'B-Condition Value', 18: 'I-Condition Value',
#           19: 'B-Equipment Used', 20: 'I-Equipment Used',
#           21: 'B-Adding Elements', 22: 'I-Adding Elements',
#           23: 'B-Application', 24: 'I-Application',
#           25: 'B-Experiment Output', 26: 'I-Experiment Output'}
#
#
#



def decode_tags_from_ids(batch_ids):
    batch_tags = []
    for ids in batch_ids:
        sequence_tags = []
        for id in ids:
            sequence_tags.append(id2tag[int(id)])
        batch_tags.append(sequence_tags)
    return batch_tags



def multi_char_replace_word(model, word):
    word_emb = np.zeros(100, dtype='float32')
    size = len(word)
    for i in range(size):
        try:
            word = word[:-1]
            word_emb = model[word]
            return word_emb
        except:
            continue
    return np.zeros(100, dtype='float32')



class SLSDataset(Dataset):
    """
    Pytorch Dataset for SLS
    """
    def __init__(self, path_to_sls, tokenizer, fasttext):
        self.data = read_train_json(path_to_sls)
        self.tokenizer = tokenizer
        self.fasttext = fasttext


    def collate_fn(self, batch):
        """
        collate_fn for 'torch.utils.data.DataLoader'
        """
        texts, labels = list(zip(*[[item[0], item[1]] for item in batch]))
        token = self.tokenizer(list(texts), padding='max_length', return_offsets_mapping=True,truncation=True,max_length=510)
        modify_labels = [self._align_label(text, offset, label) for text, offset, label in zip(texts, token['offset_mapping'], labels)]
        token = self.tokenizer.pad(token, padding=True, return_attention_mask=True)
        bert_inputs = torch.LongTensor(token['input_ids'])
        bert_masks = torch.ByteTensor(token['attention_mask'])
        bert_labels = self._pad(modify_labels)
        mappings = [self.get_mapping(text) for text in texts]

        #max_len = max([len(text.lower().split()) for text in texts])
        max_len = 510  # 设置最大长度为512
        texts_list = []
        masks_list = []
        labels_list = []

        for text, label in zip(texts, labels):
            # text_list = text.lower().split()
            text_list = text.lower().split()[:max_len]
            sentence_list = []
            mask_list = []
            label = label[:max_len]  # 截断标签

            for word in text_list:
                mask_list.append(1)
                try:
                    sentence_list.append(self.fasttext[word])
                except:
                    sentence_list.append(multi_char_replace_word(self.fasttext, word))
            pad_len = max_len - len(text_list)
            while (pad_len > 0):
                sentence_list.append(np.zeros(100, dtype='float32'))
                label.append(0)
                mask_list.append(0)
                pad_len = pad_len - 1
            texts_list.append(sentence_list)
            labels_list.append(label)
            masks_list.append(mask_list)
        ft_inputs = torch.FloatTensor(texts_list)
        ft_masks = torch.ByteTensor(masks_list)
        ft_labels = torch.LongTensor(labels_list)
        mappings = self.mapping_pad(mappings)
        return bert_inputs, bert_masks, bert_labels, ft_inputs, ft_masks, ft_labels, mappings



    def _align_label(self, text, offset, label):
        label_align = [0]
        text_list = text.split()
        for word, tag_id in zip(text_list, label):
            size = len(self.tokenizer.tokenize(word))
            i = 0
            while i < size:
                label_align.append(tag_id)
                i = i+1
        label_align.append(0)
        return label_align

    def get_mapping(self, text):
        label_mapping = []
        text_list = text.split()
        for num, word in enumerate(text_list):
            size = len(self.tokenizer.tokenize(word))
            i = 0
            while i < size:
                label_mapping.append(num)
                i = i+1
        return label_mapping


    def mapping_pad(self, mappings):
        max_len = 510
        #max_len = max([len(mapping) for mapping in mappings])
        updated_mappings = []
        for mapping in mappings:
            if max_len - len(mapping) > 0 :
                mapping = (mapping + [-1] * (max_len - len(mapping)))
            else :
                mapping = mapping[:max_len]
            updated_mappings.append(mapping)
        return torch.IntTensor(updated_mappings)

    @staticmethod
    def _pad(labels):
        max_len = 510
        # max_len = max([len(label) for label in labels])
        # labels = [(label + [tag2id['O']] * (max_len - len(label))) for label in labels]
        updated_labels = []
        for label in labels:
            if max_len - len(label) > 0 :
                label = label + [tag2id['O']] * (max_len - len(label))
            else :
                label = label[:max_len]
            updated_labels.append(label)
        return torch.LongTensor(updated_labels)



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        pkg = self.data[index]
        text = pkg['text']
        label = [tag2id[tag] for tag in pkg["label"]]
        return text, label
