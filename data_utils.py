import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
import pickle

batch_size = 32

def tokenize_and_preserve_labels(sentence, text_labels, sent_labels, tokenizer):
    tokenized_sentence = []
    labels = []

    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels, sent_labels

def getData4Bert(FNAME, tokenizer, sentenceLabel2idx=None, tag2idx=None, MAX_LEN=None):
    data = pd.read_csv(FNAME, sep='\t', error_bad_lines=False).fillna(method="ffill")

    sentences = []
    tags = []
    labels = []

    for index, row in data.iterrows():
        sentences.append(row['utterance'].split())
        tags.append(row['slot_labels'].split())
        labels.append(row['intent'])

    sentence_labels_set = list(set(labels))
    sentence_labels_set.sort()
    if sentenceLabel2idx is None:
        sentenceLabel2idx = {t: i for i, t in enumerate(sentence_labels_set)}
    else:
        a = list(sentenceLabel2idx.keys())
        b = list(sentence_labels_set)
        #print("not in...")
        for l in b:
            if l not in a:
                #print(l)
                sentenceLabel2idx[l] = len(sentenceLabel2idx)

    all_tags = []
    for tag in tags:
        all_tags.extend(tag)
    tag_values = list(set(all_tags))
    tag_values.append("PAD")
    tag_values.sort()
    if tag2idx is None:
        tag2idx = {t: i for i, t in enumerate(tag_values)}
    else:
        for val in tag_values:
            if tag2idx.get(val, -1) == -1:
                tag2idx[val] = tag2idx['O']

    seq_len = [len(sentence) for sentence in sentences]

    if MAX_LEN is None:
        MAX_LEN = int(max(seq_len))

    tokenized_texts_and_labels = [
        tokenize_and_preserve_labels(sent, labs, slabs,tokenizer)
        for sent, labs, slabs in zip(sentences, tags, labels)
    ]

    sent_labels = labels

    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
    sent_labels = [token_label_pair[2] for token_label_pair in tokenized_texts_and_labels]

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", value=0.0,
                              truncating="post", padding="post")

    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                         maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                         dtype="long", truncating="post")

    sent_tags = [sentenceLabel2idx[sl] for sl in sent_labels]

    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(tags), torch.tensor(sent_tags), sentenceLabel2idx, tag2idx, MAX_LEN

def getData4BertPickle(FNAME, tokenizer, sentenceLabel2idx=None, tag2idx=None, MAX_LEN=None):
    data = pickle.load(open(FNAME, "rb" ) )

    sentences = []
    tags = []
    labels = []

    for triple in data:
        sentences.append(list(triple[0]))
        tags.append(list(triple[1]))
        labels.append(triple[2])

    sentence_labels_set = list(set(labels))
    sentence_labels_set.sort()
    if sentenceLabel2idx is None:
        sentenceLabel2idx = {t: i for i, t in enumerate(sentence_labels_set)}
    else:
        a = list(sentenceLabel2idx.keys())
        b = list(sentence_labels_set)
        #print("not in...")
        for l in b:
            if l not in a:
                #print(l)
                sentenceLabel2idx[l] = len(sentenceLabel2idx)

    all_tags = []
    for tag in tags:
        all_tags.extend(tag)
    tag_values = list(set(all_tags))
    tag_values.append("PAD")
    tag_values.sort()
    if tag2idx is None:
        tag2idx = {t: i for i, t in enumerate(tag_values)}
    else:
        for val in tag_values:
            if tag2idx.get(val, -1) == -1:
                tag2idx[val] = tag2idx['O']

    seq_len = [len(sentence) for sentence in sentences]

    if MAX_LEN is None:
        MAX_LEN = int(max(seq_len))

    tokenized_texts_and_labels = [
        tokenize_and_preserve_labels(sent, labs, slabs,tokenizer)
        for sent, labs, slabs in zip(sentences, tags, labels)
    ]

    sent_labels = labels

    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]
    sent_labels = [token_label_pair[2] for token_label_pair in tokenized_texts_and_labels]

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                              maxlen=MAX_LEN, dtype="long", value=0.0,
                              truncating="post", padding="post")

    tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                         maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                         dtype="long", truncating="post")

    sent_tags = [sentenceLabel2idx[sl] for sl in sent_labels]

    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

    return torch.tensor(input_ids), torch.tensor(attention_masks), torch.tensor(tags), torch.tensor(sent_tags), sentenceLabel2idx, tag2idx, MAX_LEN

def create_loaders(seq, mask, slot_y, y):
  # wrap tensors
  data = TensorDataset(seq, mask, slot_y, y)
  # sampler for sampling the data during training
  sampler = RandomSampler(data)
  # dataLoader for train set
  dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
  return dataloader