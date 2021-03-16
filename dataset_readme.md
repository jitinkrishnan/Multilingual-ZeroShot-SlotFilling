

### English (en) - Haitian Creole (ht) Dataset (7 files)

#### TEST Haitian Creole:
- disaster_response_test_HT.tsv

#### TRAIN-DEV ENGLISH:
- disaster_response_train_EN.tsv
- disaster_response_dev_EN.tsv

#### TRAIN-DEV CODESWITCHED:
- disaster_response_train_EN_codeswitched.p
- disaster_response_dev_EN_codeswitched.p

#### TRAIN-DEV TRANSLATED:
- disaster_response_train_HT_translated.p
- disaster_response_dev_HT_translated.p

### Code-Switched MultiATTIS++ (18 files)

#### Original MultiATTIS++ Dataset Source: 
https://github.com/amazon-research/multiatis 

#### Code-Switched Dataset:

- word-level: train_word_cs.p, dev_word_cs.p
- chunk-level: train_chunk_cs.p, dev_chunk_cs.p
- sentence-level: train_sentence_cs.p, dev_sentence_cs.p

#### Code-Switched Language Family Dataset (chunk-level):

```x = train, dev```
- x_afroasiatic.p
- x_romance.p
- x_germanic.p
- x_sinotibetan_japonic.p
- x_indoaryan.p
- x_turkic.p

#### How to read from Pickle:
triples (words, slot labels, intent class)

```
import pickle
triples = pickle.load(open('filename', "rb"))
```
