## Multilingual Code-Switching for Zero-Shot Cross-Lingual Intent Prediction and Slot Filling

**Goal**: Train a joint intent prediction and slot fillinf model using English and generalize to other languages.

### Paper/Cite
https://arxiv.org/abs/2103.07792

### Requirements

1) !pip install transformers

2) !pip install googletrans

3) Enable Cuda for Joint Training

### Datasets
MultiAtis++: Please visit https://github.com/amazon-research/multiatis 

CrisisData: Please send us an email to obtain a copy of the data.

Original Source: [Appen](https://appen.com/datasets/combined-disaster-response-data)

[Datasets](https://github.com/jitinkrishnan/Multilingual-ZeroShot-SlotFilling/blob/main/dataset_readme.md) to setup before running the experiments.

### How to Run: Joint Training

#### joint training (English Only)
```
python3 joint_en.py <location-of-data-folder> '0'
```

#### joint training (English Only with Code Switching)
```
python3 joint_en.py <location-of-data-folder> '1'
```

### How to Run: Code-Switching
```
python3 code_switch.py <input_fileName> <pickle_output_fileName>
#e.g., python3 code_switch.py 'train_EN.tsv' 'train_cs.p'
```
