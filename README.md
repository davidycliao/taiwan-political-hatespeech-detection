
## Hate Speech Detection Using Flair NLP

This is a traditional Chinese hate speech classifier trained on the albert-tiny-chinese model, 
maintained by CKIP Lab, and utilizing a collection from the traditional Chinese hate speech 
dataset as described in '[Political Hate Speech Detection and Lexicon Building: 
A Study in Taiwan (2022)](https://www.researchgate.net/publication/363074513_Political_Hate_Speech_Detection_and_Lexicon_Building_A_Study_in_Taiwan),' published in _IEEE Explore_.


## Overview of Training Dataset Details

- I reorganized the training data provided by the authors of [Political Hate Speech Detection and Lexicon Building: 
A Study in Taiwan](https://www.researchgate.net/publication/363074513_Political_Hate_Speech_Detection_and_Lexicon_Building_A_Study_in_Taiwan)
  and made adjustments. The fianl dataset includes 24,856 training set, 3,108 development set, and 3,107 test set.
  The authors of the ariticle retain the rights to the original data.
- The training distribution consists of 14,035 instances of Hate Speech and 10,821 of Non-Hate Speech.
- The original weights were 1.149 for Hate Speech and 0.886 for Non-Hate Speech, which I normalized to 1.0 for Hate Speech and 0.771 for Non-Hate Speech, respectively.


## Performance

- F-score (micro): **0.7597**
- F-score (macro): **0.7584**
- Accuracy: **0.7596**

| Class           | Precision | Recall | F1-Score | Support |
|-----------------|-----------|--------|----------|---------|
| Hate Speech     | 0.8219    | 0.7345 | 0.7757   | 1759    |
| Non Hate Speech | 0.6962    | 0.7923 | 0.7412   | 1348    |
| **Micro Avg**   | **0.7598**| **0.7596** | **0.7597** | **3107** |
| **Macro Avg**   | **0.7591**| **0.7634** | **0.7584** | **3107** |
| **Weighted Avg**| **0.7674**| **0.7596** | **0.7607** | **3107** |



## Demo

### How to Use in Flair (Python)

- System: Python 3.10
- Requires: **[Flair](https://github.com/flairNLP/flair/)** (`pip install flair`)

```python
import requests

# URL to your Flair model file (direct link to the raw file)
model_file_url = 'https://github.com/davidycliao/taiwan-hatespeech-detection/raw/main/ch-hs-model/best-model.pt'

# Local path to save the model file
model_file_path = 'best-model.pt'

# Download the model
response = requests.get(model_file_url)
with open(model_file_path, 'wb') as file:
    file.write(response.content)

```

```python
from flair.data import Sentence
from flair.models import TextClassifier
# Sentence to classify
sentence = Sentence("這位女士有點志氣好嗎？韓粉都是這種人")

# Model detection
classifier.predict(sentence)

# Print sentence
print(sentence)

```

This yields the following output:

```terminal
Sentence[1]: "這位女士有點志氣好嗎？韓粉都是這種人" → Hate Speech (0.8336)
```


### How to Use in R

```r
library(flaiR)# flaiR: An R Wrapper for Accessing Flair NLP 0.12.2

```

```rSentence <- flair_data()$Sentence TextClassifier <- flair_models()$TextClassifier# Specify the URL of the model filemodel_file_url <- "https://github.com/davidycliao/taiwan-hatespeech-detection/raw/main/ch-hs-model/best-model.pt"# Set the local path where you want to save the modelmodel_file_path <- "best-model.pt"# Use download.file() to download the modeldownload.file(model_file_url, model_file_path, method="auto")classifier <- TextClassifier$load(model_file_path)# Sentence to classifysentence <- Sentence("這位女士有點志氣好嗎？韓粉都是這種人")# Model detectionclassifier$predict(sentence)# Print sentenceprint(sentence)
```

This yields the following output:

```terminal
Sentence[1]: "這位女士有點志氣好嗎？韓粉都是這種人" → Hate Speech (0.8336)
```


### Training: Script to Train the Model


The model was trained using the following Flair script and completed **30** epochs:

```python

from flair.data import Sentence
from flair.datasets import SentenceDataset
from flair.embeddings import TransformerDocumentEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import json
import flair

# Organize the Datasets

## 1. Load the file with the hate speech data

data = pd.read_csv('merged-hate-speeches.csv', encoding='utf-8')

## 2. Tokenize the text and add Labels
sentences = [Sentence(text) for text in data['text']]

for sentence, tag in zip(sentences, data['label']):
    sentence.add_label('classification', str(tag))

## 3. Split dataset into train and test
train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=2046)
test_sentences, dev_sentences = train_test_split(test_sentences, test_size=0.5, random_state=2046)

## 4. Create a corpus using the training and test datasets
corpus = Corpus(train=train_sentences, 
                test=test_sentences,
                dev= dev_sentences)

# Create Document-level Embeddings 

document_embeddings = TransformerDocumentEmbeddings('ckiplab/albert-tiny-chinese', fine_tune=True)
label_dict = corpus.make_label_dictionary(label_type="classification")

# Dealing with Imbalance Instances

train_labels = [label.value for sentence in train_sentences for label in sentence.labels]
train_labels = np.array(train_labels)

class_weights = compute_class_weight(
    class_weight='balanced', 
    classes= np.unique(train_labels),
    y=train_labels
)

weights = {index: weight for index, weight in enumerate(class_weights)}
max_weight = max(weights.values())
normalized_weights = {k: v / max_weight for k, v in weights.items()}


# Creating the TextClassifier

classifier = TextClassifier(document_embeddings,
                            label_dictionary=label_dict,
                            label_type='classification',
                            loss_weights=normalized_weights)

classifier = classifier.to('gpu')

trainer = ModelTrainer(classifier, corpus)


trainer.train('taiwan-ckip-hatespeech-detection',
              shuffle = True,                 
              patience=5,                     
              learning_rate=0.02,             
              mini_batch_size=16,            
              write_weights = True,          
              max_epochs=30)                 
```


## Cite

Please cite the following paper when using this model.

```
@misc{davidliao2024,
  title={Political Hate Speech Detection in Traiditonal Chinese},
  author={Yen-Chieh Liao}
  year      = {2024}
}

@article{Wang2022,
  title={Political Hate Speech Detection and Lexicon Building: A study in Taiwan},
  author={Wang, Chih-Chien and Day, Min-Yuh and Wu, Chun-Lian},
  journal={IEEE Access},
  volume={10},
  pages={44337--44346},
  year={2022},
  publisher={IEEE}
}

```