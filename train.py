import json
import numpy as np
import pandas as pd
from src.corenlp import CoreNLP
from src.feature_extraction import extract_features
from src.model import train_model

def train():
  core_nlp = CoreNLP()
  
  df = pd.read_table('data/train.txt', delimiter=' ', names=['word', 'label'])

  sentences = " ".join(df['word']).replace(' ,', ',').replace(' .', '.').split('. ')
  sentences = [s.strip().replace('.', '') + '.' for s in sentences]
  
  features = []
  for s in sentences:
    tokens = core_nlp.tokenize_sentence(s)
    features.append(extract_features(tokens))

  X = pd.concat([pd.DataFrame(f) for f in features], sort=False)
  X = X.fillna(0) * 1 # convert True to 0 before one_hot_enc

  X = pd.get_dummies(X)            # one_hot_enc
  Y = pd.get_dummies(df['label'])  # aka to_categorical

  model = train_model(X, Y)
  
  model.save('model.h5')

  with open('labels.json', 'w') as f1:    # Labels used
    json.dump(Y.columns.tolist(), f1)
  
  with open('features.json', 'w') as f2:  # Features used
    json.dump(X.columns.tolist(), f2)

  return model

if __name__ == "__main__":
  train()
  