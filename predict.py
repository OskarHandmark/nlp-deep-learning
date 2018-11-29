import plac
import json
import numpy as np
import pandas as pd
from keras.models import load_model
from src.corenlp import CoreNLP
from src.feature_extraction import extract_features

def predict(sentence: "Sentence string to predict output classes for"):

  with open('features.json', 'r') as f:
    feature_names = json.loads(f.read())

  with open('labels.json', 'r') as f2:
    labels = json.loads(f2.read())

  core_nlp  = CoreNLP()
  tokens    = core_nlp.tokenize_sentence(sentence)
  features  = extract_features(tokens)
  
  X = pd.DataFrame(features).fillna(0) * 1
  X = pd.get_dummies(X)
  X = pd.DataFrame(X, columns = feature_names).fillna(0)
  
  model  = load_model('model.h5')
  
  pred   = model.predict(X)
  result = pd.DataFrame(pred, columns=labels)
  result = result.round(2)

  result['word'] = [token['token'] for token in tokens]
  result['vote'] = result[labels].idxmax(axis=1)
  print(result)
  return result

if __name__ == "__main__":
    plac.call(predict)