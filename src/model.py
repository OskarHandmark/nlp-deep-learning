import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from keras.wrappers.scikit_learn import KerasClassifier

def cross_val(model, X, Y):
  scores = cross_val_score(model, X, Y, cv=5)

  print("Accuracy: %0.4f (+/- %0.4f)" % (scores.mean(), scores.std() * 2))

def get_model(input_dim, output_dim):
  def create():
    model = Sequential()
    model.add(Dense(100, input_dim=input_dim, activation='relu'))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(output_dim, activation='softmax'))

    # Compile model
    model.compile(
      loss='categorical_crossentropy',
      optimizer='adam', 
      metrics=['accuracy']
    )

    return model

  return create

def train_model(X, Y):
  epochs = 50
  
  input_dim = X.shape[1]
  output_dim = Y.shape[1]

  model = get_model(input_dim, output_dim)()
  model.fit(X, Y, epochs=epochs, batch_size=32)

  sklearn_model = KerasClassifier(build_fn=get_model(input_dim, output_dim), epochs=epochs, batch_size=32, verbose=0)
  
  cross_val(sklearn_model, X, Y)

  return model
