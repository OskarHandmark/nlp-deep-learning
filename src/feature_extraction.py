
def add_out_links(f, links):
  for link in links:
      dep, index = link
      prop = "out_" + dep
      f[prop] = True
  return f

def add_in_links(f, sentence, w_index):
  for w in sentence:
      for link in w['links']:
          dep, target_index = link
          if target_index == w_index:
              prop = "in_" + dep
              f[prop] = True
  return f

def create_feature_vector(sentence, w_index):
  word = sentence[w_index]

  f = {
      'pos': word['pos']
  }

  links = word['links']
  f = add_out_links(f, links)
  f = add_in_links(f, sentence, w_index)

  # POS WINDOW
  for steps in range(1, 3):
      pos_before_title = "pos_before" + str(steps)
      pos_after_title = "pos_after" + str(steps)
      f[pos_before_title] = f[pos_after_title] = "NONE"
      if w_index > steps - 1:
          f[pos_before_title] = sentence[w_index - steps]['pos']
      if w_index < len(sentence) - steps:
          f[pos_after_title] = sentence[w_index + steps]['pos']
  return f

def extract_features(sentence_tokens):
  features = []
  
  for i, w in enumerate(sentence_tokens):
      f = create_feature_vector(sentence_tokens, i)
      features.append(f)
  
  return features