import json
from pycorenlp import StanfordCoreNLP

class CoreNLP(object):

    def __init__(self):
        self.corenlp = StanfordCoreNLP('http://localhost:9000')

    def tokenize_sentence(self, sentence):
        doc = self.corenlp.annotate(sentence, properties={
            'annotators': 'tokenize,lemma,ssplit,pos,depparse,parse',
            'outputFormat': 'json'
        })
        s = doc['sentences'][0]

        return self.structure_tokens(s)

    def structure_tokens(self, sentence):
        words = []
        for token in sentence['tokens']:
            words.append({
              "pos": token['pos'],
              "token": token['word'],
              "links": [],
              "lemma": token['lemma']
            })

        for dep in sentence['enhancedPlusPlusDependencies']:
          words[dep['governor'] - 1]['links'].append([dep['dep'], (dep['dependent'] - 1)])
        return words
