import textract
import os, sys
import collections
import string
import random

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

''' Recebe uma pasta e retorna uma lista 
com o conteúdo de cada arquivo em uma posição '''
def load_text_from_docx(folder):

    # armazeno cada documento em uma posição de uma lsita
    docs = []
    docs_clean = []
    for filename in os.listdir(folder):

        # path
        path = folder+"/"+filename
        # print(path)

        # extraio o texto e converto para string utf8
        content = textract.process(path)
        text = content.decode('utf8')

        # removo pontuação
        table = str.maketrans({key: None for key in string.punctuation})
        text = text.translate(table)

        docs.append(text)

    return docs
        

my_docs = load_text_from_docx('training-dataset')
# print(len(my_docs))
# print(my_docs[0])

stoplist = set('e i é a à as o os ou da das do dos de em na no com se ao por dr dra'.split())
texts = [[word for word in document.lower().split() if word not in stoplist] for document in my_docs]
# print(texts)

# removo palacras que aparecem apenas uma vez
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
    for token in text:
        frequency[token] += 1
texts = [[token for token in text if frequency[token] > 1] for text in texts]
# print(texts)

from gensim import corpora
dictionary = corpora.Dictionary(texts)
dictionary.save('dictionary.dict')
# print(dictionary.token2id)

corpus = [dictionary.doc2bow(text) for text in texts]
corpora.MmCorpus.serialize('deerwester.mm', corpus)
# print(corpus)








