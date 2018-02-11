import textract
import os, sys
import collections
import string
import random

def build_dictionaries(words):
    count = collections.Counter(words).most_common()

    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return dictionary, reverse_dictionary


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
        # phrases_raw = text.split('\n')

        # gero uma lista com cada uma das palavras
        words_raw = text.split()
        words = [i.lower() for i in words_raw]
        docs.append(words)

        # limpo o dataset para deixar apenas palavars, sem pontuações
        words_no_point = [i.replace('.','') for i in words]
        words_no_coma = [i.replace(',','') for i in words_no_point]
        words_no_hiphen = [i.replace('-','') for i in words_no_coma]
        words_no_doublepoint = [i.replace(':','') for i in words_no_hiphen]
        words_no_pointcoma = [i.replace(';','') for i in words_no_doublepoint]
        words_no_parenthesis1 = [i.replace('(','') for i in words_no_pointcoma]
        words_no_parenthesis2 = [i.replace(')','') for i in words_no_parenthesis1]

        # gero uma lista com todas as palavras limpinhas
        docs_clean.append(words_no_parenthesis2)

        # gero umalista com todo o vocabulário flat de palavras limpinhas
        docs_flat = [item for sublist in docs_clean for item in sublist]

    return docs, docs_clean, docs_flat


# carrego as listas
my_docs, my_docs_clean, my_docs_flat = load_text_from_docx('training-dataset')

# dicionarios: index to word e word to index
dictionary, reverse_dictionary = build_dictionaries(my_docs_flat)
vocab_size = len(dictionary)

print(dictionary)
print(reverse_dictionary)



# print("example dict: {}".format(dictionary[0:100]))
# print("example reverse: {}".format(reverse_dictionary[0:100]))
# print(vocab_size)

