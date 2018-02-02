import textract
import os, sys
import collections
import string
import random

# helper function: limpo o texto
import regex as re
def clean_text(text):

    # removo pontuação e múltiplos espaços assim como múltiplas quebras de linha
    text = re.sub("[^\P{P}-]+", "", text)
    text = re.sub(" +", " ", text)
    text = re.sub(r'\n\s*\n', '\n\n', text)

    return text


# Recebe uma pasta e retorna uma lista com o conteúdo de cada arquivo em uma posição 
def load_text_from_docx(folder):

    # armazeno cada documento em uma posição de uma lista
    dict_docs = {}
    for filename in os.listdir(folder):

        # path
        path = folder+"/"+filename

        # extraio o texto e converto para string utf8
        content = textract.process(path)
        text = content.decode('utf8')

        # limpo o texto
        text = clean_text(text)

        dict_docs[filename] = text

    return dict_docs


# Recebe dicionário com todos os textos de todos os documentos e uma lista de palavras indicando conduta radiológica
# Retorna dicionário com o nome do documento e a lista de linhas que possuem as palavras
def match_docs_with_words(key_words, d_docs):

    # lista das palavras mais comuns e que não ajudam na classificação
    stoplist = set('e i é a à as o os ou da das do dos de em na no com se ao por dr dra'.split())

    dict_possible_docs = {}

    # em cada um dos documentos
    for doc_name, v in d_docs.items():

        # gero uma lista com todas as palavras do documento
        doc_words = [word for word in v.lower().split() if word not in stoplist]
        # print(doc_name)

        # lista das frases que indicam conduta
        phrases = []

        # procuro nas palavras do documento por palavras que estejam na lista de key_words
        for k_w in key_words:
            if k_w in doc_words:
                # print(k_w)

                # abro o documento original e pego o parágrafo inteiro que possui esta palavra
                orig_docx = textract.process('training-dataset/' + doc_name)
                orig_docx_text = clean_text(orig_docx.decode('utf8'))

                lines = orig_docx_text.lower().split('\n\n')
                for line in lines:
                    sub_idx = line.find(k_w)
                    if sub_idx >= 0:
                        # print(orig_docx_text)
                        # print(line)

                        # guardo esta linha se já não guardei antes
                        if line not in phrases:
                            phrases.append(line)

        # adiciono a lista de linhas no dicionário com o nome do arquivo
        dict_possible_docs[doc_name] = phrases

    return(dict_possible_docs)


### 1. carrego todos os documentos em um dicionário com key nome e value conteúdo
my_d_docs = load_text_from_docx('data_original')
# for key in sorted(my_d_docs):
#   print("%s: \n %s" % (key, my_d_docs[key]))

### 2. crio um segundo dicionário com todos os documentos que possuam QUALQUER palavra indicadora de conduta radiológica
### e com a(s) frase(s) que possui estas palavras numa lista (estas frases serão o input da RNN)
key_words = (
    'sugerimos', 
    'sugere-se', 
    'sugerindo-se',
    'sugiro',
    'conveniente', 
    'convém',
    'necessária',
    'correlação', 
    'correlacionar', 
    'recomenda', 
    'recomendamos', 
    'recomenda-se', 
    'recomendando-se',
    'recomendação',
    'considerar',
    'merece',
    'merecendo',
    'devendo-se',
    'estudo',
    'poderá',
    'poderão',
    'pode',
    'podem',
    'poderia',
    'repetir',
    'complementar',
    'manter',
    'controle')

d_possible_docs = match_docs_with_words(key_words, my_d_docs)
training_set = []
for key in sorted(d_possible_docs):
    # print("%s: \n %s" % (key, d_possible_docs[key]))
    if len(d_possible_docs[key]) is not 0:
        # print(d_possible_docs[key])
        for sentence in d_possible_docs[key]:
            training_set.append(sentence)

print(len(training_set))
f = open("sentence_extracted.txt","w")
for s in training_set:
    f.write(s + " . \n\n")
f.close()








