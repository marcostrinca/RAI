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

    # armazeno cada documento em uma posição de uma lista
    docs = []
    doc_name = []
    for filename in os.listdir(folder):

        # path
        path = folder+"/"+filename
        # print(path)

        # extraio o texto e converto para string utf8
        content = textract.process(path)
        text = content.decode('utf8')

        # removo pontuação
        # table = str.maketrans({key: None for key in string.punctuation})
        # text = text.translate(table)
        import regex as re
        text = re.sub("[^\P{P}-]+", "", text)

        docs.append(text)
        doc_name.append(filename)

    return docs, doc_name
        

# guardo a lista de documentos assim como a lista de nomes com os mesmos indices
my_docs, my_docs_name = load_text_from_docx('training-dataset')

# removo as palavras mais comuns e que não ajudam na classificação
stoplist = set('e i é a à as o os ou da das do dos de em na no com se ao por dr dra'.split())
texts = [[word for word in document.lower().split() if word not in stoplist] for document in my_docs]

# removo palavras que aparecem apenas uma vez
# from collections import defaultdict
# frequency = defaultdict(int)
# for text in texts:
#     for token in text:
#         frequency[token] += 1
# texts = [[token for token in text if frequency[token] > 1] for text in texts]
# print(texts[0])

##### debug search indexes #####
# i=0
# for dd in my_docs_name:
    # i += 1
    # if dd == 'Abdome - Laudo 02.docx':
        # print(i)

# print(texts[10])
# print(my_docs_name[10])
# sys.exit(0)
##### debug search indexes #####
        
# procuro documentos que possuam palavras que estão no array de referência abaixo
key_words = ('sugerimos', 
'sugere-se', 
'recomendação', 
'conveniente', 
'correlação', 
'correlacionar', 
'recomenda', 
'recomendamos', 
'recomenda-se', 
'recomendando-se',
'controle')

results_positive = []
results_negative = []

# dictionary
d = {}

# em cada um dos documentos
for t in texts:

    # armazeno o index do documento atual
    idx = texts.index(t)

    # if my_docs_name[idx] == "Abdome - Laudo 08.docx":
    #         print(t)
    #         print(my_docs_name[idx])
    #         sys.exit(0)

    ##### debug #####
    # print(my_docs_name[idx])
    # print(t)
    # if 'recomenda' in t:
        # print(my_docs_name[idx])
    ##### debug #####

    # se encontro alguma das keywords
    for k in key_words:
        if k in t:

            # armazeno nome do documento na lista de positivos
            doc_name = my_docs_name[idx]

            # if doc_name not in results_positive:
            #     results_positive.append(doc_name)
            #     if doc_name in results_negative:
            #         results_negative.remove(doc_name)
            
            ##### debug
            # if doc_name == 'Abdome - Laudo 08.docx':
            #     docx = textract.process('training-dataset/'+my_docs_name[idx])
            #     docx_text = docx.decode('utf8')
            #     lines = docx_text.lower().split('\n\n')
            #     print(lines)
            #     sys.exit(0)
            ##### debug

            # abro documento e procuro pela linha de texto que contém a key
            docx = textract.process('training-dataset/'+my_docs_name[idx])
            docx_text = docx.decode('utf8')

            lines = docx_text.lower().split('\n')
            for line in lines:
                sub_idx = line.find(k)
                if sub_idx >= 0:

                    # adiciono a linha no dicionário com o nome do arquivo
                    d[doc_name] = line

        else:
            if my_docs_name[idx] not in results_negative and my_docs_name[idx] not in results_positive:
                results_negative.append(my_docs_name[idx])

for key in sorted(d):
    print("%s: \n %s" % (key, d[key]))
    # print("%s" % (key))

# import pprint as pp
# pp.pprint(sorted(results_negative))
# pp.pprint(sorted(results_positive))













