import sys
import textract
import regex as re
import json
from pprint import pprint

import numpy as np
from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, GRU
from keras.models import Sequential, model_from_json
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

from data_helpers import pad_sentences, build_input_data


def clean_text(text):

    # removo pontuação e múltiplos espaços assim como múltiplas quebras de linha
    text = re.sub("[^\P{P}\-]+", "", text)
    text = re.sub(" +", " ", text)
    text = re.sub("\n\s*\n", "\n", text)

    return text

def clean_string(string):
    string = string.replace(".", "")
    string = re.sub(" +", " ", string)
    string = re.sub(r'\n\s*\n', '\n\n', string)
    string = string.replace("\n", "")

    return string.strip().lower()


def main():
    my_file = sys.argv[1]

    # extraio o texto e converto para string utf8
    content = textract.process(my_file)
    text = content.decode('utf8')

    # limpo o texto
    text = clean_text(text)

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

    stoplist = set('e i é a à as o os ou da das do dos de em na no com se ao por dr dra'.split())
    doc_words = [word for word in text.lower().split() if word not in stoplist]
    # print(doc_words)

    # lista das frases que indicam conduta
    phrases = []

    # procuro nas palavras do documento por palavras que estejam na lista de key_words
    for k_w in key_words:
        if k_w in doc_words:

            lines = text.lower().split('\n')
            for line in lines:

                sub_idx = line.find(k_w)
                if sub_idx >= 0:
                    # print(orig_docx_text)
                    # print(line)

                    # guardo esta linha se já não guardei antes
                    if line not in phrases:
                        phrases.append(line)

    print("\n\n ----> frases coletadas no arquivo: ", phrases)

    # prepare data to be feeded
    x_text = [clean_string(sent) for sent in phrases]
    x_text = [s.split(" ") for s in x_text]
    # print("-----> frases preparadas: ", x_text)

    sequence_lenght = 182
    padding_word = "<PAD/>"
    padded_sentences = []
    for i in range(len(x_text)):
        sentence = x_text[i]
        num_padding = sequence_lenght - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    # print("-----> com pads: ", padded_sentences)

    # carrego o vocabulário
    with open('vocabulary.json') as data_file:    
        vocab = json.load(data_file)
    
    x = np.array([[vocab[word] for word in sentence] for sentence in padded_sentences])
    # print("----> input para predição: ", x)

    # carrego o modelo gravado no json
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # carrego os pesos gravados no hdf5
    loaded_model.load_weights('./weights_final.h5')

    # faço as predições
    preds = loaded_model.predict(x)

    # armazeno as predições se tiverem probabilidade acima de 50%
    pred_count = 0
    for i, item in enumerate(preds):
        if item > 0.5:
            pred_count = pred_count + 1
            print("\n\n ***** Frase com indicação de conduta radiológica encontrada: ", phrases[i])

    if pred_count == 0:
        print("\n\n ***** Não foi encontrada indicação de conduta radiológica *****\n\n")

    # json_return = json.dumps(d, ensure_ascii=False)
    # print(json_return)

    sys.exit(0)

if __name__ == '__main__':
    main()