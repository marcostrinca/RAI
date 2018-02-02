from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, GRU
# from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from data_helpers import load_data

import sys

print('Loading data')
x, y, vocabulary, vocabulary_inv = load_data("rnn")
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)
sequence_length = x.shape[1]
vocabulary_size = len(vocabulary_inv)
embedding_dim = 80

print("sequence length: ", sequence_length)
print("vocabulary size: ", vocabulary_size)
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
print(X_test[:10])
print(y_test[:10])

def create_model(input_length):
    print ('Creating model...')
    model = Sequential()
    model.add(Embedding(input_dim = vocabulary_size, output_dim = embedding_dim, input_length = input_length))

    # GRU(128) gives 80% at epoch 80 with embedding_dim = 64
    model.add(Bidirectional(GRU(128)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # output_dim = 128 gives 76% at epoch 80 with embedding_dim = 64
    # model.add(GRU(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


checkpoint = ModelCheckpoint('weights_lstm.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
model = create_model(sequence_length)

# sys.exit(0)
model.fit(X_train, y_train, batch_size=33, epochs=25, callbacks=[checkpoint], validation_data=(X_test, y_test))

score, acc = model.evaluate(X_test, y_test)
print('Test score:', score)
print('Test accuracy:', acc)
