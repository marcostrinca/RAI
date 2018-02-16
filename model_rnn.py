from keras.layers import Dense, Dropout, LSTM, Embedding, Bidirectional, GRU
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras import optimizers

from sklearn.model_selection import train_test_split
from data_helpers import load_data

import sys, json

print('Loading data')
x, y, vocabulary, vocabulary_inv = load_data("rnn")

# save dictionarys for predictions later
with open('vocabulary.json', 'w') as fp:
    json.dump(vocabulary, fp)

with open('vocabulary_inv.json', 'w') as fp2:
    json.dump(vocabulary_inv, fp2)


# split train and test data
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=42)
sequence_length = x.shape[1]
vocabulary_size = len(vocabulary_inv)
embedding_dim = 138

print("sequence length: ", sequence_length)
print("vocabulary size: ", vocabulary_size)
print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)
# print(X_test[:10])
# print(y_test[:10])

def create_model(input_length):
    print ('Creating model...')
    model = Sequential()
    model.add(Embedding(input_dim = vocabulary_size, output_dim = embedding_dim, input_length = input_length))

    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.6))
    model.add(Dense(1, activation='sigmoid'))

    # model.add(Bidirectional(LSTM(128, return_sequences=True)))
    # model.add(Dropout(0.5))
    # model.add(Bidirectional(LSTM(64)))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))

    # output_dim = 128 gives 76% at epoch 80 with embedding_dim = 64
    # model.add(GRU(output_dim=256, activation='sigmoid', inner_activation='hard_sigmoid'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1, activation='sigmoid'))

    optimizer = optimizers.Adam(lr=0.00035)
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model


model = create_model(sequence_length)

# sys.exit(0)
checkpoint = ModelCheckpoint('./weights/w_rnn_128.{epoch:03d}-{val_acc:.4f}.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
model.fit(X_train, y_train, batch_size=80, epochs=60, callbacks=[checkpoint], validation_data=(X_test, y_test))

score, acc = model.evaluate(X_test, y_test)
print('Test score:', score)
print('Test accuracy:', acc)

# serialize model to JSON
model_json = model.to_json()
with open("model_rnn_128.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("w_rnn_128.h5")
print("Saved model to disk")

sys.exit(0)
