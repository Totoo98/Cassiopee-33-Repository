import numpy
import os
import inspect
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

# tf.compat.v1.disable_eager_execution()

#is_training = False
is_training = False
is_generating = not is_training

epochs = 10
batch_size = 64
characters_generate_count = 1000


# Load ASCII text and convert it to lowercase
src_file_path = os.path.dirname(inspect.getfile(lambda: None))
rel_path = "data.txt"
abs_file_path = src_file_path + '/' + rel_path
raw_text = open(abs_file_path, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()


# Create a dictionary between characters and integers for all characters of the data
characters = sorted(list(set(raw_text)))
characters_to_int = dict((c, i) for i, c in enumerate(characters))
int_to_characters = dict((i, c) for i, c in enumerate(characters))


# Create meta data
characters_count = len(raw_text)
vocabulary_count = len(characters)

print("Characters count:", characters_count);
print("Vocabulary count:", vocabulary_count);


# Setup the dataset of input and output pairs, encoded as integers

sequence_length = 100
dataX = []
dataY = []

for i in range(0, characters_count - sequence_length, 1):
    sequence_input = raw_text[i : i + sequence_length]
    sequence_output = raw_text[i + sequence_length]
    dataX.append([characters_to_int[char] for char in sequence_input])
    dataY.append(characters_to_int[sequence_output])

patterns_count = len(dataX)
print("Patterns count:", patterns_count)


# Reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (patterns_count, sequence_length, 1))

# Normalize in 0 - 1 range
X = X / float(vocabulary_count)

# One hot encore the output variable
y = np_utils.to_categorical(dataY)


# Define the LSTM model
model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))


if(is_generating):
    # Load the network weights
    weight_filename = "weights-improvement-1.6558-10.hdf5"
    weight_filepath = src_file_path + "/weights/" + weight_filename
    model.load_weights(weight_filepath)

if(is_training):
    # Load the network weights
    weight_filename = "weights-improvement-1.6558-10.hdf5"
    weight_filepath = src_file_path + "/weights/" + weight_filename
    model.load_weights(weight_filepath)


model.compile(loss='categorical_crossentropy', optimizer='adam')

if(is_training):
    # Define the checkpoint to record the network weights each time an improvement in loss is observed at the end of the epoch
    filepath=src_file_path + "/weights/" + "weights-improvement-{loss:.4f}-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]


    # Fit our model to the data
    model.fit(X, y, batch_size, epochs, callbacks = callbacks_list)


if(is_generating):
    # Pick a random seed
    start = numpy.random.randint(0, len(dataX) - 1)
    pattern = dataX[start]
    print("Seed:")
    seed = ''.join([int_to_characters[value] for value in pattern])
    sys.stdout.write(seed)
    sys.stdout.write('\n\nText generated:\n')

    output = ""

    # Generate characters
    for i in range(characters_generate_count):
        x = numpy.reshape(pattern, (1, len(pattern), 1))
        x = x / float(vocabulary_count)
        prediction = model.predict(x, verbose = 0)
        index = numpy.argmax(prediction)
        result = int_to_characters[index]
        sequence_input = [int_to_characters[value] for value in pattern]
        sys.stdout.write(result)
        output += result
        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    print("\nDone")
