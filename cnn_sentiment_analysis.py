# Thank you to the COMP474 Winter 2022 team for this fun tutorial!
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Dropout, Activation, MaxPooling1D, Flatten
import csv
from random import shuffle
import spacy
from sklearn.model_selection import train_test_split
import numpy as np

# use a random seed
np.random.seed(1337)

# creates tuples from the data where 1st element is target, 2nd is the text
def pre_process_data(filepath):
    print('Preprocessing data...')
    dataset = []
    with open(filepath, 'r', encoding='utf8') as in_file:
        csv_reader = csv.reader(in_file, delimiter=',')
        next(csv_reader, None)
        for row in csv_reader:
            dataset.append((int(row[1]), row[0]))

    shuffle(dataset)
    return dataset

# Use spaCy to extract all word embeddings for tokens
def tokenize_and_vectorize(dataset):
    print('Vectorizing data...')
    nlp = spacy.load('en_core_web_md')
    vectorized_data = []
    for sample in dataset:
        doc = nlp(sample[1])
        sample_vec = []
        for token in doc:
            try:
                sample_vec.append(token.vector)
            except KeyError:
                pass
        vectorized_data.append(sample_vec)
    return vectorized_data

# extract all target labels from dataset
def collect_expected(dataset):
    print('Target extraction...')
    # Extract target values from the dataset
    expected = []
    for sample in dataset:
        expected.append(sample[0])
    return expected


#  truncate or pad with 0s to have same input vector length for all data points
def pad_trunc(data, maxlen):
    # For a given dataset pad with zero vectors or truncate to maxlen
    new_data = []

    # Create a vector of 0s the length of our word vectors
    zero_vector = [0] * len(data[0][0])
    for sample in data:
        if len(sample) > maxlen:
            temp = sample[:maxlen]
        elif len(sample) < maxlen:
            temp = sample
            additional_elems = maxlen - len(sample)
            for _ in range(additional_elems):
                temp.append(zero_vector)
        else:
            temp = sample
        new_data.append(temp)
    return new_data

dataset = pre_process_data('IMDB.csv')
vectorized_data = tokenize_and_vectorize(dataset)
expected = collect_expected(dataset)

x_train, x_test, y_train, y_test = train_test_split(vectorized_data, expected, test_size=.20, random_state=40)

# initial params
max_len = 400
batch_size = 32
embedding_dim = 300
filters = 250
kernel_size = 3
hidden_dim = 250
epochs = 2
num_classes = 1

# normalize datapoint
x_train = pad_trunc(x_train, max_len)
x_test = pad_trunc(x_test, max_len)

# convert to numpy data structure
x_train = np.reshape(x_train, (len(x_train), max_len, embedding_dim))
y_train = np.array(y_train)
x_test = np.reshape(x_test, (len(x_test), max_len, embedding_dim))
y_test = np.array(y_test)

# construct the CNN architecture
model = Sequential()
model.add(Conv1D(filters=filters,
                 kernel_size=kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1,
                 input_shape=(max_len, embedding_dim)))

model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(hidden_dim))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

# compile CNN architecture
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# fine tune model according to training data
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test))

loss, acc = model.evaluate(x_test, y_test, verbose=0)
print ('Test Accuracy: %f' % (acc * 100))

sample_1 = """I always wrote this series off as being a complete stink-fest because Jim Belushi was involved in it, and heavily. But then one day a tragic happenstance occurred. After a White Sox game ended I realized that the remote was all the way on the other side of the room somehow. Now I could have just gotten up and walked across the room to get the remote, or even to the TV to turn the channel. But then why not just get up and walk across the country to watch TV in another state? ""Nuts to that"", I said. So I decided to just hang tight on the couch and take whatever Fate had in store for me. What Fate had in store was an episode of this show, an episode about which I remember very little except that I had once again made a very broad, general sweeping blanket judgment based on zero objective or experiential evidence with nothing whatsoever to back my opinions up with, and once again I was completely right! This show is a total crud-pie! Belushi has all the comedic delivery of a hairy lighthouse foghorn. The women are physically attractive but too Stepford-is to elicit any real feeling from the viewer. There is absolutely no reason to stop yourself from running down to the local TV station with a can of gasoline and a flamethrower and sending every copy of this mutt howling back to hell. <br /><br />Except.. <br /><br />Except for the wonderful comic sty lings of Larry Joe Campbell, America's Greatest Comic Character Actor. This guy plays Belushi's brother-in-law, Andy, and he is gold. How good is he really? Well, aside from being funny, his job is to make Belushi look good. That's like trying to make butt warts look good. But Campbell pulls it off with style. Someone should invent a Nobel Prize in Comic Buffoonery so he can win it every year. Without Larry Joe this show would consist of a slightly vacant looking Courtney Thorne-Smith smacking Belushi over the head with a frying pan while he alternately beats his chest and plays with the straw on the floor of his cage. 5 stars for Larry Joe Campbell designated Comedic Bacon because he improves the flavor of everything he's in!"""
vec_list = tokenize_and_vectorize([(0, sample_1)])
test_vec_list  = pad_trunc(vec_list, max_len)
test_vec = np.reshape(test_vec_list, (len(test_vec_list), max_len, embedding_dim))
print(model.predict(test_vec))
