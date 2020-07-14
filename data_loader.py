import bz2
from collections import Counter
import re
import nltk
import numpy as np
import pickle


def create_data():
    nltk.download('punkt')

    train_file = bz2.BZ2File('./train.ft.txt.bz2')
    test_file = bz2.BZ2File('./test.ft.txt.bz2')

    train_file = train_file.readlines()
    test_file = test_file.readlines()

    print("Number of training reivews: " + str(len(train_file)))
    print("Number of test reviews: " + str(len(test_file)))

    num_train = 800000  # We're training on the first 800,000 reviews in the dataset
    num_test = 200000  # Using 200,000 reviews from test set

    train_file = [x.decode('utf-8') for x in train_file[:num_train]]
    test_file = [x.decode('utf-8') for x in test_file[:num_test]]

    print(train_file[0])

    # Extracting labels from sentences

    import spacy
    sp = spacy.load('en_core_web_sm')

    train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file]
    train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file]

    test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file]
    test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file]

    # Some simple cleaning of data

    for i in range(len(train_sentences)):
        train_sentences[i] = re.sub('\d', '0', train_sentences[i])

    for i in range(len(test_sentences)):
        test_sentences[i] = re.sub('\d', '0', test_sentences[i])

    # Modify URLs to <url>
    for i in range(len(train_sentences)):
        if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in \
                train_sentences[i]:
            train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])

    for i in range(len(test_sentences)):
        if 'www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i] or '.com' in \
                test_sentences[i]:
            test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])

    train_pos = []

    del train_file, test_file

    words = Counter()  # Dictionary that will map a word to the number of times it appeared in all the training sentences
    for i, sentence in enumerate(train_sentences):
        # The sentences will be stored as a list of words/tokens
        train_sentences[i] = []
        pos_taggs = []
        sen = sp(sentence)
        for word in sen:  # Tokenizing the words
            words.update([word.text.lower()])  # Converting all the words to lower case
            train_sentences[i].append(word.text)
            pos_taggs.append(word.pos_)
        train_pos.append(pos_taggs)

        if i % 20000 == 0:
            print(str((i * 100) / num_train) + "% done")
    print("100% done")

    # Removing the words that only appear once
    words = {k: v for k, v in words.items() if v > 1}
    # Sorting the words according to the number of appearances, with the most common word being first
    words = sorted(words, key=words.get, reverse=True)
    # Adding padding and unknown to our vocabulary so that they will be assigned an index
    words = ['_PAD', '_UNK'] + words
    # Dictionaries to store the word to index mappings and vice versa
    word2idx = {o: i for i, o in enumerate(words)}
    idx2word = {i: o for i, o in enumerate(words)}

    for i, sentence in enumerate(train_sentences):
        # Looking up the mapping dictionary and assigning the index to the respective words
        train_sentences[i] = [word2idx[word] if word in word2idx else 1 for word in sentence]

    test_pos = []
    for i, sentence in enumerate(test_sentences):
        sen = sp(sentence)
        # For test sentences, we have to tokenize the sentences as well
        test_sentences[i] = [word2idx[word.text.lower()] if word.text.lower() in word2idx else 0 for word in sen]
        test_pos.append([word.pos_ for word in sen])

    def pad_input(sentences, seq_len):
        features = np.zeros((len(sentences), seq_len), dtype=int)
        for ii, review in enumerate(sentences):
            if len(review) != 0:
                features[ii, -len(review):] = np.array(review)[:seq_len]
        return features

    seq_len = 200  # The length that the sentences will be padded/shortened to

    train_sentences = pad_input(train_sentences, seq_len)
    test_sentences = pad_input(test_sentences, seq_len)

    train_pos = numerize_pos_tags(train_pos, seq_len)
    test_pos = numerize_pos_tags(test_pos, seq_len)

    # Converting our labels into numpy arrays
    train_labels = np.array(train_labels)
    test_labels = np.array(test_labels)

    #split_frac = 0.5
    #split_id = int(split_frac * len(test_sentences))
    #val_sentences, test_sentences = test_sentences[:split_id], test_sentences[split_id:]
    #val_labels, test_labels = test_labels[:split_id], test_labels[split_id:]

    return train_sentences, train_pos, train_labels, test_sentences, test_pos, test_labels, word2idx


all_pos_tags = [
        'ADJ',
        'ADP',
        'ADV',
        'AUX',
        'CONJ',
        'CCONJ',
        'DET',
        'INTJ',
        'NOUN',
        'NUM',
        'PART',
        'PRON',
        'PROPN',
        'PUNCT',
        'SCONJ',
        'SYM',
        'VERB',
        'X',
        'SPACE'
    ]

def numerize_pos_tags(pos_tags, seq_len):

    for i, tag_list in enumerate(pos_tags):
        #for ii, tag in enumerate(tag_list):
        #    if tag in tags:
        #        tag_list[i] = tags.index(tag)
        #    else:
        #        tag_list[i] = tags.index('X')

        pos_tags[i] = [all_pos_tags.index(tag) if tag in all_pos_tags else all_pos_tags.index('X') for tag in tag_list]

    features = np.zeros((len(pos_tags), seq_len), dtype=int)
    for ii, review in enumerate(pos_tags):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

def save_data(data, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file, protocol=4)


def load_data(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)
