import sys

from collections import Counter
from collections import defaultdict

import numpy as np
from numpy.random import randint, choice


def average_word_length(text):
    """ Return average word length based on text """

    words = text.split()
    return sum([len(word) for word in words]) / len(words)

def english_alphabet_generator(letters=True, space=True, numbers=False):
    """ Return a list of letters """

    characters = [' '] if space else []
    characters += [chr(i) for i in range(ord('a'), ord('z') + 1)] if letters else []
    characters += [chr(i) for i in range(ord('0'), ord('9') + 1)] if numbers else []

    return characters

def conver_array_to_probabilities(values):
    probabilities = np.array(values, dtype=float)
    probabilities /= probabilities.sum()
    return probabilities

def get_file_content(filename, limit=None):
    """ Return content (limited) of provided file """

    file = open(filename, 'r')
    return file.read(limit)

# --- EXERCISES ---

def exercise_1():
    """ See: exercises/lab01.pdf -> Exercise 1 """

    text_length = 10 ** 7
    alphabet = english_alphabet_generator()
    generated_text = [alphabet[randint(len(alphabet))] for _ in range(text_length)]
    return average_word_length(''.join(generated_text))

def exercise_2(corpus=None):
    """ See: exercises/lab01.pdf -> Exercise 2 """

    filename = 'files/lab01/norm_wiki_sample.txt'
    alphabet = english_alphabet_generator()

    letters = Counter(corpus) if corpus else Counter(get_file_content(filename))

    filterd_letters = {letter: value for letter, value in letters.items() if letter in alphabet}
    in_total = sum(filterd_letters.values())

    letter_probability = {letter: value / in_total for letter, value in filterd_letters.items()}
    sorted_letters = dict(sorted(letter_probability.items(), key=lambda item: item[1], reverse=True))

    return sorted_letters

def exercise_3():
    """ See: exercises/lab01.pdf -> Exercise 3 """

    text_length = 10 ** 6
    freq_of_letters = exercise_2()
    letters, values = zip(*freq_of_letters.items())
    probabilities = conver_array_to_probabilities(values)

    generated_text = [choice(a=letters, p=probabilities) for _ in range(text_length)]
    return average_word_length(''.join(generated_text))

def exercise_4():
    """ See: exercises/lab01.pdf -> Exercise 4 """

    corpus = get_file_content('files/lab01/norm_wiki_sample.txt')
    freq_of_letters = exercise_2(corpus)

    top_letters = list(freq_of_letters)[:2]
    bigrams = {key: defaultdict(int) for key in top_letters}

    for letter_idx in range(1, len(corpus)):
        if corpus[letter_idx - 1] in top_letters:
            previous_letter, current_letter = corpus[letter_idx - 1], corpus[letter_idx]
            bigrams[previous_letter][current_letter] += 1

    return bigrams

def exercise_5(seed='', depth=1):
    """ Markov chain text generator (letters).
        See: exercises/lab01.pdf -> Exercise 5 """

    output_length = 10 ** 3

    corpus = get_file_content('files/lab01/norm_wiki_sample.txt')
    ngrams = defaultdict(lambda: defaultdict(int))

    for letter_idx in range(depth, len(corpus)):
        previous_ngram, current_letter = corpus[letter_idx - depth:letter_idx], corpus[letter_idx]
        ngrams[previous_ngram][current_letter] += 1

    output_text = list(seed) if seed else []
    alphabet = english_alphabet_generator()

    for _ in range(output_length):

        current_ngram = ''.join(output_text[-depth:])
        if current_ngram in ngrams.keys():

            freq_of_next_letters = ngrams[current_ngram]

            letters, values = zip(*freq_of_next_letters.items())
            probabilities = conver_array_to_probabilities(values)

            output_text.append(choice(a=letters, p=probabilities))

        else:
            output_text.append(alphabet[randint(len(alphabet))])

    return ''.join(output_text)


if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print('Usage: %s [exercise_id]' % sys.argv[0])
        sys.exit(2)
    try:
        print(globals()['exercise_{}'.format(sys.argv[1])]())

    except:
        print("Err: Exercise not found!")