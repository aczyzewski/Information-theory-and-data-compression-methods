import sys
from collections import Counter, defaultdict

import numpy as np
from numpy.random import choice, randint

from lab01_simple_markov_gen import get_file_content


def sum_n_counter_values(counter, top_n):
    """ Sum values of n most common counter values"""

    most_common_values = counter.most_common(top_n)
    return sum([value for _, value in most_common_values])

# --- EXERCISES ---

def exercise_1():
    """ See: exercises/lab02.pdf -> Exercise 1 """
    words = Counter(get_file_content('files/lab01/norm_wiki_sample.txt').split())
    counter_total = sum(words.values())

    first_6000_values = sum_n_counter_values(words, 6000) / counter_total
    first_30000_values = sum_n_counter_values(words, 30000) / counter_total

    return [first_6000_values, first_30000_values]

def exercise_2():
    """ See: exercises/lab02.pdf -> Exercise 2 """

    words = Counter(get_file_content('files/lab01/norm_wiki_sample.txt').split())
    values, probabilities = zip(*words.items())

    probabilities = np.array(probabilities, dtype=float)
    probabilities /= probabilities.sum()

    output_text_length = 10 ** 3
    output_text = [choice(a=values, p=probabilities) for _ in range(output_text_length)]

    return ' '.join(output_text)

def exercise_3(seed='probability', depth=5):
    """ Markov chain text generator (words).
        See: exercises/lab01.pdf -> Exercise 5 """

    output_length = 10 ** 4
    corpus = get_file_content('files/lab01/norm_wiki_sample.txt').split()

    ngrams = defaultdict(lambda: defaultdict(int))

    for word_idx in range(depth, len(corpus)):
        previous_ngram, current_word = ' '.join(corpus[word_idx - depth:word_idx]), corpus[word_idx]
        ngrams[previous_ngram][current_word] += 1

    output_text = seed.split() if seed else []

    for _ in range(output_length):

        current_ngram = ' '.join(output_text[-depth:])

        if current_ngram in ngrams.keys():

            freq_of_next_words = ngrams[current_ngram]

            words, probabilities = zip(*freq_of_next_words.items())
            probabilities = np.array(probabilities, dtype=float)
            probabilities /= probabilities.sum()

            output_text.append(choice(a=words, p=probabilities))

        else:
            output_text.append(corpus[randint(len(corpus))])

    return ' '.join(output_text)

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print('Usage: %s [exercise_id]' % sys.argv[0])
        sys.exit(2)
        
    try:
        print(globals()['exercise_{}'.format(sys.argv[1])]())
    except:
        print("Err: Exercise not found!")