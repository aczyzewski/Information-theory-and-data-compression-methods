import sys

from collections import Counter
from collections import defaultdict

import numpy as np

from math import log
from lab01_simple_markov_gen import english_alphabet_generator, \
    get_file_content, conver_array_to_probabilities

def calculate_entropy(probabilities, log_base=2):
    return -1 * sum([probabilities[i] * log(probabilities[i], log_base) for i in range(len(probabilities))])

def calculate_conditional_entropy_on_file(file, depth, words=False):

    depth += 1

    # Read file content and transform it to list of chars/words.
    file_content = list(get_file_content(file)) if not words else get_file_content(file).split()

    # Prepare counters for every possible ngram
    ngram_counter = defaultdict(int)
    base_counter = defaultdict(int)

    # Count ngrams and their prefixes.
    for item_idx in range(depth, len(file_content)):
        if words:
            ngram_counter[' '.join(file_content[item_idx - depth:item_idx])] += 1
            base_counter[' '.join(file_content[item_idx - depth:item_idx - 1])] += 1
        else:
            ngram_counter[''.join(file_content[item_idx - depth:item_idx])] += 1
            base_counter[''.join(file_content[item_idx - depth:item_idx - 1])] += 1

    num_engrams = len(file_content) - depth

    # Calculate entropy
    entropy = 0
    for ngram in ngram_counter:
        prob_of_ngram = ngram_counter[ngram] / num_engrams
        cond_prob = ngram_counter[ngram] / base_counter[' '.join(ngram.split()[:-1])] if words else ngram_counter[ngram] / base_counter[ngram[:-1]]
        entropy -= prob_of_ngram * np.log2(cond_prob)

    return entropy

# --- EXERCISES ---

def exercise_1():
    """ See: exercises/lab03.pdf -> Exercise 1 """

    length_of_the_alphabet = len(english_alphabet_generator(numbers=True))
    probabities = [1/length_of_the_alphabet] * length_of_the_alphabet
    pure_alphabet_entropy = calculate_entropy(probabities)

    wiki_file_counter = Counter(get_file_content('files/lab01/norm_wiki_sample.txt'))
    _, values = zip(*wiki_file_counter.most_common(None))
    wiki_probabilities = conver_array_to_probabilities(values)
    wiki_alphabet_entropy = calculate_entropy(wiki_probabilities)

    return (pure_alphabet_entropy, wiki_alphabet_entropy)

def exercise_2(file='files/lab03/norm_wiki_la.txt', depth=5, words=False):
    """ See: exercises/lab03.pdf -> Exercise 2/3 """
    return calculate_conditional_entropy_on_file(file, depth, words)

if __name__ == '__main__':
    if (len(sys.argv) < 2):
        print('Usage: %s [exercise_id]' % sys.argv[0])
        sys.exit(2)
    try:
        print(globals()['exercise_{}'.format(sys.argv[1])]())

    except:
        print("Err: Exercise not found!")