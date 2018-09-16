# -*- coding: utf-8 -*-

import logging
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from math import ceil

import numpy as np
from bitarray import bitarray

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'files/lab05/lab05-compression-{datetime.now().strftime("%I_%M_%S%p")}.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%m-%Y %I:%M:%S %p', level=logging.DEBUG)

from lab03_conditional_entropy import calculate_entropy
from lab01_simple_markov_gen import conver_array_to_probabilities
from lab04_fixed_length_compression import Compressor

class Metadata:
    def __init__(self, add_bits, tree):
        self.bits = add_bits
        self.tree = tree

class Node:
    def __init__(self, character, weight, leaf = False):
        self.child_left = None  # 0
        self.child_right = None  # 1
        self.character = character
        self.weight = weight
        self.leaf = leaf

    def __str__(self):
        return f'{self.character} : {self.weight}'

class HuffmanCompressor(Compressor):

    def __init__(self, filename=None, outputpath=''):
        super().__init__(filename, outputpath)

        self.alphabet = {}
        self.tree = Node('_', 0)
        self.char_to_bin = {}
        self.bin_to_char = {}
        self.words  = {}
        self.bits = {}

    def create(self):
        try:
            self.data = open(self.filename, 'r').read()
            logging.info(f'File loaded! ({self.filename})')
            super(HuffmanCompressor, self)._construct_probs_dict()
            self.__create_tree()

        except FileNotFoundError:
            logging.error(f'File not found!')


    def encode(self):
        try:
            logging.info("Compressing...")

            self.encoded_string = bitarray()
            self.words = defaultdict(int)

            for char in self.data:

                word = self.char_to_bin[char]
                self.words[word] += 1
                self.encoded_string.extend(word)

            logging.info(f"Done! ({self.encoded_string.length()} bits) ")

        except TypeError:
            logging.warning("Initialize data first!")


    def save(self, output_filename="compressed_file.bin", output_alphabet_filename='alphabet.bin'):

        num_additional_bits = self.encoded_string.fill()
        logging.info(f'Added {num_additional_bits} extra bits!')

        with open(self.outputpath + output_filename, 'wb') as file:
            self.encoded_string.tofile(file)

        with open(self.outputpath + output_alphabet_filename, 'wb') as file:
            pickle.dump(Metadata(num_additional_bits, self.tree), file)

        self.encoded_string = self.encoded_string[:-num_additional_bits]
        logging.info(f'{output_filename} & {output_alphabet_filename} has been saved.')

    def load(self, filename="compressed_file.bin", alphabet='alphabet.bin'):

        self.encoded_string = bitarray()

        logging.info(f'Loading {filename} & {alphabet}.')

        with open(self.outputpath + filename, 'rb') as file:
            self.encoded_string.fromfile(file)

        with open(self.outputpath + alphabet, 'rb') as file:
            metadata = pickle.load(file)

        self.encoded_string = self.encoded_string[:-metadata.bits]
        self.tree = metadata.tree

    def calculate_eff(self):
        fixed_length = int(ceil(np.log2(len(self.alphabet))))
        items, values = zip(*self.words.items())
        probabilities = conver_array_to_probabilities(values)

        entropy = calculate_entropy(probabilities)
        mean_val = sum([len(items[i]) * probabilities[i] for i in range(len(probabilities))]) / sum(probabilities)

        print(f"Huffman (eff): {round(entropy/mean_val * 100, 2)}%")
        print(f"Fixed-length (eff): {round(entropy/fixed_length * 100, 2)}%")

    def decode(self):

        try:
            chars = []
            bits, total_size = self.encoded_string[::-1], self.encoded_string.length()
            current_node = self.tree

            while(bits.length()):

                current_node = current_node.child_right if bits.pop() else current_node.child_left

                if current_node.leaf:
                    chars.append(current_node.character)
                    current_node = self.tree

            self.data = ''.join(chars)

            logging.info("Decoded")
        except TypeError:
            logging.warning("Initialize data first!")
        except KeyboardInterrupt:
            logging.warning(f"Keyboard interrupt! ({round((total_size - len(bits))/total_size, 2)}%)")


    def __create_tree(self):

        nodes = []
        for key in self.alphabet.keys():
            nodes.append(Node(key, self.alphabet[key], True))

        self.char_to_bin = {key : [] for key in self.alphabet.keys()}

        while(len(nodes) > 1):
            nodes.sort(key=lambda x: x.weight, reverse=True)
            new_node = Node(nodes[-2].character + nodes[-1].character, nodes[-2].weight + nodes[-1].weight)
            new_node.child_left, new_node.child_right = nodes[-2], nodes[-1]

            for char in new_node.child_left.character:
                self.char_to_bin[char].append('0')

            for char in new_node.child_right.character:
                self.char_to_bin[char].append('1')

            del nodes[-2:]
            nodes.append(new_node)

        self.tree = nodes[0]
        self.char_to_bin = {key: ''.join(value)[::-1] for key, value in self.char_to_bin.items()}
        self.bin_to_char = {value: key for key, value in self.char_to_bin.items()}


def main():
    file = 'files/lab01/norm_wiki_sample.txt' if len(sys.argv) == 1 else sys.argv[1]

    print("Compressor A... ", end=' ')
    compressor_a = HuffmanCompressor(filename=file, outputpath='files/lab05/')
    compressor_a.create()
    compressor_a.encode()
    compressor_a.save()
    print("Done!")

    print("Compressor B... ", end=' ')
    compressor_b = HuffmanCompressor(filename=file, outputpath='files/lab05/')
    compressor_b.load()
    compressor_b.decode()
    print("Done!")

    print("Result: ", end='')
    if compressor_a.data == compressor_b.data:
        print("OK!")
    else:
        print("Error!")

    compressor_a.calculate_eff()

if __name__ == '__main__':
    main()