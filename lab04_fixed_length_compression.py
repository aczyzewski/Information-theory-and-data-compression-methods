# -*- coding: utf-8 -*-

import logging
import sys
from collections import Counter
from datetime import datetime
from math import ceil

import numpy as np
from bitarray import bitarray

from lab01_simple_markov_gen import conver_array_to_probabilities

logger = logging.getLogger(__name__)
logging.basicConfig(filename=f'files/lab04/lab04-compression-{datetime.now().strftime("%I_%M_%S%p")}.log', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%m-%Y %I:%M:%S %p', level=logging.DEBUG)

class Compressor:

    def __init__(self, filename=None, outputpath=''):
        self.filename = filename
        self.outputpath = outputpath

    def create(self):
        try:

            self.data = open(self.filename, 'r').read()
            logging.info(f'File loaded! ({self.filename})')

            self.__construct_probs_dict()
            self.fixed_length = int(ceil(np.log2(len(self.alphabet))))
            logging.info(f'Calculated fixed-length: {self.fixed_length}')

            self.char_to_bin = {character: f'{code:b}'.zfill(self.fixed_length) for code, character in
                                enumerate(self.alphabet.keys())}

            self.bin_to_char = {value: key for key, value in self.char_to_bin.items()}

            logging.info(f'Construced map:\n {self.char_to_bin}')

        except FileNotFoundError:
            logging.error(f'File not found!')


    def encode(self):
        try:
            logging.info("Encoding data...")

            self.encoded_string = bitarray()

            for char in self.data:
                self.encoded_string.extend(self.char_to_bin[char])

            logging.info(f"File encoded!  ({self.encoded_string.length()} bits.)")

        except TypeError:
            logging.error("Load file first.")


    def save(self, output_filename="compressed_file.bin", output_alphabet_filename='alphabet.bin'):

        num_additional_bits = self.encoded_string.fill()
        logging.info(f'Added {num_additional_bits} extra bits to encoded file!')

        with open(self.outputpath + output_filename, 'wb') as file:
            self.encoded_string.tofile(file)

        chartobin = self.__chartobin_dict_to_bitarray()
        num_additional_bits = self.encoded_string.fill()

        logging.info(f'Added {num_additional_bits} extra bits to encoded map!')

        with open(self.outputpath + output_alphabet_filename, 'wb') as alpabet_file:
            chartobin.tofile(alpabet_file)

        logging.info(f'{output_filename} & {output_alphabet_filename} has been saved.')

    def load(self, filename="compressed_file.bin", alphabet='alphabet.bin'):

        self.encoded_string = bitarray()
        encoded_alphabet = bitarray()
        logging.info(f'Reading files: {filename} & {alphabet}.')

        with open(self.outputpath + filename, 'rb') as file:
            self.encoded_string.fromfile(file)

        with open(self.outputpath + alphabet, 'rb') as alphabet_file:
            encoded_alphabet.fromfile(alphabet_file)

        self.fixed_length, self.char_to_bin = self.__bitarray_to_chartobin_dict(to_decode=encoded_alphabet)
        self.bin_to_char = {value: key for key, value in self.char_to_bin.items()}

        logging.info(f' -> Fixed-lenght: {self.fixed_length}.')
        logging.info(f' -> Map:\n {self.char_to_bin}.')

    def decode(self):

        thresholds = [i for i in range(5, 101, 5)]
        current_threshold = 0

        try:
            chars = []
            total_size = self.encoded_string.length()
            num_additional_bits = total_size % self.fixed_length
            logging.info(f'Decoding. ({num_additional_bits} extra bits)')

            bits = self.encoded_string[:total_size - num_additional_bits][::-1]
            num_bits = len(bits)

            for _ in range((total_size - num_additional_bits) // self.fixed_length):
                temp = ""
                for _ in range(self.fixed_length):
                    temp += '1' if bits.pop() else '0'
                chars.append(self.bin_to_char[temp])

                if round((num_bits - len(bits))/num_bits, 2) * 100 > thresholds[current_threshold]:
                    logging.info(f" ... {thresholds[current_threshold]}%")
                    current_threshold += 1

            self.data = ''.join(chars)

            logging.info("Done!")

        except TypeError:
            logging.warning("Corrupted data.")
        except KeyboardInterrupt:
            logging.warning(f"Keyboard Interrupt! ({round((num_bits - len(bits))/num_bits, 2)}%)")


    def _construct_probs_dict(self):

        collector = Counter(self.data)
        keys, values = zip(*collector.items())
        probabilities = conver_array_to_probabilities(values)

        result = dict(zip(keys, probabilities))
        self.alphabet = dict(sorted(result.items(), key=lambda x : x[1], reverse=True))

    def _chartobin_dict_to_bitarray(self):
        output_bitarray = bitarray(f'{self.fixed_length:b}'.zfill(8))
        for key in self.char_to_bin:
            output_bitarray.extend(f'{ord(key):b}'.zfill(8) + f'{self.char_to_bin[key]}')

        return output_bitarray

    def _bitarray_to_chartobin_dict(self, to_decode):
        to_decode = to_decode.to01()
        size_string, alphabet_string = to_decode[:8], to_decode[8:]
        fixed_length = int(size_string, 2)
        values = [alphabet_string[i :i + fixed_length + 8] for i in range(0, len(alphabet_string), fixed_length + 8)]
        return fixed_length, {chr(int(value[:8], 2)) : value[8:] for value in values if len(value) == fixed_length + 8}

def main():

    file = 'files/lab01/norm_wiki_sample.txt' if len(sys.argv) == 1 else sys.argv[1]

    print("Compressor A... ", end=' ')
    compressor_a = Compressor(filename=file, outputpath='files/lab04/')
    compressor_a.create()
    compressor_a.encode()
    compressor_a.save()
    print("Done!")

    print("Compressor B... ", end=' ')
    compressor_b = Compressor(filename=file, outputpath='files/lab04/')
    compressor_b.load()
    compressor_b.decode()
    print("Done!")

    print("Result: ", end='')
    if compressor_a.data == compressor_b.data:
        print("OK!")
    else:
        print("Error!")


if __name__ == '__main__':
    main()