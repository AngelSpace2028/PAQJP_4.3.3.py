import os
import sys
import math
import struct
import array
import random
import heapq
import binascii
import logging
import paq
import zlib
from typing import List, Dict, Tuple, Optional, Union
from enum import Enum
from mpmath import mp

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
PROGNAME = "PAQJP_4"
DEFAULT_OPTION = 9
MEM = 1 << 15  # 32,768
HUFFMAN_THRESHOLD = 1024  # Bytes threshold for Huffman vs. zlib/PAQ compression

# Generate 1,000,000 base-10 digits of pi, mapped to 0-255
def generate_pi_digits(num_digits: int = 1000000) -> List[int]:
    """Generate base-10 pi digits, mapped to 0-255, without saving to file."""
    try:
        mp.dps = num_digits + 2  # Extra precision to ensure enough digits
        pi_str = mp.nstr(mp.pi, num_digits, strip_zeros=False, min_fixed=0, max_fixed=0)
        # Extract digits after decimal point, skipping '3.'
        pi_digits = [int(d) for d in pi_str[2:] if d.isdigit()][:num_digits]
        if len(pi_digits) < num_digits:
            logging.warning(f"Generated only {len(pi_digits)} digits, padding with fallback")
            # Pad with repeating [3, 1, 4] if insufficient digits
            fallback_digits = [3, 1, 4]
            pi_digits.extend(fallback_digits * ((num_digits - len(pi_digits)) // len(fallback_digits) + 1))
            pi_digits = pi_digits[:num_digits]
        if not all(0 <= d <= 9 for d in pi_digits):
            logging.error("Generated pi digits contain invalid values")
            raise ValueError("Invalid pi digits generated")
        mapped_digits = [(d * 255 // 9) % 256 for d in pi_digits]
        logging.info(f"Generated {len(mapped_digits)} base-10 pi digits (mapped to 0-255)")
        return mapped_digits
    except Exception as e:
        logging.error(f"Failed to generate base-10 pi digits: {e}")
        # Fallback to repeating [3, 1, 4]
        fallback_digits = [3, 1, 4]
        mapped_fallback = [(d * 255 // 9) % 256 for d in fallback_digits]
        logging.warning(f"Using {len(mapped_fallback)} fallback base-10 digits, repeating to {num_digits}")
        return mapped_fallback * (num_digits // len(fallback_digits) + 1)[:num_digits]

PI_DIGITS = generate_pi_digits(1000000)

# Prime numbers for transformation
PRIMES = [p for p in range(2, 256) if all(p % d != 0 for d in range(2, int(p**0.5)+1))]

class Filetype(Enum):
    DEFAULT = 0
    JPEG = 1
    EXE = 2
    TEXT = 3

class Mode(Enum):
    COMPRESS = 0
    DECOMPRESS = 1

class String:
    def __init__(self, s: str = ""):
        self.data = bytearray(s.encode('utf-8'))
    
    def resize(self, new_size: int):
        if new_size > len(self.data):
            self.data += bytearray(new_size - len(self.data))
        else:
            self.data = self.data[:new_size]
    
    def size(self) -> int:
        return len(self.data)
    
    def c_str(self) -> str:
        return self.data.decode('utf-8')
    
    def __iadd__(self, s: str):
        self.data += s.encode('utf-8')
        return self
    
    def __getitem__(self, index: int) -> int:
        return self.data[index]
    
    def __setitem__(self, index: int, value: int):
        self.data[index] = value
    
    def __str__(self) -> str:
        return self.data.decode('utf-8')

class Array:
    def __init__(self, size: int = 0, initial_value: int = 0):
        self.data = array.array('B', [initial_value] * size)
    
    def resize(self, new_size: int):
        if new_size > len(self.data):
            self.data.extend([0] * (new_size - len(self.data)))
        else:
            self.data = self.data[:new_size]
    
    def size(self) -> int:
        return len(self.data)
    
    def __getitem__(self, index: int) -> int:
        return self.data[index]
    
    def __setitem__(self, index: int, value: int):
        self.data[index] = value
    
    def __len__(self) -> int:
        return len(self.data)

class Buf:
    def __init__(self, size: int = 0):
        self.size_ = size
        self.data = Array(size)
        self.pos = 0
    
    def setsize(self, size: int):
        if size > 0 and (size & (size - 1)) == 0:
            self.size_ = size
            self.data.resize(size)
    
    def __getitem__(self, index: int) -> int:
        return self.data[index & (self.size_ - 1)]
    
    def __call__(self, i: int) -> int:
        assert i > 0
        return self.data[(self.pos - i) & (self.size_ - 1)]
    
    def size(self) -> int:
        return self.size_

buf = Buf()

class Node:
    def __init__(self, left=None, right=None, symbol=None):
        self.left = left
        self.right = right
        self.symbol = symbol

    def is_leaf(self):
        return self.left is None and self.right is None

class StateTable:
    def __init__(self):
        self.table = [
            [1, 2, 0, 0], [3, 5, 1, 0], [4, 6, 0, 1], [7, 10, 2, 0],
            [8, 12, 1, 1], [9, 13, 1, 1], [11, 14, 0, 2], [15, 19, 3, 0],
            [16, 23, 2, 1], [17, 24, 2, 1], [18, 25, 2, 1], [20, 27, 1, 2],
            [21, 28, 1, 2], [22, 29, 1, 2], [26, 30, 0, 3], [31, 33, 4, 0],
            [32, 35, 3, 1], [32, 35, 3, 1], [32, 35, 3, 1], [32, 35, 3, 1],
            [34, 37, 2, 2], [34, 37, 2, 2], [34, 37, 2, 2], [34, 37, 2, 2],
            [34, 37, 2, 2], [34, 37, 2, 2], [36, 39, 1, 3], [36, 39, 1, 3],
            [36, 39, 1, 3], [36, 39, 1, 3], [38, 40, 0, 4], [41, 43, 5, 0],
            [42, 45, 4, 1], [42, 45, 4, 1], [44, 47, 3, 2], [44, 47, 3, 2],
            [46, 49, 2, 3], [46, 49, 2, 3], [48, 51, 1, 4], [48, 51, 1, 4],
            [50, 52, 0, 5], [53, 43, 6, 0], [54, 57, 5, 1], [54, 57, 5, 1],
            [56, 59, 4, 2], [56, 59, 4, 2], [58, 61, 3, 3], [58, 61, 3, 3],
            [60, 63, 2, 4], [60, 63, 2, 4], [62, 65, 1, 5], [62, 65, 1, 5],
            [50, 66, 0, 6], [67, 55, 7, 0], [68, 57, 6, 1], [68, 57, 6, 1],
            [70, 73, 5, 2], [70, 73, 5, 2], [72, 75, 4, 3], [72, 75, 4, 3],
            [74, 77, 3, 4], [74, 77, 3, 4], [76, 79, 2, 5], [76, 79, 2, 5],
            [62, 81, 1, 6], [62, 81, 1, 6], [64, 82, 0, 7], [83, 69, 8, 0],
            [84, 76, 7, 1], [84, 76, 7, 1], [86, 73, 6, 2], [86, 73, 6, 2],
            [44, 59, 5, 3], [44, 59, 5, 3], [58, 61, 4, 4], [58, 61, 4, 4],
            [60, 49, 3, 5], [60, 49, 3, 5], [76, 89, 2, 6], [76, 89, 2, 6],
            [78, 91, 1, 7], [78, 91, 1, 7], [80, 92, 0, 8], [93, 69, 9, 0],
            [94, 87, 8, 1], [94, 87, 8, 1], [96, 45, 7, 2], [96, 45, 7, 2],
            [48, 99, 2, 7], [48, 99, 2, 7], [88, 101, 1, 8], [88, 101, 1, 8],
            [80, 102, 0, 9], [103, 69, 10, 0], [104, 87, 9, 1], [104, 87, 9, 1],
            [106, 57, 8, 2], [106, 57, 8, 2], [62, 109, 2, 8], [62, 109, 2, 8],
            [88, 111, 1, 9], [88, 111, 1, 9], [80, 112, 0, 10], [113, 85, 11, 0],
            [114, 87, 10, 1], [114, 87, 10, 1], [116, 57, 9, 2], [116, 57, 9, 2],
            [62, 119, 2, 9], [62, 119, 2, 9], [88, 121, 1, 10], [88, 121, 1, 10],
            [90, 122, 0, 11], [123, 85, 12, 0], [124, 97, 11, 1], [124, 97, 11, 1],
            [126, 57, 10, 2], [126, 57, 10, 2], [62, 129, 2, 10], [62, 129, 2, 10],
            [98, 131, 1, 11], [98, 131, 1, 11], [90, 132, 0, 12], [133, 85, 13, 0],
            [134, 97, 12, 1], [134, 97, 12, 1], [136, 57, 11, 2], [136, 57, 11, 2],
            [62, 139, 2, 11], [62, 139, 2, 11], [98, 141, 1, 12], [98, 141, 1, 12],
            [90, 142, 0, 13], [143, 95, 14, 0], [144, 97, 13, 1], [144, 97, 13, 1],
            [68, 57, 12, 2], [68, 57, 12, 2], [62, 81, 2, 12], [62, 81, 2, 12],
            [98, 147, 1, 13], [98, 147, 1, 13], [100, 148, 0, 14], [149, 95, 15, 0],
            [150, 107, 14, 1], [150, 107, 14, 1], [108, 151, 1, 14], [108, 151, 1, 14],
            [100, 152, 0, 15], [153, 95, 16, 0], [154, 107, 15, 1], [108, 155, 1, 15],
            [100, 156, 0, 16], [157, 95, 17, 0], [158, 107, 16, 1], [108, 159, 1, 16],
            [100, 160, 0, 17], [161, 105, 18, 0], [162, 107, 17, 1], [108, 163, 1, 17],
            [110, 164, 0, 18], [165, 105, 19, 0], [166, 117, 18, 1], [118, 167, 1, 18],
            [110, 168, 0, 19], [169, 105, 20, 0], [170, 117, 19, 1], [118, 171, 1, 19],
            [110, 172, 0, 20], [173, 105, 21, 0], [174, 117, 20, 1], [118, 175, 1, 20],
            [110, 176, 0, 21], [177, 105, 22, 0], [178, 117, 21, 1], [118, 179, 1, 21],
            [120, 184, 0, 23], [185, 115, 24, 0], [186, 127, 23, 1], [128, 187, 1, 23],
            [120, 188, 0, 24], [189, 115, 25, 0], [190, 127, 24, 1], [128, 191, 1, 24],
            [120, 192, 0, 25], [193, 115, 26, 0], [194, 127, 25, 1], [128, 195, 1, 25],
            [120, 196, 0, 26], [197, 115, 27, 0], [198, 127, 26, 1], [128, 199, 1, 26],
            [120, 200, 0, 27], [201, 115, 28, 0], [202, 127, 27, 1], [128, 203, 1, 27],
            [120, 204, 0, 28], [205, 115, 29, 0], [206, 127, 28, 1], [128, 207, 1, 28],
            [120, 208, 0, 29], [209, 125, 30, 0], [210, 127, 29, 1], [128, 211, 1, 29],
            [130, 212, 0, 30], [213, 125, 31, 0], [214, 137, 30, 1], [138, 215, 1, 30],
            [130, 216, 0, 31], [217, 125, 32, 0], [218, 137, 31, 1], [138, 219, 1, 31],
            [130, 220, 0, 32], [221, 125, 33, 0], [222, 137, 32, 1], [138, 223, 1, 32],
            [130, 224, 0, 33], [225, 125, 34, 0], [226, 137, 33, 1], [138, 227, 1, 33],
            [130, 228, 0, 34], [229, 125, 35, 0], [230, 137, 34, 1], [138, 231, 1, 34],
            [130, 232, 0, 35], [233, 125, 36, 0], [234, 137, 35, 1], [138, 235, 1, 35],
            [130, 236, 0, 36], [237, 125, 37, 0], [238, 137, 36, 1], [138, 239, 1, 36],
            [130, 240, 0, 37], [241, 125, 38, 0], [242, 137, 37, 1], [138, 243, 1, 37],
            [130, 244, 0, 38], [245, 135, 39, 0], [246, 137, 38, 1], [138, 247, 1, 38],
            [140, 248, 0, 39], [249, 135, 40, 0], [250, 69, 39, 1], [80, 251, 1, 39],
            [140, 252, 0, 40], [249, 135, 41, 0], [250, 69, 40, 1], [80, 251, 1, 40],
            [140, 252, 0, 41]
        ]
    
    def nex(self, state: int, sel: int) -> int:
        return self.table[state][sel]

nex = StateTable()

def transform_with_prime_xor_every_3_bytes(data, repeat=7):
    transformed = bytearray(data)
    for prime in PRIMES:
        xor_val = prime if prime == 2 else max(1, math.ceil(prime * 4096 / 28672))
        for _ in range(repeat):
            for i in range(0, len(transformed), 3):
                transformed[i] ^= xor_val
    return bytes(transformed)

def transform_with_pattern_chunk(data, chunk_size=4):
    transformed = bytearray()
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        transformed.extend([b ^ 0xFF for b in chunk])
    return bytes(transformed)

def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(n ** 0.5) + 1, 2):
        if n % i == 0:
            return False
    return True

def find_nearest_prime_around(n):
    offset = 0
    while True:
        if is_prime(n - offset):
            return n - offset
        if is_prime(n + offset):
            return n + offset
        offset += 1

def quit(message: str = None):
    if message:
        print(message)
    sys.exit(1)

def ilog(x: int) -> int:
    if x < 0:
        return 0
    l = 0
    while x > 0:
        x >>= 1
        l += 1
    return l

def squash(d: int, n: int = 12, repeat: int = 1000) -> int:
    max_output = (1 << n) - 1
    result = d
    for _ in range(repeat):
        if result > 2047:
            result = max_output
        if result < -2047:
            result = 0
        scaled = (1 << n) / (1 + math.exp(-result / 512.0))
        result = int(scaled)
        result = min(max(result, 0), max_output)
    return result

def stretch(p: int) -> int:
    t = Array(4096)
    pi = 0
    for x in range(-2047, 2048):
        i = squash(x)
        for j in range(pi, i + 1):
            t[j] = x
        pi = i + 1
    t[4095] = 2047
    return t[p]

def hash(*args: int) -> int:
    h = (args[0] * 200002979 + args[1] * 30005491 + 
         (args[2] if len(args) > 2 else 0xffffffff) * 50004239 + 
         (args[3] if len(args) > 3 else 0xffffffff) * 70004807 + 
         (args[4] if len(args) > 4 else 0xffffffff) * 110002499)
    return h ^ (h >> 9) ^ (args[0] >> 2) ^ (args[1] >> 3) ^ (
        (args[2] if len(args) > 2 else 0) >> 4) ^ (
        (args[3] if len(args) > 3 else 0) >> 5) ^ (
        (args[4] if len(args) > 4 else 0) >> 6)

class SmartCompressor:
    def __init__(self):
        self.compressor = None
        self.PI_DIGITS = PI_DIGITS
        self.PRIMES = PRIMES
        self.seed_tables = self.generate_seed_tables()
        self.max_intersections = 28

    def generate_seed_tables(self, num_tables=126, table_size=256, min_val=5, max_val=255, seed=42):
        random.seed(seed)
        tables = []
        for _ in range(num_tables):
            table = [random.randint(min_val, max_val) for _ in range(table_size)]
            tables.append(table)
        return tables
    
    def get_seed(self, table_idx: int, value: int) -> int:
        if 0 <= table_idx < len(self.seed_tables):
            return self.seed_tables[table_idx][value % len(self.seed_tables[table_idx])]
        return 0
    
    def binary_to_file(self, binary_data, filename):
        try:
            n = int(binary_data, 2)
            num_bytes = (len(binary_data) + 7) // 8
            hex_str = "%0*x" % (num_bytes * 2, n)
            if len(hex_str) % 2 != 0:
                hex_str = '0' + hex_str
            byte_data = binascii.unhexlify(hex_str)
            with open(filename, 'wb') as f:
                f.write(byte_data)
            return True
        except Exception as e:
            logging.error(f"Error saving file: {str(e)}")
            return False

    def file_to_binary(self, filename):
        try:
            with open(filename, 'rb') as f:
                data = f.read()
                if not data:
                    logging.error("Error: Empty file")
                    return None
                binary_str = bin(int(binascii.hexlify(data), 16))[2:]
                return binary_str.zfill(len(data) * 8)
        except Exception as e:
            logging.error(f"Error reading file: {str(e)}")
            return None

    def calculate_frequencies(self, binary_str):
        frequencies = {}
        for bit in binary_str:
            frequencies[bit] = frequencies.get(bit, 0) + 1
        return frequencies

    def build_huffman_tree(self, frequencies):
        heap = [(freq, Node(symbol=symbol)) for symbol, freq in frequencies.items()]
        heapq.heapify(heap)
        while len(heap) > 1:
            freq1, node1 = heapq.heappop(heap)
            freq2, node2 = heapq.heappop(heap)
            new_node = Node(left=node1, right=node2)
            heapq.heappush(heap, (freq1 + freq2, new_node))
        return heap[0][1]

    def generate_huffman_codes(self, root, current_code="", codes={}):
        if root.is_leaf():
            codes[root.symbol] = current_code or "0"
            return codes
        if root.left:
            self.generate_huffman_codes(root.left, current_code + "0", codes)
        if root.right:
            self.generate_huffman_codes(root.right, current_code + "1", codes)
        return codes

    def compress_data_huffman(self, binary_str):
        if not binary_str:
            return ""
        frequencies = self.calculate_frequencies(binary_str)
        huffman_tree = self.build_huffman_tree(frequencies)
        huffman_codes = self.generate_huffman_codes(huffman_tree)
        if '0' not in huffman_codes:
            huffman_codes['0'] = '0'
        if '1' not in huffman_codes:
            huffman_codes['1'] = '1'
        compressed_str = ''.join(huffman_codes[bit] for bit in binary_str)
        return compressed_str

    def decompress_data_huffman(self, compressed_str):
        if not compressed_str:
            return ""
        frequencies = self.calculate_frequencies(compressed_str)
        huffman_tree = self.build_huffman_tree(frequencies)
        huffman_codes = self.generate_huffman_codes(huffman_tree)
        reversed_codes = {code: symbol for symbol, code in huffman_codes.items()}
        decompressed_str = ""
        current_code = ""
        for bit in compressed_str:
            current_code += bit
            if current_code in reversed_codes:
                decompressed_str += reversed_codes[current_code]
                current_code = ""
        return decompressed_str

    def compress_data_zlib(self, data_bytes):
        try:
            return zlib.compress(data_bytes)
        except zlib.error as e:
            logging.error(f"zlib compression error: {e}")
            return None

    def decompress_data_zlib(self, compressed_data):
        try:
            return zlib.decompress(compressed_data)
        except zlib.error as e:
            logging.error(f"zlib decompression error: {e}")
            return None

    def paq_compress(self, data):
        return paq.compress(data)
    
    def paq_decompress(self, data):
        return paq.decompress(data)

    def transform_01(self, data):
        return transform_with_prime_xor_every_3_bytes(data, repeat=7)
    
    def reverse_transform_01(self, data):
        return self.transform_01(data)
    
    def transform_03(self, data):
        return transform_with_pattern_chunk(data)
    
    def reverse_transform_03(self, data):
        return self.transform_03(data)
    
    def transform_04(self, data, repeat=50):
        transformed = bytearray(data)
        for _ in range(repeat):
            for i in range(len(transformed)):
                transformed[i] = (transformed[i] - (i % 256)) % 256
        return bytes(transformed)
    
    def reverse_transform_04(self, data, repeat=50):
        transformed = bytearray(data)
        for _ in range(repeat):
            for i in range(len(transformed)):
                transformed[i] = (transformed[i] + (i % 256)) % 256
        return bytes(transformed)
    
    def transform_05(self, data, shift=3):
        transformed = bytearray(data)
        for i in range(len(transformed)):
            transformed[i] = ((transformed[i] << shift) | (transformed[i] >> (8 - shift))) & 0xFF
        return bytes(transformed)
    
    def reverse_transform_05(self, data, shift=3):
        transformed = bytearray(data)
        for i in range(len(transformed)):
            transformed[i] = ((transformed[i] >> shift) | (transformed[i] << (8 - shift))) & 0xFF
        return bytes(transformed)
    
    def transform_06(self, data, seed=42):
        random.seed(seed)
        substitution = list(range(256))
        random.shuffle(substitution)
        transformed = bytearray(data)
        for i in range(len(transformed)):
            transformed[i] = substitution[transformed[i]]
        return bytes(transformed)
    
    def reverse_transform_06(self, data, seed=42):
        random.seed(seed)
        substitution = list(range(256))
        random.shuffle(substitution)
        reverse_substitution = [0] * 256
        for i, v in enumerate(substitution):
            reverse_substitution[v] = i
        transformed = bytearray(data)
        for i in range(len(transformed)):
            transformed[i] = reverse_substitution[transformed[i]]
        return bytes(transformed)

    def transform_07(self, data):
        """
        Transformation using 1,000,000 base-10 digits of pi, mapped to 0-255.
        Includes circular shift based on file size for enhanced randomness.
        """
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))  # 1 cycle for <1KB, up to 10 for larger
        logging.info(f"transform_07: Using {cycles} cycles for {len(data)} bytes (base-256)")
        shift = len(data) % pi_length
        pi_digits_shifted = self.PI_DIGITS[shift:] + self.PI_DIGITS[:shift]
        size_byte = len(data) % 256
        for i in range(len(transformed)):
            transformed[i] ^= size_byte
        for _ in range(cycles):
            for i in range(len(transformed)):
                pi_digit = pi_digits_shifted[i % pi_length]
                transformed[i] ^= pi_digit
        return bytes(transformed)
    
    def reverse_transform_07(self, data):
        """
        Reverse transformation for transform_07 (self-inverse since XOR is its own inverse).
        """
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        logging.info(f"reverse_transform_07: Using {cycles} cycles for {len(data)} bytes (base-256)")
        shift = len(data) % pi_length
        pi_digits_shifted = self.PI_DIGITS[shift:] + self.PI_DIGITS[:shift]
        for _ in range(cycles):
            for i in range(len(transformed)):
                pi_digit = pi_digits_shifted[i % pi_length]
                transformed[i] ^= pi_digit
        size_byte = len(data) % 256
        for i in range(len(transformed)):
            transformed[i] ^= size_byte
        return bytes(transformed)

    def transform_08(self, data):
        """
        Transformation using 1,000,000 base-10 digits of pi with circular shift and prime-based pre-transformation.
        """
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))  # 1 cycle for <1KB, up to 10 for larger
        logging.info(f"transform_08: Using {cycles} cycles for {len(data)} bytes (base-256)")
        shift = len(data) % pi_length
        pi_digits_shifted = self.PI_DIGITS[shift:] + self.PI_DIGITS[:shift]
        size_prime = find_nearest_prime_around(len(data) % 256)
        for i in range(len(transformed)):
            transformed[i] ^= size_prime
        for _ in range(cycles):
            for i in range(len(transformed)):
                pi_digit = pi_digits_shifted[i % pi_length]
                transformed[i] ^= pi_digit
        return bytes(transformed)
    
    def reverse_transform_08(self, data):
        """
        Reverse transformation for transform_08 (self-inverse since XOR is its own inverse).
        """
        transformed = bytearray(data)
        pi_length = len(self.PI_DIGITS)
        data_size_kb = len(data) / 1024
        cycles = min(10, max(1, int(data_size_kb)))
        logging.info(f"reverse_transform_08: Using {cycles} cycles for {len(data)} bytes (base-256)")
        shift = len(data) % pi_length
        pi_digits_shifted = self.PI_DIGITS[shift:] + self.PI_DIGITS[:shift]
        for _ in range(cycles):
            for i in range(len(transformed)):
                pi_digit = pi_digits_shifted[i % pi_length]
                transformed[i] ^= pi_digit
        size_prime = find_nearest_prime_around(len(data) % 256)
        for i in range(len(transformed)):
            transformed[i] ^= size_prime
        return bytes(transformed)

    def compress_with_best_method(self, data, filetype, output_file):
        # Apply transform_08
        transformed = self.transform_08(data)
        
        methods = [
            ('paq', self.paq_compress),
            ('zlib', self.compress_data_zlib),
        ]
        best_compressed = None
        best_size = float('inf')
        best_method = None

        for method_name, compress_func in methods:
            try:
                compressed = compress_func(transformed)
                if compressed is None:
                    continue
                size = len(compressed)
                if size < best_size:
                    best_size = size
                    best_compressed = compressed
                    best_method = method_name
            except Exception as e:
                logging.warning(f"Compression method {method_name} failed: {e}")
                continue

        if best_compressed is None:
            logging.error("All compression methods failed.")
            return None

        # Prepend 0x08 marker byte
        final_output = bytes([0x08]) + best_compressed
        logging.info(f"Best compression method: {best_method} for {filetype.name}")

        # Save the compressed data (with marker) to output_file
        try:
            with open(output_file, "wb") as f_out:
                f_out.write(final_output)
            logging.info(f"Compression successful. Output saved to {output_file}. Size: {len(final_output)} bytes")
        except Exception as e:
            logging.error(f"Error saving compressed file: {e}")
            return None

        return final_output, best_method
    
    def decompress_with_best_method(self, data):
        # Check for 0x08 marker byte
        if len(data) < 1 or data[0] != 0x08:
            logging.error("Invalid input: Missing 0x08 marker byte.")
            return b'', None
        
        # Remove marker byte
        compressed_data = data[1:]
        
        # Try PAQ decompression first
        try:
            decompressed = self.paq_decompress(compressed_data)
            return self.reverse_transform_08(decompressed), 8
        except Exception as e:
            logging.warning(f"PAQ decompression failed: {e}. Trying zlib...")

        # Try zlib decompression
        decompressed = self.decompress_data_zlib(compressed_data)
        if decompressed is None:
            logging.error("All decompression methods failed.")
            return b'', None
        
        return self.reverse_transform_08(decompressed), 8

def detect_filetype(filename: str) -> Filetype:
    """Detect filetype based on extension."""
    _, ext = os.path.splitext(filename.lower())
    if ext == '.jpg' or ext == '.jpeg':
        return Filetype.JPEG
    elif ext == '.txt':
        return Filetype.TEXT
    else:
        return Filetype.DEFAULT

def main():
    print("PAQJP_4 Compression System with Base-10 Pi Transformation (1,000,000 digits, Transform 08, 0x08 Marker, Save Best)")
    print("Created by Jurijus Pacalovas.")
    print("Options:")
    print("1 - Compress file (PAQJP_4 with transform_08, PAQ/zlib, 0x08 marker, save best)")
    print("2 - Decompress file (PAQJP_4 with transform_08, PAQ/zlib, check 0x08 marker)")

    compressor = SmartCompressor()

    try:
        choice = input("Enter 1 or 2: ").strip()
        if choice not in ('1', '2'):
            logging.error("Invalid choice. Exiting.")
            return
    except EOFError:
        logging.info("No input detected. Defaulting to Compress (1).")
        choice = '1'

    input_file = input("Input file name: ").strip()
    output_file = input("Output file name: ").strip()

    if not os.path.isfile(input_file):
        logging.error(f"Error: Input file '{input_file}' does not exist.")
        return

    filetype = detect_filetype(input_file)
    logging.info(f"Detected filetype: {filetype.name}")

    if choice == '1':
        with open(input_file, "rb") as f:
            input_data = f.read()
        
        compressed, best_method = compressor.compress_with_best_method(input_data, filetype, output_file)
        if compressed is None:
            return
        
        # Log sizes and compression ratio
        orig_size = len(input_data)
        comp_size = len(compressed)
        ratio = (comp_size / orig_size) * 100 if orig_size > 0 else 0
        logging.info(f"Original: {orig_size} bytes, Compressed (with 0x08 marker): {comp_size} bytes, Ratio: {ratio:.2f}%")
    
    elif choice == '2':
        with open(input_file, "rb") as f:
            input_data = f.read()
        
        try:
            decompressed, marker = compressor.decompress_with_best_method(input_data)
            if decompressed is None:
                return
            
            # Preserve original extension
            orig_ext = os.path.splitext(input_file)[1].lower()
            if not output_file.endswith(('.jpg', '.jpeg', '.txt')):
                if orig_ext in ['.jpg', '.jpeg']:
                    output_file += '.jpg'
                elif orig_ext == '.txt':
                    output_file += '.txt'
            
            with open(output_file, "wb") as f_out:
                f_out.write(decompressed)
            
            comp_size = len(input_data)
            decomp_size = len(decompressed)
            logging.info(f"Decompression successful. Output saved to {output_file}.")
            logging.info(f"Compressed (with 0x08 marker): {comp_size} bytes, Decompressed: {decomp_size} bytes")
        except Exception as e:
            logging.error(f"Error during decompression: {e}")

if __name__ == "__main__":
    main()
