import pickle
import regex as re
from tqdm import tqdm
import datetime


class MinBPE:
    MAX_ITER: int
    regex_str = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

    def __init__(self, vocab=None, merges=None, regex_sep=regex_str):
        """
        Initializes the MinBPE tokenizer.

        :param vocab: Path to the vocabulary file (pickle format).
        :param merges: Path to the merges file (pickle format).
        :param regex_sep: Regular expression for tokenizing the text.
        """
        if vocab is not None:
            with open(vocab, 'rb') as f:
                self.vocab = pickle.load(f)
            with open(merges, 'rb') as f:
                self.merges = pickle.load(f)
        else:
            self.vocab = {}
            self.merges = {}
        self.separator = re.compile(regex_sep)

    def train(self, text_data: str, vocab_size: int):
        """
        Trains the BPE tokenizer on the provided text data.

        :param text_data: Text data to train the tokenizer.
        :param vocab_size: Desired size of the vocabulary.
        """
        assert len(self.vocab) == 0, "Already has prior data"

        MAX_ITER = vocab_size - 256
        new_token = 256
        level1 = MinBPE.str2ids(self.separator.findall(text_data))

        for i in tqdm(range(MAX_ITER)):
            stats = MinBPE.get_stats_u(level1)
            max_pair = max(stats, key=stats.get)
            level1 = MinBPE.change_u(level1, max_pair, new_token)
            self.merges[max_pair] = new_token
            new_token += 1

        self.vocab = {id: bytes([id]) for id in range(256)}
        for (k1, k2), v in self.merges.items():
            self.vocab[v] = self.vocab[k1] + self.vocab[k2]

        with open(f'./data/{datetime.datetime.now().strftime("%Y-%m-%d-T-%H-%M")}_vocab.pkl', 'wb') as f:
            pickle.dump(self.vocab, f)

        with open(f'./data/{datetime.datetime.now().strftime("%Y-%m-%d-T-%H-%M")}_merges.pkl', 'wb') as f:
            pickle.dump(self.merges, f)

    def encode(self, _text: str) -> list[int]:
        """
        Encodes the given text into a list of integers using BPE.

        :param _text: Text to encode.
        :return: List of encoded integers.
        """
        level1 = MinBPE.str2ids(self.separator.findall(_text))
        n: int = len(_text)

        while n > 1:
            stats = MinBPE.get_stats_u(level1)
            if len(stats) == 0:
                break
            least_pair = min(stats, key=lambda x: self.merges.get(x, float("inf")))
            if least_pair not in self.merges.keys():
                break
            else:
                level1 = MinBPE.change_u(level1, least_pair, self.merges[least_pair])
        return MinBPE.flatten(level1)

    def decode(self, ids: list[int]) -> str:
        """
        Decodes a list of integers back into text using the BPE vocabulary.

        :param ids: List of integers to decode.
        :return: Decoded text.
        """
        ans = b"".join(self.vocab[i] for i in ids)
        return ans.decode('utf-8')

    def display(self, _text: str) -> str:
        """
        Displays the encoded text with separators for each token.

        :param _text: Text to encode and display.
        :return: Encoded text with separators.
        """
        return " ".join([f"|{self.vocab[i].decode('utf-8')}|" for i in self.encode(_text)])

    @classmethod
    def flatten(cls, L: list[list[int]]) -> list[int]:
        """
        Flattens a list of lists into a single list.

        :param L: List of lists to flatten.
        :return: Flattened list.
        """
        return [item for sublist in L for item in sublist]

    @classmethod
    def str2ids(cls, _myinput101: list[str]) -> list[list[int]]:
        """
        Converts a list of strings into a list of lists of integers.

        :param _myinput101: List of strings to convert.
        :return: List of lists of integers.
        """
        return [list(word.encode("utf-8")) for word in _myinput101]

    @classmethod
    def get_stats_u(cls, ids: list[list[int]]) -> dict[tuple[int, int], int]:
        """
        Gets the frequency of each pair of bytes in the list of lists.

        :param ids: List of lists of integers.
        :return: Dictionary of byte pairs and their frequencies.
        """
        stats = {}
        for id in ids:
            if len(id) == 1:
                continue
            for (i, j) in zip(id, id[1:]):
                stats[(i, j)] = stats.get((i, j), 0) + 1

        return stats

    @classmethod
    def change_u(cls, _myinput101: list[list[int]], a: tuple[int, int], new: int) -> list[list[int]]:
        """
        Replaces all occurrences of a pair of bytes with a new byte in the list of lists.

        :param _myinput101: List of lists of integers.
        :param a: Byte pair to replace.
        :param new: New byte to replace the pair with.
        :return: Modified list of lists of integers.
        """
        newids = []
        for id in _myinput101:
            temps = []
            n: int = len(id)
            i: int = 0
            while i < n:
                if i + 1 < n and (id[i], id[i + 1]) == a:
                    temps.append(new)
                    i += 2
                else:
                    temps.append(id[i])
                    i += 1
            newids.append(temps)
        return newids
