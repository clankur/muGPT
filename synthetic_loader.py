# %%
import jax
import functools
from typing import Tuple
from input_loader import TokenBatch
from dataclasses import dataclass
import random
import string
from collections import defaultdict, deque

# %%


class TrieNode:
    def __init__(self):
        self.terminal = False
        self.children = defaultdict(TrieNode)


class VariableTrie:
    def __init__(self):
        self.root = TrieNode()

    def generate_random_variable(self, prefix=""):
        node = self.root
        random_var = prefix
        for c in prefix:
            node = node.children[c]

        while True:
            var = random.choice(string.ascii_lowercase)
            random_var += var
            node = node.children[var]
            if not node.terminal:
                break
        node.terminal = True
        return random_var


class SyntheticGenerator:

    def __init__(self, seed: int, min_seq_length: int = 256):
        self.variables = {}
        self.trie = VariableTrie()
        self.last_entries = deque(maxlen=3)
        self.min_seq_length = min_seq_length
        random.seed(seed)

    def generate_print_statement(self, var, value):
        """Generate a print statement with an expected value as a comment."""
        return f"print({var}) # {value}"

    def generate_random_variable(self):
        """Generate a random lowercase variable name, randomly use last 3 entries as a prefix"""
        if random.random() < 0.5 and len(self.last_entries) == 3:
            var = self.trie.generate_random_variable("".join(self.last_entries))
        else:
            var = self.trie.generate_random_variable()

        return var

    def generate_assignment(self):
        """Generate an assignment statement for a new or existing variable."""
        # 70% chance of creating a new variable if few variables exist, else 50%
        if len(self.variables) < 2 or random.random() < 0.5:
            var = self.generate_random_variable()
            value = random.randint(1, 100)
            self.variables[var] = value
        else:
            var = random.choice(list(self.variables.keys()))
            if random.random() < 0.2:  # 20% chance to assign a new value
                value = random.randint(1, 100)
                self.variables[var] = value
            else:  # 80% chance to assign value of another variable
                value = random.choice(list(self.variables.keys()))
                self.variables[var] = self.variables[value]
        self.last_entries.append(var[-1])
        return f"{var}={value}"

    def get_next_sequence(self):
        sequence = ""
        print_mask = []
        while len(sequence) < self.min_seq_length:
            if len(self.variables) < 0 or random.random() < 0.7:
                sequence += self.generate_assignment()
            else:
                var = random.choice(list(self.variables.keys()))
                value = self.variables[var]
                sequence += self.generate_print_statement(var, value)
                # last characters are the var's value, add to mask
                mask_region = (sequence.rfind(" ") + 1, len(sequence))
                print_mask.append(mask_region)
            sequence += "\n"
        return sequence, print_mask

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_next_sequence()


# %%
generator = SyntheticGenerator(42)
v, mask = next(generator)
print(len(v), "\n")
print(v)


# %%
@dataclass
class SyntethicDataParams:
    seed: int = 0
    streams: int


class SyntheticDataLoader:
    def __init__(self):
        pass

    def load(self, step):
        shape = (self.batch_size, self.max_seq_len)
        batch, is_start = next(self.iterator)

        def get_shard(x: jax.Array, indexing: Tuple[slice]) -> jax.Array:
            shard = x[indexing]
            return shard

        tokens = jax.make_array_from_callback(
            shape, self.sharding, functools.partial(get_shard, batch)
        )
        is_start = jax.make_array_from_callback(
            shape, self.sharding, functools.partial(get_shard, is_start)
        )
        return TokenBatch(tokens, is_start)
