# %%
import jax.numpy as jnp
from typing import Tuple, List
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


class SyntheticTokenizer:
    def __init__(self):
        self.vocab = list("abcdefghijklmnopqrstuvwxyz=()#0123456789+-*/\n ") + [
            "[PAD]",
        ]  # add "[END_OF_TEXT]" as max token id?
        self.token_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_token = {idx: char for char, idx in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id["[PAD]"]  # Special token ID for padding

    def encode(self, text: str) -> jnp.ndarray:
        """Encode text into a sequence of token IDs."""
        return jnp.array([self.token_to_id[char] for char in text], dtype=jnp.uint32)

    def decode(self, token_ids: jnp.ndarray) -> str:
        """Decode a sequence of token IDs back into text."""
        return "".join([self.id_to_token.get(int(id_), "") for id_ in token_ids])


class SyntheticGenerator:

    def __init__(self, seed: int, seq_length: int, batch_size: int):
        self.variables = {}
        self.trie = VariableTrie()
        self.last_entries = deque(maxlen=3)
        self.seq_length = seq_length
        self.min_padding = 20
        self.tokenizer = SyntheticTokenizer()
        self.batch_size = batch_size
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
        # If no variables exist or if the random chance dictates, generate a new variable
        if not self.variables or random.random() < 0.5:
            var = self.generate_random_variable()
            value = random.randint(1, 100)
            self.variables[var] = value
        else:
            # Choose an existing variable
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
        mask = jnp.zeros((self.seq_length))
        while len(sequence) + self.min_padding < self.seq_length:
            if len(self.variables) == 0 or random.random() < 0.7:
                sequence += self.generate_assignment()
            else:
                var = random.choice(list(self.variables.keys()))
                value = self.variables[var]
                sequence += self.generate_print_statement(var, value)
                # last characters are the var's value, add to mask
                mask_region = (sequence.rfind(" ") + 1, len(sequence))
                mask = mask.at[mask_region[0] : mask_region[1]].set(1)
            sequence += "\n"
        return sequence, mask

    def reset_state(self):
        """Reset stateful attributes to ensure independence between batches."""
        self.variables = {}  # Clear variable assignments
        self.last_entries = deque(maxlen=3)  # Clear recent entries
        self.trie = VariableTrie()

    def generate_batch(
        self,
    ) -> Tuple[jnp.ndarray, List[List[Tuple[int, int]]]]:
        """
        Generate a batch of tokenized sequences and their respective print masks.

        Returns:
            Tuple[jnp.ndarray, List[List[Tuple[int, int]]]]: A batch of tokenized sequences and their masks.
        """
        sequences = []
        masks = []

        for _ in range(self.batch_size):
            sequence, print_mask = self.get_next_sequence()
            encoded_sequence = jnp.pad(
                self.tokenizer.encode(sequence),
                (0, self.seq_length - len(sequence)),
                constant_values=self.tokenizer.pad_token_id,
            )
            sequences.append(encoded_sequence)
            masks.append(print_mask)
            self.reset_state()

        padded_sequences = jnp.stack(sequences)
        masks = jnp.stack(masks)
        return padded_sequences, masks

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_batch()


# %%
