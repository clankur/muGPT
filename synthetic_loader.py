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
        self.seq_length = seq_length
        self.min_padding = int(seq_length * 0.1)
        self.tokenizer = SyntheticTokenizer()
        self.batch_size = batch_size
        self.print_freq = 0.02
        random.seed(seed)

    def generate_random_variable(self):
        """Generate a random lowercase variable name, randomly use last 3 entries as a prefix"""
        return self.trie.generate_random_variable()

    def generate_assignment(self):
        """Generate an assignment statement for a new or existing variable."""
        # If no variables exist or if the random chance dictates, generate a new variable
        if not self.variables or random.random() < 0.5:
            var = self.generate_random_variable()
            value = random.randint(1, 999)
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

        return f"{var}={value}"

    def generate_random_print(self):
        var = random.choice(list(self.variables.keys()))
        value =  self.variables[var]
        return f"print({var}) # {value}"

    def get_next_sequence(self):
        sequence = ""
        total_len = self.seq_length - self.min_padding
        num_prints = int( self.print_freq * total_len )
        comment_start = jnp.zeros((num_prints), dtype=jnp.uint32)
        comment_end = jnp.zeros((num_prints), dtype=jnp.uint32)
        
        # Predetermine random positions for print statements
        random_positions = sorted(random.sample(range(1, total_len), num_prints))
        
        current_len = 0  # Track current sequence length

        for pos, target_pos in enumerate(random_positions):
            while current_len < target_pos:
                stmnt = self.generate_assignment()
                sequence += stmnt + "\n"  
                current_len += len(stmnt) + 1  

            stmnt = self.generate_random_print()
            sequence += stmnt + "\n"  
            region = (sequence.rfind("#") + 2, len(sequence) + 1)
            
            comment_start = comment_start.at[pos].set(region[0])
            comment_end = comment_end.at[pos].set(region[1])
            current_len += len(stmnt) + 1  

        while current_len < total_len - self.min_padding:
            stmnt = self.generate_assignment()
            sequence += stmnt + "\n"  
            current_len += len(stmnt) + 1  

        if len(sequence) > self.seq_length:
            print(f"truncating {sequence[self.seq_length:]=}")
            sequence = sequence[:self.seq_length]

        assert comment_start[-1] != comment_end[-1], f"Not all comment positions were set properly. \n{comment_start=}\n{comment_end=}"
        return sequence, comment_start, comment_end

    def reset_state(self):
        """Reset stateful attributes to ensure independence between batches."""
        self.variables = {}  # Clear variable assignments
        self.trie = VariableTrie()

    def generate_batch(
        self,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Generate a batch of tokenized sequences and their respective print masks.

        Returns:
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: A batch of tokenized sequences and their masks.
        """
        sequences = []
        starts = []
        ends = []
        for _ in range(self.batch_size):
            sequence, comment_start, comment_end = self.get_next_sequence()
            encoded_sequence = jnp.pad(
                self.tokenizer.encode(sequence),
                (0, self.seq_length - len(sequence)),
                constant_values=self.tokenizer.pad_token_id,
            )
            sequences.append(encoded_sequence)
            starts.append(comment_start)
            ends.append(comment_end)
            self.reset_state()

        return jnp.stack(sequences), jnp.stack(starts), jnp.stack(ends)

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_batch()


# %%