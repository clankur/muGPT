import math
import jax
import jax.numpy as jnp
from typing import Tuple, List, Optional
import random
import string
from collections import defaultdict, deque


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
        self.vocab = list(
            string.ascii_lowercase + "=()#" + string.digits + "+-*/\n', "
        ) + [
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

    def __init__(self, seq_length: int, batch_size: int):
        assert seq_length >= 256, "Sequence length must be at least 512"
        self.seq_length = seq_length
        self.min_padding = int(seq_length * 0.25)
        self.tokenizer = SyntheticTokenizer()
        self.batch_size = batch_size
        self.void_func_freq = 0.001
        self.func_freq = 0.025
        self.pattern_cap = 5
        self.n_generate_funcs = 2

    def get_next_sequence(self):
        variables = {}
        trie = VariableTrie()
        generate_functions = {
            "print": lambda s: s,
            "eval": lambda s: s,
            "choice": lambda s: random.choice(s),
        }
        sequence = ""
        total_len = self.seq_length - self.min_padding
        n_void_calls = int(self.void_func_freq * total_len)
        void_functions = ["print", "eval"]
        current_len = len(sequence)
        loss_mask = jnp.ones((self.seq_length), dtype=jnp.bool)

        def generate_random_variable():
            return trie.generate_random_variable()

        def generate_random_value(v_type):
            if not v_type:
                v_type = random.choice(["int", "str", "tuple"])
            if v_type == "tuple":
                tuple_size = random.randint(2, 4)
                return tuple(random.randint(1, 999) for _ in range(tuple_size))
            elif v_type == "int":
                return random.randint(1, 999)
            else:  # str
                length = random.randint(1, 3)
                letters = string.ascii_lowercase
                return "".join(random.choice(letters) for _ in range(length))

        def generate_assignment(v_type: Optional[str] = None):
            if not variables or v_type or random.random() < 0.5:
                var = generate_random_variable()
                value = generate_random_value(v_type)
                variables[var] = value
            else:
                var = random.choice(list(variables.keys()))
                if random.random() < 0.2:
                    value = generate_random_value(v_type)
                    variables[var] = value
                else:
                    value = random.choice(list(variables.keys()))
                    variables[var] = variables[value]
            if type(value) == str and variables[var] == value:
                value = f"'{value}'"
            return f"{var}={value}"

        def generate_func():
            func_name = generate_random_variable()

            func_type = random.choice(["repeat", "map", "shift", "reverse"])
            chars = "".join(
                random.sample(list(string.ascii_lowercase), random.randint(0, 2))
            )

            if func_type == "repeat":
                n_repeat_input = random.randint(1, 3)
                generate_functions[func_name] = lambda s: (s * n_repeat_input) + chars

            elif func_type == "shift":
                shift_amount = random.randint(1, len(self.tokenizer.vocab) - 1)

                def shift_func(s):
                    result = ""
                    for c in s:
                        idx = string.ascii_lowercase.index(c)
                        new_idx = (
                            idx + shift_amount
                        ) % 26  # length of lowercase alphabet
                        result += string.ascii_lowercase[new_idx]
                    return result + chars

                generate_functions[func_name] = shift_func

            elif func_type == "reverse":
                generate_functions[func_name] = lambda s: s[::-1] + chars

            else:
                char_mapping = {
                    c: random.choice(string.ascii_lowercase)
                    for c in string.ascii_lowercase
                }

                def map_func(s):
                    return "".join(char_mapping.get(c, c) for c in s) + chars

                generate_functions[func_name] = map_func

            return generate_functions[func_name]

        def call_func(func_name: str, var: Optional[str] = None):
            nonlocal variables, current_len, sequence
            if var:
                value = variables[var]
            else:
                if func_name == "choice":
                    filter_cond = lambda x: isinstance(x, tuple)
                    choices = [
                        (key, value)
                        for (key, value) in variables.items()
                        if filter_cond(value)
                    ]
                    v_type = "tuple"
                else:
                    filter_cond = (
                        lambda x: isinstance(x, str) and len(x) < self.pattern_cap
                    )
                    choices = [
                        (key, value)
                        for (key, value) in variables.items()
                        if filter_cond(value)
                    ]
                    v_type = "str"
                if len(choices) < 2:
                    stmnt = generate_assignment(v_type) + "\n"
                    current_len += len(stmnt)
                    sequence += stmnt
                    choices = [
                        (key, value)
                        for (key, value) in variables.items()
                        if filter_cond(value)
                    ]
                var, value = random.choice(choices)

            if func_name in void_functions:
                if type(value) == str:
                    value = f"'{value}'"
                return f"{func_name}({var}) # {value}", None

            new_var = generate_random_variable()
            variables[new_var] = generate_functions[func_name](value)
            return f"{new_var}={func_name}({var})", new_var

        sequence = ""
        total_len = self.seq_length - self.min_padding
        n_void_calls = int(self.void_func_freq * total_len)
        void_functions = ["print", "eval"]
        current_len = len(sequence)

        for _ in range(self.n_generate_funcs):
            generate_func()

        n_non_void_funcs = len(
            [k for k in generate_functions.keys() if k not in void_functions]
        )
        n_eval_calls = int(n_non_void_funcs + math.log2(total_len) // n_non_void_funcs)
        comment_start = jnp.zeros((n_eval_calls), dtype=jnp.uint32)
        comment_end = jnp.zeros((n_eval_calls), dtype=jnp.uint32)

        func_calls_print = [
            (func, "print")
            for func in generate_functions.keys()
            for _ in range(max(2, int(math.log2(total_len) / len(generate_functions))))
            if func not in void_functions
        ]
        random.shuffle(func_calls_print)

        func_calls_eval = [
            (func, "eval")
            for func in generate_functions.keys()
            if func not in void_functions
        ]
        random.shuffle(func_calls_eval)

        required_calls = func_calls_print + func_calls_eval
        required_calls += random.choices(
            [(func, "eval") for func in generate_functions.keys()],
            k=int(n_eval_calls) - len(func_calls_eval),
        )

        positions = sorted(random.sample(range(1, total_len), len(required_calls)))
        gen_positions = [
            (pos, func, post_func)
            for pos, (func, post_func) in zip(positions, required_calls)
        ]
        void_positions = [
            (pos, "print", None)
            for pos in random.sample(range(1, total_len), n_void_calls)
        ]
        random_positions = sorted(gen_positions + void_positions, key=lambda x: x[0])
        eval_count = 0
        for target_pos, func, post_func in random_positions:
            while current_len < target_pos:
                stmnt = generate_assignment()
                sequence += stmnt + "\n"
                current_len += len(stmnt) + 1

            stmnt, var = call_func(func)
            if post_func:
                stmnt += "\n" + call_func(post_func, var)[0]
                region = (
                    current_len + stmnt.rfind("#") + 2,
                    current_len + len(stmnt) + 1,
                )
                if post_func == "eval":
                    comment_start = comment_start.at[eval_count].set(region[0])
                    comment_end = comment_end.at[eval_count].set(region[1])
                    eval_count += 1
                else:
                    loss_mask = loss_mask.at[region[0] : region[1]].set(0)

            sequence += stmnt + "\n"
            current_len += len(stmnt) + 1

        while current_len < total_len - self.min_padding:
            stmnt = generate_assignment()
            sequence += stmnt + "\n"
            current_len += len(stmnt) + 1

        if len(sequence) > self.seq_length:
            truncate_idx = sequence[: self.seq_length].rfind("\n") + 1
            sequence = sequence[:truncate_idx]
        assert (
            comment_start[-1] != comment_end[-1]
        ), f"Not all comment positions were set properly. \n{comment_start=}\n{comment_end=}"
        return sequence, comment_start, comment_end, loss_mask

    def generate_batch(self) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        sequences = []
        starts = []
        ends = []
        loss_masks = []

        for _ in range(self.batch_size):
            sequence, comment_start, comment_end, loss_mask = self.get_next_sequence()
            encoded_sequence = jnp.pad(
                self.tokenizer.encode(sequence),
                (0, self.seq_length - len(sequence)),
                constant_values=self.tokenizer.pad_token_id,
            )
            sequences.append(encoded_sequence)
            starts.append(comment_start)
            ends.append(comment_end)
            loss_masks.append(loss_mask)

        return (
            jnp.stack(sequences),
            jnp.stack(starts, dtype=jnp.uint32),
            jnp.stack(ends, dtype=jnp.uint32),
            jnp.stack(loss_masks, dtype=jnp.bool),
        )

    def __iter__(self):
        return self

    def __next__(self):
        return self.generate_batch()
