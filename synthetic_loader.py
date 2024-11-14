# %%
import jax.numpy as jnp
from typing import Tuple, List, Optional
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
        self.vocab = list(string.ascii_lowercase + "=()#" + string.digits + "+-*/\n', ") + [
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
        self.funcs = {}
        self.trie = VariableTrie()
        self.seq_length = seq_length
        self.min_padding = int(seq_length * 0.1)
        self.tokenizer = SyntheticTokenizer()
        self.batch_size = batch_size
        self.void_func_freq = 0.02
        self.func_freq = 0.025
        random.seed(seed)

    def generate_random_variable(self):
        """Generate a random lowercase variable name, randomly use last 3 entries as a prefix"""
        return self.trie.generate_random_variable()
    
    def generate_random_value (self, v_type) -> int | str:
       if  v_type == 'int' or not v_type and random.random() < 0.5:
           return random.randint(1, 999)
       else:
           length = random.randint(1, 3)
           letters = string.ascii_lowercase 
           return ''.join(random.choice(letters) for _ in range(length))

    def generate_assignment(self, v_type:Optional[str]=None):
        """Generate an assignment statement for a new or existing variable."""
        # If no variables exist or if the random chance dictates, generate a new variable
        if not self.variables or v_type or random.random() < 0.5:
            var = self.generate_random_variable()
            value = self.generate_random_value(v_type)
            self.variables[var] = value
        else:
            var = random.choice(list(self.variables.keys()))
            if random.random() < 0.2:  # 20% chance to assign a new value
                value = self.generate_random_value(v_type)
                self.variables[var] = value
            else:  # 80% chance to assign value of another variable
                value = random.choice(list(self.variables.keys()))
                self.variables[var] = self.variables[value]
        if type(value) == str and self.variables[var] == value :
            value = f"'{value}'"
        return f"{var}={value}"

    def generate_random_print(self):
        var = random.choice(list(self.variables.keys()))
        value =  self.variables[var]
        if type(value) == str:
            value = f"'{value}'"
        return f"print({var}) # {value}"
    
    def generate_eval_stmnt(self):
        var = random.choice(list(self.variables.keys()))
        value =  self.variables[var]
        if type(value) == str:
            value = f"'{value}'"
        return f"eval({var}) # {value}"
    
    def get_next_sequence(self):
        sequence =  ""
        total_len = self.seq_length - self.min_padding
        n_gen_calls = int( self.func_freq * total_len )
        n_void_calls = int( self.void_func_freq * total_len )
        n_func_calls = n_void_calls + n_gen_calls
        comment_start = jnp.zeros((n_func_calls), dtype=jnp.uint32)
        comment_end = jnp.zeros((n_func_calls), dtype=jnp.uint32)
        generate_functions = {
            "rep": lambda s: s * 2
        } 
        
        current_len = len(sequence)  # Track current sequence length
        def generate_func ():
            func_name = self.generate_random_variable()  

            n_repeat_input = random.randint(1, 3) 
            chars = ''.join(random.sample(list(string.ascii_lowercase), random.randint(0, 2)))
            generate_functions[func_name] = lambda s: (s * n_repeat_input) + chars
            return generate_functions[func_name]

        def call_func (func_name):
            nonlocal current_len, sequence
            choices = [ ( key, value ) for ( key, value ) in self.variables.items() if isinstance(value, str)]
            if not choices:
               stmnt = self.generate_assignment("str") + "\n"
               current_len += len(stmnt) 
               sequence += stmnt
               choices = [ ( key, value ) for ( key, value ) in self.variables.items() if isinstance(value, str)]

            select_var, select_value = random.choice(choices)
            new_var = self.generate_random_variable()
            new_value = self.variables[new_var] = generate_functions[func_name](select_value)

            return f"{new_var}={func_name}({select_var}) # '{new_value}'"

                
        # Predetermine random positions for print statements
        void_functions = [
            self.generate_random_print,
            self.generate_eval_stmnt
        ]

        for i in range(3):
            generate_func()
        
        required_calls = [func for func in generate_functions.keys() for _ in range(2)]
        required_calls += random.choices(list(generate_functions.keys()), k=n_gen_calls - len(required_calls))

        positions = random.sample(range(1, total_len), n_gen_calls)
        gen_positions = [(pos, func) for pos, func in zip(positions, required_calls)]
        void_positions = [ (pos, random.choice(void_functions)) for pos in random.sample(range(1, total_len), n_void_calls) ]
        random_positions = sorted(gen_positions + void_positions, key=lambda x: x[0])
        for i, ( target_pos, func ) in enumerate(random_positions):
            while current_len < target_pos:
                stmnt = self.generate_assignment()
                sequence += stmnt + "\n"  
                current_len += len(stmnt) + 1  
            if func in generate_functions:
                stmnt = call_func(func)
            else:
                stmnt = func()
            sequence += stmnt + "\n"  
            region = (sequence.rfind("#") + 2, len(sequence))
        
            comment_start = comment_start.at[i].set(region[0])
            comment_end = comment_end.at[i].set(region[1])
            current_len += len(stmnt) + 1  
        while current_len < total_len - self.min_padding:
            stmnt = self.generate_assignment()
            sequence += stmnt + "\n"  
            current_len += len(stmnt) + 1  

        if len(sequence) > self.seq_length:
            print(len(sequence))
            sequence = sequence[:self.seq_length]
        assert comment_start[-1] != comment_end[-1], f"Not all comment positions were set properly. \n{comment_start=}\n{comment_end=}"
        return sequence, comment_start, comment_end

    def reset_state(self):
        """Reset stateful attributes to ensure independence between batches."""
        self.variables = {} 
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
