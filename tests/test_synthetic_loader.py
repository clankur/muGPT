import pytest
import math
import jax.numpy as jnp
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from synthetic_loader import SyntheticTokenizer, SyntheticGenerator, VariableTrie


@pytest.fixture
def tokenizer():
    return SyntheticTokenizer()


@pytest.fixture
def generator():
    return SyntheticGenerator(seed=42, seq_length=512, batch_size=2)


def test_tokenizer(tokenizer):
    test_cases = ["a=123\n", "x='abc'\n", "y=(1,2,3)\n"]
    for test_text in test_cases:
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        assert decoded == test_text


def test_tokenizer_invalid_chars(tokenizer):
    with pytest.raises(KeyError):
        tokenizer.encode("invalid_char$")


def test_variable_trie():
    trie = VariableTrie()

    variables = [trie.generate_random_variable() for _ in range(10)]
    assert len(set(variables)) == len(variables), "Generated variables should be unique"
    assert all(var.islower() for var in variables), "Variables should be lowercase"


@pytest.mark.parametrize("seq_length,batch_size", [(512, 1), (512, 2), (1024, 4)])
def test_synthetic_generator_shapes(seq_length, batch_size):
    generator = SyntheticGenerator(
        seed=42, seq_length=seq_length, batch_size=batch_size
    )
    sequences, starts, ends, loss_masks = generator.generate_batch()
    n_non_void_funcs = 1 + generator.n_generate_funcs
    n_eval_calls = int(
        n_non_void_funcs
        + math.log2(generator.seq_length - generator.min_padding) // n_non_void_funcs
    )
    print(sequences.shape, starts.shape, ends.shape, loss_masks.shape)
    assert sequences.shape == (batch_size, seq_length)
    assert starts.shape == (batch_size, n_eval_calls)
    assert ends.shape == (batch_size, n_eval_calls)
    assert loss_masks.shape == (batch_size, seq_length)
    assert jnp.any(
        starts[-1] != ends[-1]
    ), "Not all comment positions were set properly."

    assert jnp.all(starts < generator.seq_length), "Starts are out of bounds."
