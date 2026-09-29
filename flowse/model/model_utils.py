from __future__ import annotations



import torch
from torch.nn.utils.rnn import pad_sequence


def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d


# simple utf-8 tokenizer, since paper went character based
def list_str_to_tensor(text: list[str], padding_value=-1) -> int["b nt"]:  # noqa: F722
    list_tensors = [torch.tensor([*bytes(t, "UTF-8")], dtype=torch.long) for t in text]  # ByT5 style
    text = pad_sequence(list_tensors, padding_value=padding_value, batch_first=True)
    return text


# char tokenizer, based on custom dataset's extracted .txt file
def list_str_to_idx(
    text: list[str] | list[list[str]],
    vocab_char_map: dict[str, int],  # {char: idx}
    padding_value=-1,
) -> int["b nt"]:  # noqa: F722
    list_idx_tensors = [
        torch.tensor([vocab_char_map.get(c, 0) for c in t], dtype=torch.long)
        for t in text
    ]
    text = pad_sequence(list_idx_tensors, padding_value=padding_value, batch_first=True)
    return text


def convert_text(text: str, tokenizer: str) -> str | list[str]:
    """Apply the same text normalization used by the training dataset."""
    if tokenizer == "pinyin":
        from pypinyin import Style, pinyin

        return ["".join(item) for item in pinyin(text, style=Style.TONE3)]
    if tokenizer in {"char", "byte", "custom"}:
        return text
    raise ValueError(f"Unsupported tokenizer: {tokenizer}")


def get_tokenizer(tokenizer_path: str, tokenizer: str = "pinyin"):
    """
    tokenizer   - "pinyin" do g2p for only chinese characters, need .txt vocab_file
                - "char" for char-wise tokenizer, need .txt vocab_file
                - "byte" for utf-8 tokenizer
                - "custom" if you're directly passing in a path to the vocab.txt you want to use
    vocab_size  - if use "pinyin", all available pinyin types, common alphabets (also those with accent) and symbols
                - if use "char", derived from unfiltered character & symbol counts of custom dataset
                - if use "byte", set to 256 (unicode byte range)
    """
    if tokenizer in {"pinyin", "char", "custom"}:
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            vocab_char_map = {}
            for i, char in enumerate(f):
                vocab_char_map[char.rstrip("\r\n")] = i
        vocab_size = len(vocab_char_map)
        if tokenizer in {"pinyin", "char"}:
            assert vocab_char_map[" "] == 0, "space must be index 0 because it is also the unknown token"

    elif tokenizer == "byte":
        vocab_char_map = None
        vocab_size = 256

    else:
        raise ValueError(f"Unsupported tokenizer: {tokenizer}")

    return vocab_char_map, vocab_size
