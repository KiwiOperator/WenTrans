import argparse

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer

import config
import label
from metrics import end_of_chunk, start_of_chunk
from model import BertSegPos


DEFAULT_TEXTS = [
    "書曰衞人立晉。",
    "頃公之嬖人盧蒲就魁門焉。",
    "子曰學而時習之不亦說乎。",
]


def normalize_char(token):
    if token in {"“", "”", "「", "」"}:
        return '"'
    if token in {"‘", "’", "『", "』", "（", "）", "(", ")"}:
        return "'"
    return token


def chunk_text(text, max_len=511):
    chars = [normalize_char(ch) for ch in text.strip() if ch.strip()]
    if not chars:
        return []
    return [chars[i:i + max_len] for i in range(0, len(chars), max_len)]


def encode_chunks(texts, tokenizer):
    encoded = []
    original_chunks = []
    for text in texts:
        for chars in chunk_text(text):
            tokens = ["[CLS]"] + chars
            token_ids = [tokenizer.convert_tokens_to_ids(tok) for tok in tokens]
            encoded.append(torch.LongTensor(token_ids))
            original_chunks.append(chars)
    return encoded, original_chunks


def decode_words(chars, seg_tags, pos_tags):
    prev_tag = "O"
    begin_offset = 0
    pieces = []
    for i, chunk in enumerate(seg_tags + ["O"]):
        tag = chunk[0]
        if end_of_chunk(prev_tag, tag):
            word = chars[begin_offset:i]
            word_pos = pos_tags[begin_offset:i]
            if word:
                pieces.append(f"{''.join(word)}/{max(word_pos, key=word_pos.count)}")
        if start_of_chunk(prev_tag, tag):
            begin_offset = i
        prev_tag = tag
    return " ".join(pieces)


def predict(texts):
    tokenizer = AutoTokenizer.from_pretrained(config.berta_model)
    model = BertSegPos(config, None)
    model.to(config.device)
    model.load_state_dict(torch.load(config.save_checkpoint, map_location=config.device))
    model.eval()

    encoded, original_chunks = encode_chunks(texts, tokenizer)
    if not encoded:
        return []

    batch_data = pad_sequence(encoded, batch_first=True, padding_value=0).to(config.device)
    attention_mask = batch_data.gt(0).to(config.device)

    with torch.no_grad():
        batch_output = model(batch_data, token_type_ids=None, attention_mask=attention_mask)[0]
        batch_segoutput, batch_posoutput = batch_output
        label_mask = attention_mask[:, 1:]
        pred_seg = model.crf_seg.decode(batch_segoutput, mask=label_mask)
        pred_pos = model.crf_pos.decode(batch_posoutput, mask=label_mask)

    outputs = []
    for chars, seg_ids, pos_ids in zip(original_chunks, pred_seg, pred_pos):
        seg_tags = [label.id_seg2label[idx] for idx in seg_ids[:len(chars)]]
        pos_tags = [label.id_pos2label[idx] for idx in pos_ids[:len(chars)]]
        outputs.append(decode_words(chars, seg_tags, pos_tags))
    return outputs


def main():
    parser = argparse.ArgumentParser(description="Run a toy Ancient Chinese segmentation+POS example.")
    parser.add_argument(
        "texts",
        nargs="*",
        help="Raw ancient Chinese text. If omitted, built-in toy examples are used.",
    )
    args = parser.parse_args()

    texts = args.texts or DEFAULT_TEXTS
    outputs = predict(texts)

    for raw_text, pred in zip(texts, outputs):
        print(f"INPUT : {raw_text}")
        print(f"OUTPUT: {pred}")
        print()


if __name__ == "__main__":
    main()
