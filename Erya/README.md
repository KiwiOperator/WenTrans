---
license: apache-2.0
pipeline_tag: translation
language:
- zh
---

# Model Description

Erya is a pretrained model specifically designed for translating Ancient Chinese into Modern Chinese. It utilizes an Encoder-Decoder architecture and has been trained using a combination of DMLM (Dual Masked Language Model) and DAS (Disyllabic Aligned Substitution) techniques on datasets comprising both Ancient Chinese and Modern Chinese texts. The detailed information of our work can be found here: [RUCAIBox/Erya (github.com)](https://github.com/RUCAIBox/Erya) 

More information about Erya dataset can be found here: [RUCAIBox/Erya-dataset · Datasets at Hugging Face](https://huggingface.co/datasets/RUCAIBox/Erya-dataset), which can be used to tune the Erya model further for a better translation performance.



# Example

```python
>>> from transformers import BertTokenizer, CPTForConditionalGeneration

>>> tokenizer = BertTokenizer.from_pretrained("RUCAIBox/Erya")
>>> model = CPTForConditionalGeneration.from_pretrained("RUCAIBox/Erya")

>>> input_ids = tokenizer("安世字子孺，少以父任为郎。", return_tensors='pt')
>>> input_ids.pop("token_type_ids")

>>> pred_ids = model.generate(max_new_tokens=256, **input_ids)
>>> print(tokenizer.batch_decode(pred_ids, skip_special_tokens=True))
    ['安 世 字 子 孺 ， 年 轻 时 因 父 任 郎 官 。']
```