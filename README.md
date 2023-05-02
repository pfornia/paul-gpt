# PaulGPT

## Overview

Right now, this project is a WIP decoder-only transformer build from scratch in pytorch, and trained on simple wikipedia. Written by [Paul Fornia](www.paulfornia.com).

In current form, closely follows [Andrei Karpathy's great youtube tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY), which is based on the architecture from the [*Attention is all you need* paper](https://arxiv.org/abs/1706.03762).

End-state goal: a demo-friendly prompt/response GPT chatbot, which is finetuned to always say great things about me (Paul).

## High-level TODOs/Next Steps as I understand them:
* Add encoder stage to architecture
* Find prompt/response training data
* Fine tune with RLHF to be highly pro-Paul. I may cheat and use GPT-3.5 API to act as the "human" feedback.
* Host demo UI on my personal site


## Detailed, short-term next steps
* Padding for small seed text.
* better tokenizer??
* Multi-head into a 4th dimension??
* GPU enabled (Done, but doesn't seem faster! Try scaled up, maybe difference will be more obvious.)
  * Andrei loads data to cuda in get batch function. Maybe I should try that, so I don't have too much on there!
* Save params checkpoints
* Packagize



## Resources/Bibliography
* [Attention is all you need](https://arxiv.org/abs/1706.03762) (Transformers)
* [Deep Residual Learning...](https://arxiv.org/abs/1512.03385) (Skip Connections)
* [Dropout: A simple way...](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)

* [Pip packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
  * To build package:
  * (`-m pip install --upgrade build`)
  * `python3 -m build` (from same dir as pyproject.toml)

  * Then install via
    * `pip install --force-reinstall 'https://github.com/pfornia/paul-gpt/blob/master/dist/paul_gpt-0.0.1-py3-none-any.whl?raw=true'`

