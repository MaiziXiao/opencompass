from mmengine.config import read_base

with read_base():
    from opencompass.configs.models.qwen2_5.lmdeploy_qwen2_5_7b_instruct import (
        models,
    )
    from opencompass.configs.datasets.o1_puzzle.o1_cipher_gen import (
        o1_cipher_datasets,
    )
    from opencompass.configs.datasets.o1_puzzle.o1_game24_gen import (
        o1_game24_datasets,
    )
    from opencompass.configs.datasets.o1_puzzle.o1_crossword_gen import (
        o1_crossword_datasets,
    )

datasets = o1_crossword_datasets + o1_game24_datasets + o1_cipher_datasets
models = models
