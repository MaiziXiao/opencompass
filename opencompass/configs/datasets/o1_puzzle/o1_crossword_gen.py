from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

from opencompass.datasets.o1_puzzle import (
    O1CrosswordDataset,
    O1CrosswordEvaluator,
)


reader_cfg = dict(input_columns=['question'], output_column='answer_info')

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{question}'),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer,
    ),
)
eval_cfg = dict(evaluator=dict(type=O1CrosswordEvaluator))


o1_crossword_datasets = [
    dict(
        type=O1CrosswordDataset,
        abbr='o1_crossword',
        path='./data/o1_puzzle/crossword_100.jsonl',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]
