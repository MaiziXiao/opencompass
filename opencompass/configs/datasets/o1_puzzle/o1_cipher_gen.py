from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

from opencompass.datasets.o1_puzzle import O1CipherDataset, O1CipherEvaluator


reader_cfg = dict(input_columns=['question'], output_column='answer')

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
eval_cfg = dict(evaluator=dict(type=O1CipherEvaluator))


o1_cipher_datasets = [
    dict(
        type=O1CipherDataset,
        abbr='o1_cipher',
        path='./data/o1_puzzle/eval_cipher.jsonl',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]
