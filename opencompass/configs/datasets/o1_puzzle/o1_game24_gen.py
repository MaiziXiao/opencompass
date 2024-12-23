from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer

from opencompass.datasets.o1_puzzle import O1Game24Dataset, O1Game24Evaluator


reader_cfg = dict(input_columns=['question'], output_column='answer')

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='Use numbers and basic arithmetic operations (+ - * /) to obtain the target number. Note that all given numbers must be used exactly once.\n\nInput: {question}\nTarget: 24'),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(
        type=GenInferencer,
    ),
)
eval_cfg = dict(evaluator=dict(type=O1Game24Evaluator))


o1_game24_datasets = [
    dict(
        type=O1Game24Dataset,
        abbr='o1_game24',
        path='./data/o1_puzzle/test_game24_testset.jsonl',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]
