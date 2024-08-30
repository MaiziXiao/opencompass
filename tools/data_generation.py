import argparse
import json
import os
from typing import Dict

from mmengine.config import Config, ConfigDict

from opencompass.models.base import BaseModel
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.registry import ICL_PROMPT_TEMPLATES, ICL_RETRIEVERS
from opencompass.utils import build_dataset_from_cfg, dataset_abbr_from_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='View generated prompts based on datasets (and models)')
    parser.add_argument('config', help='Train config file path')
    parser.add_argument('--output-path',
                        default='./outputs/prompt_data',
                        help='Output file path')
    args = parser.parse_args()
    return args


def parse_dataset_cfg(dataset_cfg: ConfigDict) -> Dict[str, ConfigDict]:
    dataset2cfg = {}
    for dataset in dataset_cfg:
        dataset2cfg[dataset_abbr_from_cfg(dataset)] = dataset
    return dataset2cfg


def write_to_json(results_dict, filename: str):
    """Dump the result to a json file."""
    # Create the directory if it does not exist
    dir_name = os.path.dirname(filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(results_dict, json_file, indent=4, ensure_ascii=False)


def print_prompts(dataset_cfg):
    # TODO: A really dirty method that copies code from PPLInferencer and
    # GenInferencer. In the future, the prompt extraction code should be
    # extracted and generalized as a static method in these Inferencers
    # and reused here.

    infer_cfg = dataset_cfg.get('infer_cfg')

    dataset = build_dataset_from_cfg(dataset_cfg)

    ice_template = None
    if hasattr(infer_cfg, 'ice_template'):
        ice_template = ICL_PROMPT_TEMPLATES.build(infer_cfg['ice_template'])

    prompt_template = None
    if hasattr(infer_cfg, 'prompt_template'):
        prompt_template = ICL_PROMPT_TEMPLATES.build(
            infer_cfg['prompt_template'])

    infer_cfg['retriever']['dataset'] = dataset
    retriever = ICL_RETRIEVERS.build(infer_cfg['retriever'])

    ice_idx_list = retriever.retrieve()

    supported_inferencer = [GenInferencer]
    if infer_cfg.inferencer.type not in supported_inferencer:
        print(f'Only {supported_inferencer} are supported')
        return

    # Prompt List
    prompt_list = []
    for idx, ice_idx in enumerate(ice_idx_list):
        ice = retriever.generate_ice(ice_idx, ice_template=ice_template)
        prompt = retriever.generate_prompt_for_generate_task(
            idx,
            ice,
            ice_template=ice_template,
            prompt_template=prompt_template)
        base_model = BaseModel('internlm/internlm2_5-1_8b-chat')
        new_prompt = base_model.parse_template(prompt, mode='gen')
        prompt_list.append(new_prompt)
        # prompt_list.append(prompt)

    # Fetch and zip prompt & gold answer if output column exists
    ds_reader = retriever.dataset_reader
    if ds_reader.output_column:
        gold_ans = ds_reader.dataset['test'][ds_reader.output_column]
        # prompt_list = list(zip(prompt_list, gold_ans))

    index = 0
    results_dict = {}
    for prompt, gold in zip(prompt_list, gold_ans):
        results_dict[str(index)] = {
            'origin_prompt': prompt,
            'gold': gold,
        }
        index = index + 1

    return results_dict


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if 'datasets' in cfg:
        dataset2cfg = parse_dataset_cfg(cfg.datasets)
    else:
        dataset2cfg = {}
        for key in cfg.keys():
            if key.endswith('_datasets'):
                dataset2cfg.update(parse_dataset_cfg(cfg[key]))

    print(f'Loading config from {args.config}')
    # Output path
    if isinstance(args.config, str) and args.config.endswith('.py'):
        dataset_folder = args.config.strip('.py')
        last_dir = os.path.basename(dataset_folder)
        second_last_dir = os.path.basename(os.path.dirname(dataset_folder))
        dataset_config_path = os.path.join(second_last_dir, last_dir)
    output_path = os.path.join(args.output_path, dataset_config_path)
    print(f'Output path: {output_path}')

    for dataset_abbr, dataset_cfg in dataset2cfg.items():
        print('=' * 64, '[BEGIN]', '=' * 64)
        print(f'[DATASET]: {dataset_abbr}')
        print('---')
        results_dict = print_prompts(dataset_cfg)
        output_file = os.path.join(output_path, f'./{dataset_abbr}.json')
        print(f'Write jsonl to {output_file}')
        write_to_json(results_dict, output_file)
        print('=' * 65, '[END]', '=' * 65)
        print()


if __name__ == '__main__':
    main()
