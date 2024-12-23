# flake8: noqa: E501
import json
import re

from datasets import Dataset
from Levenshtein import distance

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class O1CipherDataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                question = line['prompt']
                answer = line['ground_truth']
                # Ensure answer is converted to string if it's a list or any other type
                if isinstance(answer, list):
                    answer = ' '.join(map(str, answer))
                dataset.append({'question': question, 'answer': answer})
        return Dataset.from_list(dataset)


class O1CipherEvaluator(BaseEvaluator):

    def score(self, predictions, gold):
        extracted_answers = []
        details = []
        total_score = 0
        for pred, ref in zip(predictions, gold):
            extracted_answer = self.extract_from_output_str(pred)
            extracted_answers.append(extracted_answer)
            ground_truth = str(ref).strip()
            score = 1 - distance(extracted_answer, ground_truth) / max(
                len(extracted_answer), len(ground_truth))
            total_score += score
            details.append({
                'model_output': pred,
                'extracted_answer': extracted_answer,
                'answer': ground_truth,
                'score': score,
            })
        average_score = total_score / len(predictions) if predictions else 0
        result = {'score': average_score, 'details': details}
        return result

    def extract_from_output_str(self, output: str):
        pattern1 = r'boxed\{(.*?)\}'
        pattern2 = r'conclude>(.*?)</conclude>'
        keywords = [
            'text is:',
            'is:',
            '是:',
            '是：',
            '结果:',
            '结果：',
            '为:',
            '为：',
            '本:',
            '本：',
            '串:',
            '串：',
        ]
        match = re.search(pattern1, output)
        if match:
            return match.group(1)
        else:
            match = re.search(pattern2, output, re.DOTALL)
            if match:
                res = match.group(1)
                res = res.strip()
                for keyword in keywords:
                    if keyword in res:
                        res = ''.join(res.split(keyword)[1:])
                        break
                return res.strip().strip('```').strip()
            else:
                res = output
                for keyword in keywords:
                    if keyword in res:
                        res = res.split(keyword)[-1]
                        break
                return res.strip().strip('```').strip()
