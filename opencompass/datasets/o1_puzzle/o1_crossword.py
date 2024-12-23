# flake8: noqa: E501, E722
import ast
import json
import re

from datasets import Dataset
from Levenshtein import distance

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class O1CrosswordDataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                question = line['question']
                answer = line['raw_info']['ans_info']
                # answer = line['ans_info']
                # Ensure answer is converted to string if it's a list or any other type
                dataset.append({'question': question, 'answer_info': answer})
        return Dataset.from_list(dataset)


class O1CrosswordEvaluator(BaseEvaluator):

    def score(self, predictions, gold, origin_prompt):
        CORR_WORD = 0
        TOT_WORD = 0

        details = []
        for res_words, ans_info, prompt in zip(predictions, gold,
                                               origin_prompt):
            extract_result = self.verify_result(res_words)
            if not isinstance(extract_result, dict):
                extract_result = {}

            STEP_WORDS = {}
            for TYPE, infos in ans_info.items():
                for word_info in infos:
                    STEP_WORDS[str(
                        word_info['index'])] = word_info['word_info']['word']

            # word level
            WORD_COUNT = 0
            WORD_TOTAL = 0
            correct_words = {}
            for id, guess in extract_result.items():
                TOT_WORD += 1
                WORD_TOTAL += 1
                if guess.upper() == STEP_WORDS.get(id, None):
                    WORD_COUNT += 1
                    CORR_WORD += 1
                    correct_words[id] = guess

            details.append({
                'original_prompt':
                prompt,
                'res_words':
                res_words,
                'extract_result':
                extract_result,
                'WORD_TOTAL':
                WORD_TOTAL,
                'WORD_COUNT':
                WORD_COUNT,
                'correct_words':
                correct_words,
                'final_score':
                (WORD_COUNT / WORD_TOTAL if WORD_TOTAL > 0 else 0),
            })

        result = {
            'score': CORR_WORD / TOT_WORD if TOT_WORD > 0 else 0,
            'details': details,
        }
        return result

    def verify_result(self, reply):
        # 使用正则表达式找到所有的字典部分
        matches = re.findall(r'\{.*?\}', reply)

        if matches:
            # 获取匹配到的所有子串中的最后一个
            last_dict_str = matches[-1]

            # 使用ast.literal_eval安全地将字符串转换为字典
            try:
                result_dict = ast.literal_eval(last_dict_str)
                if isinstance(result_dict, dict):
                    return result_dict
            except (ValueError, SyntaxError):
                pass

        # 查找字典开始和结束的位置
        start_index = reply.find('{')
        end_index = reply.rfind('}') + 1
        if start_index != -1 and end_index != 0:
            # 提取字典字符串部分
            dict_string = reply[start_index:end_index]
            # 使用 ast.literal_eval 将字符串转换为 Python 字典
            try:
                extracted_dict = ast.literal_eval(dict_string)
                if isinstance(extracted_dict, dict):
                    return extracted_dict
            except (ValueError, SyntaxError):
                pass

        return reply
