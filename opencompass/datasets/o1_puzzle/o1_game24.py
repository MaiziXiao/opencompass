# flake8: noqa: E501, E722
import fractions
import json

from datasets import Dataset

from opencompass.openicl import BaseEvaluator
from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class O1Game24Dataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
        dataset = []
        with open(path, 'r') as f:
            for line in f:
                line = json.loads(line)
                question = line['puzzle']
                answer = line['target']
                # Ensure answer is converted to string if it's a list or any other type
                if isinstance(answer, list):
                    answer = ' '.join(map(str, answer))
                dataset.append({'question': question, 'answer': answer})
        return Dataset.from_list(dataset)


class O1Game24Evaluator(BaseEvaluator):

    def score(self, predictions, gold, origin_prompt):
        corrects = []
        parsed_solutions = []
        for text, ex, prompt in zip(predictions, gold, origin_prompt):
            flag, parsed_solution = False, 'answer not found'
            if '</conclude>' in text:
                model_pred = (text.split('<conclude>')[-1].strip().replace(
                    '</conclude>', '').strip().split('\n\n')[-1])
                flag, parsed_solution = self.check_solution(
                    ex['puzzle'], ex['target'], model_pred)
            else:
                model_pred = text
            corrects.append(flag)
            parsed_solutions.append({
                'origin_prompt': prompt,
                'prediction': text,
                'extracted_output': model_pred,
                'is_correct': flag,
                'parsed_solution': parsed_solution,
            })
        num_correct = sum(corrects)
        return {
            'score': num_correct / len(gold) * 100 if gold else 0,
            'details': parsed_solutions,
        }

    def convert_number(self, n):
        if '/' in n:
            try:
                return fractions.Fraction(n)
            except:
                return None
        else:
            try:
                return int(n)
            except:
                return None

    def check_calculate(self, n1, op, n2, res):
        n1 = self.convert_number(n1)
        n2 = self.convert_number(n2)
        res = self.convert_number(res)
        if n1 is None or n2 is None or res is None:
            return False
        if op == '+':
            return n1 + n2 == res
        elif op == '-':
            return n1 - n2 == res
        elif op == '*':
            return n1 * n2 == res
        elif op == '/':
            if n2 == 0:
                return False
            return n1 / n2 == res

    def check_solution(self, puzzle, target, solution):
        current_numbers = puzzle.split()
        parsed_solution = []
        for step in solution.split('\n'):
            if '(' in step:
                step = step.split('(')[0]
            else:
                continue
            if '=' not in step:
                parsed_solution.append(['Invalid step', step])
                return False, parsed_solution
            tokens = step.split()
            if len(tokens) != 5 or tokens[1] not in '+-*/' or tokens[3] != '=':
                parsed_solution.append(['Invalid tokens', tokens])
                return False, parsed_solution
            n1, op, n2, res = tokens[0], tokens[1], tokens[2], tokens[4]
            if n1 not in current_numbers:
                parsed_solution.append(
                    ['Invalid numbers', [n1, current_numbers]])
                return False, parsed_solution
            else:
                current_numbers.remove(n1)
            if n2 not in current_numbers:
                parsed_solution.append(
                    ['Invalid numbers', [n2, current_numbers]])
                return False, parsed_solution
            else:
                current_numbers.remove(n2)
            current_numbers.append(res)
            calc_check = self.check_calculate(n1, op, n2, res)
            if not calc_check:
                parsed_solution.append(
                    ['Invalid calculation', [n1, op, n2, res]])
                return False, parsed_solution
            parsed_solution.append(
                ['Valid', [n1, op, n2, res, current_numbers]])
        if len(current_numbers) != 1 or current_numbers[0] != target:
            parsed_solution.append(['Invalid final numbers', current_numbers])
            return False, parsed_solution
        return True, parsed_solution
