# flake8: noqa: F401, E501
import os
import subprocess


def find_and_execute_gen_files(directory, script_path):
    # 遍历指定目录及其子目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            # 检查文件是否为 .py 文件并且文件名包含 "gen"
            if file.endswith('.py') and 'gen' in file:
                # 构建完整的文件路径
                file_path = os.path.join(root, file)
                # 执行 data_generation.py 脚本并传入当前找到的文件作为参数
                subprocess.run(['python', script_path, file_path])


# 设置需要遍历的目录和数据生成脚本的路径
# directory_to_search = '/cpfs01/user/xiaolinchen/opencompass_fork/opencompass/configs/datasets/bbh'
# directory_to_search = '/cpfs01/user/xiaolinchen/opencompass_fork/opencompass/configs/datasets/mmlu'
directory_to_search = '/cpfs01/user/xiaolinchen/opencompass_fork/opencompass/configs/datasets/cmmlu'
directory_to_search = '/cpfs01/user/xiaolinchen/opencompass_fork/opencompass/configs/datasets/math'
directory_to_search = '/cpfs01/user/xiaolinchen/opencompass_fork/opencompass/configs/datasets/humaneval'
directory_to_search = '/cpfs01/user/xiaolinchen/opencompass_fork/opencompass/configs/datasets/drop'
directory_to_search = '/cpfs01/user/xiaolinchen/opencompass_fork/opencompass/configs/datasets/gpqa'
directory_to_search = '/cpfs01/user/xiaolinchen/opencompass_fork/opencompass/configs/datasets/IFEval'
directory_to_search = '/cpfs01/user/xiaolinchen/opencompass_fork/opencompass/configs/datasets/gsm8k'

data_generation_script_path = '/cpfs01/user/xiaolinchen/opencompass_fork/tools/data_generation.py'

# 调用函数
directories = [
    '/cpfs01/user/xiaolinchen/opencompass_fork/opencompass/configs/datasets/bbh',
    '/cpfs01/user/xiaolinchen/opencompass_fork/opencompass/configs/datasets/mmlu',
    '/cpfs01/user/xiaolinchen/opencompass_fork/opencompass/configs/datasets/cmmlu',
    '/cpfs01/user/xiaolinchen/opencompass_fork/opencompass/configs/datasets/math',
    '/cpfs01/user/xiaolinchen/opencompass_fork/opencompass/configs/datasets/drop',
    '/cpfs01/user/xiaolinchen/opencompass_fork/opencompass/configs/datasets/gpqa',
    '/cpfs01/user/xiaolinchen/opencompass_fork/opencompass/configs/datasets/IFEval',
    '/cpfs01/user/xiaolinchen/opencompass_fork/opencompass/configs/datasets/gsm8k',
    '/cpfs01/user/xiaolinchen/opencompass_fork/opencompass/configs/datasets/mmlu_pro',
]

for a in directories:
    find_and_execute_gen_files(a, data_generation_script_path)
