from model.setting_model import Settings
from util.llm_util import deconstruct_passage, filter_person_name
from util.text_util import extract_person_names_from_passage_with_weight
import logging
import logging.config
import os
import json

# 确保logs目录存在
base_dir = os.path.dirname(os.path.dirname(__file__))
logs_dir = os.path.join(base_dir, "logs")

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# 开发环境下，使用配置文件
logging.config.fileConfig(
    os.path.join(os.path.dirname(__file__), "config", "logging.conf")
)

logger = logging.getLogger("voicecast")


def test_deconstruct_passage():
    # 构建Settings对象
    settings = Settings(
        model="deepseek-chat",
        baseUrl="https://api.deepseek.com/v1",
        apiKey="sk-",
    )
    # settings = Settings(
    #     model="chatgpt-4o-latest",
    #     baseUrl="https://api.openai.com/v1",
    #     apiKey="sk-",
    # )

    # 从文件读取测试文本
    with open(
        "d:/code/python-workspace/VoiceCast/server/resource/三体第一章.txt",
        "r",
        encoding="utf-8",
    ) as f:
        text = f.read()

    # 提取人名
    name_counts = extract_person_names_from_passage_with_weight(text)

    # 过滤人名
    name_counts = filter_person_name(name_counts, settings)

    # 输出结果
    name_list = []
    print("人名提取完成，按出现频率排序输出:")
    for name, count in name_counts.items():
        print(f"{name}: {count}次")
        name_list.append(name)

    # 调用deconstruct_passage函数
    result = deconstruct_passage(text, name_list, settings)

    print("输出结果:" + result)


if __name__ == "__main__":
    test_deconstruct_passage()
