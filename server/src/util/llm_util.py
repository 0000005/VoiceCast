from math import log
from re import I
from model.setting_model import Settings
from langchain_community.chat_models import ChatOpenAI
from util.text_util import trim_text
import logging
import json

logger = logging.getLogger("voicecast")


def deconstruct_passage(text: str, speaker: list, settings: Settings):
    """
    使用LLM将文本提取成对话
    Args:
        text: 输入文本
        speaker: 对话的发言者
    Returns:
        list: 包含所有对话的列表
    """

    trimed_text = trim_text(text)
    speaker_list_str = str(speaker)

    # 使用LLM将文本提取成对话
    system_prompt_template = """你是一名小说对话提取器，负责从提供的文本中提取对话和旁白，并将结果以 JSON 格式输出。
# 提取规则：
1. 角色名匹配：根据"发言者列表"中提供的角色名，判断对话的发言者。
- 如果文本中出现与角色名含义相近的别称或指代（例如“汪淼”和“汪教授”指代同一人），统一将 speaker 设置为“发言者列表”中的标准角色名。示例：如果"发言者列表"中有“汪淼”，但文本中出现“汪教授”，则将其识别为“汪淼”。
- 如果文本中出现的角色名不在"发言者列表"中，统一将 speaker 设置为文中提到的人物（可能是无关紧要的配角）。
2. 旁白处理：对于非对话内容，统一将 speaker 设置为 "旁白"。
3. text 字段中的内容必须与原文完全一致，不得增删、修改或重写任何文字。
4. 提取文中所有的对话，包括旁白，不要遗漏任何文本。
5. 输出格式：按照以下 JSON 数组格式输出结果：
[
    {
        "speaker": "角色名或旁白",
        "tone":"说话语气"
        "text": "对话或旁白内容"
    }
]

# 示例:
- 发言者列表
["汪淼","史强"]
- 小说文本
两人继续讨论实验的细节。汪教授说道：“这次的实验非常重要”。“是啊，汪淼，你也太拼了”，史强回答道。
# 
[
    {
        "speaker": "旁白",
        "tone": "平静",
        "text": "两人正在讨论实验的细节"
    },
    {
        "speaker": "汪淼",
        "tone": "激动",
        "text": "这次的实验非常重要。"
    },
    {
        "speaker": "史强",
        "tone": "平静",
        "text": "是啊，汪淼，你也太拼了。"
    }
]

"""
    user_prompt_template = """
# 正文
- 发言者列表
[speaker_list_str]
- 小说文本
[trimed_text]
"""
    # 替换占位符
    user_prompt_template = user_prompt_template.replace(
        "[speaker_list_str]", speaker_list_str
    )
    user_prompt_template = user_prompt_template.replace("[trimed_text]", trimed_text)

    # Create messages and get response
    messages = [
        {"role": "system", "content": system_prompt_template},
        {"role": "user", "content": user_prompt_template},
    ]
    logger.info("LLM request: " + str(messages))
    llm = _create_llm_instance(settings)
    response = llm.invoke(messages)
    logger.info("LLM response: " + str(response))
    return json.loads(response.content)


def filter_person_name(speaker: dict, settings: Settings):
    """
    使用LLM提取过滤出真正的人名。
    Args:
        speaker: 对话的发言者
    Returns:
        dict: 对话的发言者
    """

    filter_person_name_prompt = """你是一个人名识别器，我将会给你一个数组，这个数组中包含了一些人名和其他内容，你需要从这些内容中提取出人名。并以原有的json格式返回。
# 示例
- 输入
{"张三":20,"李四":10,"电脑":5}
- 输出
{"张三":20,"李四":10}
"""

    # 如果speaker为空，使用则返回空字典
    if len(speaker) == 0:
        return {}
    user_prompt = str(speaker)

    # Create messages and get response
    messages = [
        {"role": "system", "content": filter_person_name_prompt},
        {"role": "user", "content": user_prompt},
    ]
    logger.info("LLM request: " + str(messages))
    llm = _create_llm_instance(settings)
    response = llm.invoke(messages)
    logger.info("LLM response: " + str(response.content))
    return json.loads(response.content)


def _create_llm_instance(settings: Settings) -> ChatOpenAI:
    """Create and return a ChatOpenAI instance with the given settings."""
    return ChatOpenAI(
        model=settings.model,
        openai_api_key=settings.apiKey,
        openai_api_base=settings.baseUrl,
        max_tokens=8000,
        temperature=0.5,
        response_format={"type": "json_object"},
    )
