import jieba
import jieba.posseg as pseg
import logging
import re

# from LAC import LAC
from ltp import LTP
from transformers import pipeline

# 配置日志
logger = logging.getLogger("voicecast")


def trim_text(text):
    """
    去除多余的换行和空格
    1、将文本中连续的换行符替换成一个换行符。
    2、将每一行的收尾多余空格删除。
    Args:
        text: 输入文本
    Returns:
        str: 处理后的文本
    """
    if not text:
        logger.info("输入文本为空，直接返回")
        return text

    logger.info(f"开始处理文本，原始长度: {len(text)} 字符")

    # 1. 将连续的换行符替换为单个换行符
    text = re.sub(r"\n\s*\n", "\n", text)
    logger.info("已完成多余换行符的处理")

    # 2. 处理每一行的首尾空格
    lines = text.split("\n")
    logger.info(f"文本分割为 {len(lines)} 行")

    lines = [line.strip() for line in lines]
    logger.info("已完成每行首尾空格的清理")

    result = "\n".join(lines)
    logger.info(f"文本处理完成，处理后长度: {len(result)} 字符")

    return result


def split_text(text, spliter, split_length=50):
    """
    文本分段函数
    分段逻辑：当文本的长度超过 slit_length 时，如果遇到了分隔符则进行分段。如果文本长度小于 split_length 则不进行分段。
    :param text: 文本
    :param spliter: 分隔符
    :param split_length: 每段长度
    :return: 分段数组
    """
    if not text:
        return []

    text = trim_text(text)

    if len(text) <= split_length:
        return [text]

    result = []
    current_segment = ""

    for char in text:
        current_segment += char

        # If current segment exceeds split_length and we find a splitter
        if len(current_segment) >= split_length and char in spliter:
            result.append(current_segment)
            current_segment = ""

    # Add any remaining text as the last segment
    if current_segment:
        result.append(current_segment)

    return result


def find_nr_from_text_using_jieba(text):
    """
    使用 jieba 对文本进行分词，从中找到所有的人名
    Args:
        text: 输入文本
    Returns:
        list: 包含所有识别出的人名的列表
    """
    if not text:
        return []

    # 使用jieba词性标注进行分词
    words = pseg.cut(text)

    # 提取所有词性为nr(人名)的词
    names = [word.word for word in words if word.flag == "nr"]

    return names


def find_nr_from_text_using_bert(text):
    """
    使用 Hugging Face 的 transformers 库中 BERT 模型对文本进行命名实体识别，从中找到所有的人名
    Args:
        text: 输入文本
    Returns:
        list: 包含所有识别出的人名的列表
    """
    # 如果文本为空，返回空列表
    if not text:
        return []

    # 初始化 Hugging Face 的 NER 管道
    # 默认使用的是一个支持 NER 的预训练 BERT 模型
    ner_pipeline = pipeline("ner", model="bert-base-chinese", grouped_entities=True)

    # 使用 NER 模型对输入文本进行命名实体识别
    entities = ner_pipeline(text)

    # 提取所有实体类别为 "PER"（Person，人名）的实体
    names = [entity["word"] for entity in entities if entity["entity_group"] == "PER"]

    # 去重并返回
    return list(set(names))


def find_nr_from_text_using_lac(text: str):
    """
    使用 LAC 对文本进行分词，从中找到所有的人名
    Args:
        text: 输入文本
    Returns:
        list: 包含所有识别出的人名的列表
    """
    # lac = LAC(mode="lac")
    # user_name_lis = []
    # _result = lac.run(text)
    # for _index, _label in enumerate(_result[1]):
    #     if _label == "PER":
    #         user_name_lis.append(_result[0][_index])
    # return user_name_lis


def find_nr_from_text_using_ltp(text: str):
    """
    使用 LTP 对文本进行分词，从中找到所有的人名
    Args:
        text: 输入文本
    Returns:
        list: 包含所有识别出的人名的列表
    """
    ltp = LTP()
    user_name_lis = []
    output = ltp.pipeline([text], tasks=["cws", "ner"])
    ### 循环output.ner
    for item in output.ner[0]:
        if item[0] == "Nh":
            user_name_lis.append(item[1])
    # print(output)
    return user_name_lis


def extract_person_names_from_passage_with_weight(text):
    """
    对文本进行分段，然后从每段中提取人名，并对提取的人名进行计数，最后返回一个有顺序的字典，计数最多的排在最（前面）开始。
    分段规则：每段的长度500，分隔符是换行符。
    Args:
        text: 输入文本
    Returns:
        dict: 有顺序的字典，计数最多的排在最（前面）
    """
    if not text:
        return {}

    logger.info(f"开始处理文本，总长度: {len(text)} 字符")

    # 使用换行符分段，每段500字符
    segments = split_text(text, "\n", 500)
    logger.info(f"文本已分段，共 {len(segments)} 段")

    # 用于存储每个人名出现的次数
    name_counts = {}

    # 处理每个文本段落
    for i, segment in enumerate(segments, 1):
        # 每处理10段输出一次进度
        if i % 10 == 0:
            logger.info(
                f"正在处理第 {i}/{len(segments)} 段，完成度: {(i/len(segments)*100):.1f}%"
            )

        # 从段落中提取人名
        names = find_nr_from_text_using_jieba(segment)
        # names = find_nr_from_text_using_bert(segment)
        # names = find_nr_from_text_using_ltp(segment)

        # 统计每个人名出现的次数，排除单字人名
        for name in names:
            if len(name) > 1:  # 只统计长度大于1的人名
                name_counts[name] = name_counts.get(name, 0) + 1

    logger.info(f"文本处理完成，共找到 {len(name_counts)} 个不同的人名")

    # 按照出现次数降序排序
    sorted_names = sorted(name_counts.items(), key=lambda x: x[1], reverse=True)

    # 只保留前15个人名
    top_15_names = sorted_names[:15]

    # 转换为有序字典
    result = dict(top_15_names)
    if result:
        top_names = list(result.items())[:3]
        logger.info(f"出现频率最高的前三个人名: {top_names}")

    return result
