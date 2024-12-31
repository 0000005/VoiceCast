import io
import sys
import logging
import logging.config
import os
import sys
from util.text_util import extract_person_names_from_passage_with_weight

# 确保logs目录存在
if getattr(sys, "frozen", False):
    # 如果是打包后的环境
    base_dir = os.path.dirname(os.path.dirname(sys.executable))
    logs_dir = os.path.join(base_dir, "wecharm", "logs")
else:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    logs_dir = os.path.join(base_dir, "logs")

if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# 配置日志
if getattr(sys, "frozen", False):
    # 打包环境下，使用基本的日志配置
    logging.basicConfig(
        filename=os.path.join(logs_dir, "backend.log"),
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
else:
    # 开发环境下，使用配置文件
    logging.config.fileConfig(
        os.path.join(os.path.dirname(__file__), "config", "logging.conf")
    )

logger = logging.getLogger("voicecast")


if __name__ == "__main__":
    # 执行代码
    # 小说文本路径
    novel_path = os.path.join(base_dir, "resource", "三体1.txt")

    try:
        # 读取小说文本
        logger.info(f"开始读取小说文件: {novel_path}")
        with open(novel_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 提取人名
        logger.info("开始提取人名...")
        name_counts = extract_person_names_from_passage_with_weight(content)

        # 输出结果
        logger.info("人名提取完成，按出现频率排序输出:")
        for name, count in name_counts.items():
            print(f"{name}: {count}次")

    except FileNotFoundError:
        logger.error(f"找不到文件: {novel_path}")
    except Exception as e:
        logger.error(f"处理过程中出现错误: {str(e)}")
