"""
Agent自动纠错系统 - 主入口

多Agent协作的自动纠错系统，通过循环反思机制不断优化答案。

使用方法:
    python main.py                    # 交互模式
    python main.py --question "问题"   # 单次执行

作者: Agent Auto Correction Team
版本: 1.0.0
"""

import argparse
import sys
from typing import Optional

from config.settings import get_settings, Settings
from core.orchestrator import Orchestrator
from core.memory_coordinator import MemoryCoordinator
from memory.working_memory import WorkingMemoryManager
from memory.long_term_memory import LongTermMemoryManager
from memory.meta_knowledge import MetaKnowledgeManager
from utils.llm_client import LLMClient
from utils.logger import get_logger


def create_system(settings: Settings) -> Orchestrator:
    """
    创建系统实例
    
    Args:
        settings: 配置对象
        
    Returns:
        Orchestrator实例
    """
    logger = get_logger()
    
    # 创建LLM客户端
    logger.info("初始化LLM客户端...")
    llm_client = LLMClient(
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        model_name=settings.llm.model_name
    )
    
    # 创建记忆管理器
    logger.info("初始化记忆系统...")
    wm_manager = WorkingMemoryManager(settings.storage.working_memory_dir)
    ltm_manager = LongTermMemoryManager(settings.storage.long_term_memory_dir)
    mk_manager = MetaKnowledgeManager(settings.storage.meta_knowledge_dir)
    
    # 创建记忆协调器
    memory_coordinator = MemoryCoordinator(
        wm_manager=wm_manager,
        ltm_manager=ltm_manager,
        mk_manager=mk_manager
    )
    
    # 创建编排器
    logger.info("初始化流程编排器...")
    orchestrator = Orchestrator(
        memory_coordinator=memory_coordinator,
        llm_client=llm_client
    )
    
    logger.info("系统初始化完成")
    return orchestrator


def run_single_question(
    orchestrator: Orchestrator,
    question: str,
    question_type: str = "general"
) -> None:
    """
    执行单个问题
    
    Args:
        orchestrator: 编排器实例
        question: 问题文本
        question_type: 问题类型
    """
    logger = get_logger()
    
    print("\n" + "=" * 60)
    print(f"问题: {question}")
    print("=" * 60)
    
    logger.info(f"开始处理问题: {question[:50]}...")
    
    # 执行
    result = orchestrator.execute(
        question=question,
        question_type=question_type
    )
    
    # 输出结果
    print("\n" + "-" * 60)
    print("执行结果:")
    print("-" * 60)
    print(f"任务ID: {result.task_id}")
    print(f"最终答案: {result.final_answer}")
    print(f"置信度: {result.confidence:.2f}")
    print(f"循环轮次: {result.total_rounds}")
    print(f"执行时间: {result.execution_time:.2f}秒")
    print(f"执行状态: {'成功' if result.success else '失败'}")
    
    # 显示 Insight Agent 评估结果
    if result.metadata.get("quality_score"):
        print(f"答案质量分数: {result.metadata['quality_score']:.2f}")
    if result.metadata.get("correctness"):
        print(f"正确性判断: {result.metadata['correctness']}")
    
    if result.metadata.get("stop_reason"):
        print(f"停止原因: {result.metadata['stop_reason']}")
    
    if not result.success and result.metadata.get("error"):
        print(f"错误信息: {result.metadata['error']}")
    
    print("-" * 60)
    
    logger.info(f"任务完成: {result.task_id}, 成功: {result.success}")


def run_interactive(orchestrator: Orchestrator) -> None:
    """
    交互式运行模式
    
    Args:
        orchestrator: 编排器实例
    """
    print("\n" + "=" * 60)
    print("欢迎使用 Agent 自动纠错系统")
    print("输入问题开始处理，输入 'quit' 或 'exit' 退出")
    print("=" * 60 + "\n")
    
    while True:
        try:
            question = input("请输入问题: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ('quit', 'exit', 'q'):
                print("再见!")
                break
            
            # # 询问问题类型
            # print("问题类型 (直接回车使用默认 'general'):")
            # print("  - general: 通用问题")
            # print("  - math: 数学问题")
            # print("  - logic: 逻辑推理")
            # print("  - knowledge: 知识问答")
            # question_type = input("类型: ").strip() or "general"
            
            # run_single_question(orchestrator, question, question_type)
            run_single_question(orchestrator, question, "general")
            
            
        except KeyboardInterrupt:
            print("\n操作已取消")
            break
        except Exception as e:
            print(f"发生错误: {e}")


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Agent自动纠错系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py                           # 交互模式
    python main.py --question "问题内容"      # 单次执行
    python main.py --question "问题" --type math  # 指定问题类型
        """
    )
    
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="要处理的问题"
    )
    
    parser.add_argument(
        "--type", "-t",
        type=str,
        default="general",
        choices=["general", "math", "logic", "knowledge"],
        help="问题类型 (默认: general)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    settings = get_settings()
    
    # 应用命令行参数
    if args.debug:
        settings.debug = True
    
    # 初始化日志
    logger = get_logger()
    
    try:
        # 创建系统
        orchestrator = create_system(settings)
        
        # 根据模式运行
        if args.question:
            run_single_question(orchestrator, args.question, args.type)
        else:
            run_interactive(orchestrator)
            
    except ValueError as e:
        logger.error(f"配置错误: {e}")
        print(f"\n错误: {e}")
        print("\n请确保已正确配置API密钥。参考 README.md 获取帮助。")
        sys.exit(1)
    except Exception as e:
        logger.error(f"系统错误: {e}")
        print(f"\n系统错误: {e}")
        if settings.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
