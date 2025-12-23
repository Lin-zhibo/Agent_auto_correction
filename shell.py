"""
Agent自动纠错系统 - 交互式Shell

提供一个简单的命令行界面进行系统测试和调试。

使用方法:
    python shell.py
"""

import cmd
import json
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


class AgentShell(cmd.Cmd):
    """
    Agent系统交互式Shell
    
    提供命令行界面进行交互式测试
    """
    
    intro = """
╔════════════════════════════════════════════════════════════════════╗
║              Agent 自动纠错系统 - 交互式Shell                       ║
╠════════════════════════════════════════════════════════════════════╣
║  ask <问题>     - 详细模式：显示每轮Agent评估过程                    ║
║  askq <问题>    - 简洁模式：只显示最终结果                          ║
║  quick <问题>   - 单轮模式：快速测试，不循环                        ║
║  test [verbose] - 运行内置测试用例                                  ║
║  agents         - 列出已注册的Agent                                 ║
║  config         - 查看/修改配置                                     ║
║  status         - 查看系统状态                                      ║
║  help           - 查看帮助                                          ║
║  quit/exit      - 退出系统                                          ║
╚════════════════════════════════════════════════════════════════════╝
    """
    prompt = "(agent) >>> "
    
    def __init__(self):
        """初始化Shell"""
        super().__init__()
        self.orchestrator: Optional[Orchestrator] = None
        self.settings: Optional[Settings] = None
        self.logger = get_logger()
        
        # 初始化系统
        self._init_system()
    
    def _init_system(self) -> None:
        """初始化系统"""
        try:
            self.settings = get_settings()
            
            # 创建LLM客户端
            llm_client = LLMClient(
                api_key=self.settings.llm.api_key,
                base_url=self.settings.llm.base_url,
                model_name=self.settings.llm.model_name
            )
            
            # 创建记忆管理器
            wm_manager = WorkingMemoryManager(self.settings.storage.working_memory_dir)
            ltm_manager = LongTermMemoryManager(self.settings.storage.long_term_memory_dir)
            mk_manager = MetaKnowledgeManager(self.settings.storage.meta_knowledge_dir)
            
            # 创建记忆协调器
            memory_coordinator = MemoryCoordinator(
                wm_manager=wm_manager,
                ltm_manager=ltm_manager,
                mk_manager=mk_manager
            )
            
            # 创建编排器
            self.orchestrator = Orchestrator(
                memory_coordinator=memory_coordinator,
                llm_client=llm_client
            )
            
            print("✓ 系统初始化成功")
            
        except Exception as e:
            print(f"✗ 系统初始化失败: {e}")
            self.orchestrator = None
    
    def do_ask(self, arg: str) -> None:
        """
        提问 (详细模式): ask <问题>
        
        显示每一轮各Agent的评估结果、关键参数、是否进入下一轮
        示例: ask 什么是人工智能?
        """
        if not arg.strip():
            print("请输入问题，例如: ask 什么是人工智能?")
            return
        
        if self.orchestrator is None:
            print("系统未初始化，请先运行 init 命令")
            return
        
        print(f"\n{'='*60}")
        print(f"📝 问题: {arg}")
        print(f"{'='*60}")
        
        try:
            # 使用 verbose 模式执行
            result = self.orchestrator.execute(arg, "general", verbose=True)
            
            # 最终结果汇总
            print(f"\n{'='*60}")
            print(f"📊 最终结果汇总")
            print(f"{'='*60}")
            print(f"答案: {result.final_answer}")
            print(f"置信度: {result.confidence:.2f}")
            print(f"总轮次: {result.total_rounds}")
            print(f"状态: {'✅ 成功' if result.success else '❌ 失败'}")
            
            # 显示 Insight Agent 评估结果
            if result.metadata.get("quality_score"):
                print(f"最终质量分数: {result.metadata['quality_score']:.2f}")
            if result.metadata.get("correctness"):
                print(f"最终正确性判断: {result.metadata['correctness']}")
            if result.metadata.get("recommended_action"):
                print(f"最终建议行动: {result.metadata['recommended_action']}")
            
            if result.metadata.get("stop_reason"):
                print(f"停止原因: {result.metadata['stop_reason']}")
                
        except Exception as e:
            print(f"执行出错: {e}")
        
        print(f"{'='*60}")
    
    def do_askq(self, arg: str) -> None:
        """
        快速提问 (简洁模式): askq <问题>
        
        完整多轮执行，但只显示最终结果
        示例: askq 什么是人工智能?
        """
        if not arg.strip():
            print("请输入问题，例如: askq 什么是人工智能?")
            return
        
        if self.orchestrator is None:
            print("系统未初始化，请先运行 init 命令")
            return
        
        print(f"\n处理问题: {arg}")
        print("-" * 50)
        
        try:
            result = self.orchestrator.execute(arg, "general", verbose=False)
            
            print(f"答案: {result.final_answer}")
            print(f"置信度: {result.confidence:.2f}")
            print(f"轮次: {result.total_rounds}")
            print(f"质量分数: {result.metadata.get('quality_score', 'N/A')}")
            print(f"正确性: {result.metadata.get('correctness', 'N/A')}")
            print(f"状态: {'✅ 成功' if result.success else '❌ 失败'}")
            
            if result.metadata.get("stop_reason"):
                print(f"停止原因: {result.metadata['stop_reason']}")
                
        except Exception as e:
            print(f"执行出错: {e}")
        
        print("-" * 50)
    
    def do_quick(self, arg: str) -> None:
        """
        快速提问 (单轮): quick <问题>
        
        不进行循环，只执行一轮获取结果（用于快速测试）
        示例: quick 1+1等于多少?
        """
        if not arg.strip():
            print("请输入问题")
            return
        
        if self.orchestrator is None:
            print("系统未初始化")
            return
        
        print(f"\n快速单轮处理: {arg}")
        print("-" * 50)
        
        try:
            result = self.orchestrator.execute_single_round(arg, "general")
            
            print(f"答案: {result.get('answer', 'N/A')}")
            print(f"置信度: {result.get('confidence', 0):.2f}")
            print(f"质量分数: {result.get('quality_score', 'N/A')}")
            print(f"建议行动: {result.get('recommended_action', 'N/A')}")
            
            if result.get('conflicts'):
                print(f"冲突点: {result['conflicts']}")
            if result.get('complementary_points'):
                print(f"互补观点: {result['complementary_points']}")
                
        except Exception as e:
            print(f"执行出错: {e}")
        
        print("-" * 50)
    
    def do_init(self, arg: str) -> None:
        """
        重新初始化系统: init
        """
        self._init_system()
    
    def do_status(self, arg: str) -> None:
        """
        查看系统状态: status
        """
        print("\n系统状态:")
        print("-" * 30)
        print(f"已初始化: {'是' if self.orchestrator else '否'}")
        
        if self.settings:
            print(f"模型: {self.settings.llm.model_name}")
            print(f"最大轮次: {self.settings.loop.max_rounds}")
            print(f"置信度阈值: {self.settings.loop.confidence_threshold}")
        
        print("-" * 30)
    
    def do_agents(self, arg: str) -> None:
        """
        列出已注册的Agent: agents
        """
        if self.orchestrator is None:
            print("系统未初始化")
            return
        
        print("\n已注册的Agent:")
        print("-" * 30)
        
        for name, agent in self.orchestrator._agents.items():
            print(f"  • {name}: {agent.agent_type.value} ({agent.agent_id})")
        
        print("-" * 30)
    
    def do_config(self, arg: str) -> None:
        """
        查看或修改配置: config [key] [value]
        
        config                 - 查看所有配置
        config max_rounds 5    - 设置最大轮次
        config threshold 0.9   - 设置置信度阈值
        """
        if not arg.strip():
            # 显示配置
            if self.settings:
                print("\n当前配置:")
                print(json.dumps({
                    "llm": {
                        "model": self.settings.llm.model_name,
                        "temperature": self.settings.llm.temperature
                    },
                    "loop": {
                        "max_rounds": self.settings.loop.max_rounds,
                        "confidence_threshold": self.settings.loop.confidence_threshold,
                        "alpha": self.settings.loop.alpha,
                        "beta": self.settings.loop.beta
                    }
                }, indent=2, ensure_ascii=False))
            return
        
        parts = arg.split()
        if len(parts) < 2:
            print("用法: config <key> <value>")
            return
        
        key, value = parts[0], parts[1]
        
        try:
            if key == "max_rounds":
                self.settings.loop.max_rounds = int(value)
            elif key == "threshold":
                self.settings.loop.confidence_threshold = float(value)
            elif key == "alpha":
                self.settings.loop.alpha = float(value)
            elif key == "beta":
                self.settings.loop.beta = float(value)
            else:
                print(f"未知配置项: {key}")
                return
            
            print(f"已设置 {key} = {value}")
            
        except ValueError:
            print("值格式错误")
    
    def do_test(self, arg: str) -> None:
        """
        运行测试 (完整多轮): test [verbose]
        
        test         - 运行测试，简洁输出
        test verbose - 运行测试，详细输出
        """
        if self.orchestrator is None:
            print("系统未初始化")
            return
        
        verbose = arg.strip().lower() == "verbose"
        
        print("\n🧪 运行测试...")
        print("=" * 60)
        
        test_cases = [
            "什么是人工智能?",
            "1 + 1 = ?",
            "如果A大于B，B大于C，那么A和C的关系是什么?"
        ]
        
        results_summary = []
        
        for i, q in enumerate(test_cases, 1):
            print(f"\n{'='*60}")
            print(f"📝 测试 {i}/{len(test_cases)}: {q}")
            print(f"{'='*60}")
            
            try:
                # 使用完整的 execute 而不是 execute_single_round
                result = self.orchestrator.execute(q, "general", verbose=verbose)
                
                answer_preview = result.final_answer[:100] + "..." if len(result.final_answer) > 100 else result.final_answer
                
                print(f"\n📊 测试结果:")
                print(f"   答案: {answer_preview}")
                print(f"   置信度: {result.confidence:.2f}")
                print(f"   总轮次: {result.total_rounds}")
                print(f"   质量分数: {result.metadata.get('quality_score', 'N/A')}")
                print(f"   正确性: {result.metadata.get('correctness', 'N/A')}")
                print(f"   停止原因: {result.metadata.get('stop_reason', 'N/A')}")
                
                results_summary.append({
                    "question": q[:30] + "..." if len(q) > 30 else q,
                    "rounds": result.total_rounds,
                    "confidence": result.confidence,
                    "quality": result.metadata.get('quality_score', 0),
                    "success": result.success
                })
                
            except Exception as e:
                print(f"❌ 错误: {e}")
                results_summary.append({
                    "question": q[:30] + "..." if len(q) > 30 else q,
                    "rounds": 0,
                    "confidence": 0,
                    "quality": 0,
                    "success": False
                })
        
        # 汇总表格
        print("\n" + "=" * 60)
        print("📋 测试汇总")
        print("=" * 60)
        print(f"{'问题':<35} {'轮次':>5} {'置信度':>8} {'质量':>8} {'状态':>6}")
        print("-" * 60)
        for r in results_summary:
            status = "✅" if r["success"] else "❌"
            print(f"{r['question']:<35} {r['rounds']:>5} {r['confidence']:>8.2f} {r['quality']:>8.2f} {status:>6}")
        
        print("-" * 60)
        success_count = sum(1 for r in results_summary if r["success"])
        print(f"通过: {success_count}/{len(results_summary)}")
        print("=" * 60)
    
    def do_quit(self, arg: str) -> bool:
        """退出系统: quit"""
        print("再见!")
        return True
    
    def do_exit(self, arg: str) -> bool:
        """退出系统: exit"""
        return self.do_quit(arg)
    
    def do_EOF(self, arg: str) -> bool:
        """处理Ctrl+D"""
        print()
        return self.do_quit(arg)
    
    def default(self, line: str) -> None:
        """处理未知命令"""
        print(f"未知命令: {line}")
        print("输入 help 查看可用命令")
    
    def emptyline(self) -> None:
        """处理空行"""
        pass


def main():
    """主函数"""
    try:
        shell = AgentShell()
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\n操作已取消")


if __name__ == "__main__":
    main()
