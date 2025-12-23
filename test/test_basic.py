"""
基础功能测试

测试系统的核心功能是否正常工作
"""

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import_models():
    """测试数据模型导入"""
    from core.schemas import (
        WorkingMemory,
        LongTermMemoryEntry,
        MetaKnowledge,
        AgentType,
        AgentOutput,
    )
    
    # 测试创建实例
    wm = WorkingMemory(task_id="test_001", question_text="测试问题")
    assert wm.task_id == "test_001"
    assert wm.round_index == 1
    
    mk = MetaKnowledge(question_type="general")
    assert mk.max_rounds == 10
    assert mk.confidence_threshold == 0.8
    
    print("✓ 数据模型测试通过")


def test_import_memory():
    """测试记忆模块导入"""
    from memory.working_memory import WorkingMemoryManager
    from memory.long_term_memory import LongTermMemoryManager
    from memory.meta_knowledge import MetaKnowledgeManager
    
    # 测试创建实例
    wm_manager = WorkingMemoryManager("test/tmp/wm")
    ltm_manager = LongTermMemoryManager("test/tmp/ltm")
    mk_manager = MetaKnowledgeManager("test/tmp/mk")
    
    print("✓ 记忆模块测试通过")


def test_import_agents():
    """测试Agent模块导入"""
    from agent.base_agent import BaseAgent, AgentResponse
    from agent.student_agent import StudentAgent
    from agent.insight_agent import InsightAgent
    from agent.multi_agents.authority_agent import AuthorityAgent
    
    print("✓ Agent模块测试通过")


def test_import_core():
    """测试核心模块导入"""
    from core.orchestrator import Orchestrator
    from core.loop_controller import LoopController
    from core.memory_coordinator import MemoryCoordinator
    
    print("✓ 核心模块测试通过")


def test_meta_knowledge_loop_calculation():
    """测试元知识的循环次数计算"""
    from core.schemas import MetaKnowledge
    
    mk = MetaKnowledge(
        question_type="test",
        error_rate=0.5,
        max_rounds=10,
        confidence_threshold=0.8,
        alpha=1.0,
        beta=1.0
    )
    
    # 测试不同置信度下的循环次数计算
    # 置信度0，应该接近最大轮次
    rounds_0 = mk.calculate_max_rounds(0.0)
    assert rounds_0 > 0
    
    # 置信度1，应该很少轮次
    rounds_1 = mk.calculate_max_rounds(1.0)
    assert rounds_1 >= 1
    
    # 测试停止条件
    assert mk.should_stop(0.9, 1) == True  # 置信度超过阈值
    assert mk.should_stop(0.5, 10) == True  # 达到最大轮次
    assert mk.should_stop(0.5, 5) == False  # 继续执行
    
    print("✓ 元知识循环计算测试通过")


def test_working_memory_operations():
    """测试工作记忆操作"""
    from core.schemas import WorkingMemory, AgentOutput, AgentType
    from datetime import datetime
    
    wm = WorkingMemory(task_id="test_001", question_text="测试问题")
    
    # 添加Agent输出
    output = AgentOutput(
        agent_id="test_agent",
        agent_type=AgentType.STUDENT,
        response="测试回答",
        confidence=0.8,
        reasoning="测试推理"
    )
    wm.add_agent_output(output)
    
    assert len(wm.agent_outputs) == 1
    assert wm.agent_outputs[0].confidence == 0.8
    
    # 测试轮次推进
    wm.advance_round()
    assert wm.round_index == 2
    assert len(wm.agent_outputs) == 0  # 输出已清空
    assert len(wm.round_history) == 1  # 历史已保存
    
    print("✓ 工作记忆操作测试通过")


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("运行基础功能测试")
    print("=" * 50 + "\n")
    
    tests = [
        test_import_models,
        test_import_memory,
        test_import_agents,
        test_import_core,
        test_meta_knowledge_loop_calculation,
        test_working_memory_operations,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} 失败: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
