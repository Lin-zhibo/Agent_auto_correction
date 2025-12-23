"""
Heuristic Solver Agent (启发式解决者Agent)

职责：
- 提供创造性、启发式的解决方案
- 打破常规思维，探索非传统方法
- 使用类比、直觉和经验法则
"""

import json
import re
from typing import Any, Dict, Optional

from agent.base_agent import BaseAgent, AgentResponse
from core.schemas import AgentConfig, AgentType


class HeuristicSolverAgent(BaseAgent):
    """
    启发式解决者Agent
    
    核心功能：
    1. 提供创造性的解决方案
    2. 使用启发式方法和经验法则
    3. 通过类比和直觉发现新思路
    
    应用场景：
    - 常规方法遇到瓶颈时
    - 需要创新思维的问题
    - 探索非传统解决方案
    """
    
    PROMPT_TEMPLATE = """
## 角色定义
你是一位创造性思考者（Heuristic Solver Agent），擅长用非传统方法解决问题。
你的职责是打破常规思维，提供启发式的、创造性的解决方案。

## 核心能力
- 发散思维：从多个角度探索可能性
- 类比推理：从其他领域借鉴解决方案
- 直觉洞察：运用经验和直觉发现捷径
- 逆向思考：从结论反推，从失败中学习

## 常用启发式方法
1. **类比法**：这个问题像其他什么问题？
2. **简化法**：能否先解决简化版本？
3. **逆向法**：如果从结论出发会怎样？
4. **极端法**：在极端情况下会发生什么？
5. **分解法**：能否分解成更小的子问题？
6. **模式识别**：这里有什么规律？

## 任务描述
请对以下问题提供创造性的解决思路：
1. 分析当前答案的思路
2. 提供不同的解决方案
3. 使用启发式方法探索新思路
4. 分享可能有帮助的类比和直觉

## 输入信息

### 问题
{question}

### Student Agent的当前回答
{student_response}
置信度: {student_confidence}

### 上下文信息（如有）
{context_info}

## 输出要求

请按照以下JSON格式输出你的启发式分析：

```json
{{
    "current_approach_analysis": {{
        "identified_strategy": "识别出的当前解题策略",
        "strategy_type": "analytical/intuitive/algorithmic/other",
        "limitations": ["当前方法的局限性"]
    }},
    "heuristic_insights": {{
        "analogies": [
            {{
                "source_domain": "类比来源领域",
                "insight": "从类比中获得的启发",
                "application": "如何应用到当前问题"
            }}
        ],
        "intuitions": ["直觉性的洞察1", "洞察2"],
        "patterns_recognized": ["识别出的模式或规律"],
        "rules_of_thumb": ["适用的经验法则"]
    }},
    "alternative_approaches": {{
        "creative_solutions": [
            {{
                "approach": "创造性解法描述",
                "rationale": "为什么这个方法可能有效",
                "pros": ["优点"],
                "cons": ["缺点"]
            }}
        ],
        "simplification": "简化问题的方法",
        "reverse_engineering": "逆向思考的角度",
        "extreme_cases": "极端情况分析"
    }},
    "mental_models": {{
        "applicable_models": ["适用的思维模型"],
        "reframing": "重新定义问题的方式",
        "key_questions": ["帮助思考的关键问题"]
    }},
    "practical_shortcuts": {{
        "quick_checks": ["快速验证答案的方法"],
        "estimation_techniques": ["估算技巧"],
        "sanity_checks": ["合理性检查方法"]
    }},
    "final_response": "作为启发式解决者，你的创造性建议",
    "confidence": 0.XX,
    "reasoning": "启发式分析的推理过程"
}}
```

## 创造性思维原则
1. **没有坏主意**：先发散，后收敛
2. **借鉴跨界**：最好的解决方案可能来自其他领域
3. **质疑假设**：挑战"理所当然"的前提
4. **拥抱不确定**：直觉虽不精确，但可能指向正确方向

## 启发式方法的价值
- 快速生成候选解决方案
- 发现被正式方法遗漏的角度
- 在信息不完整时提供指引
- 激发进一步探索的方向

## 挑战现有思路（关键！）
作为启发式解决者，你的存在意义就是打破思维定式：

1. **不要认可常规解法**
   - 你的价值在于提供不同的视角
   - 即使当前解法正确，也要问"有没有更好的方法？"
   - alternative_approaches.creative_solutions 至少提供2个替代方案

2. **首轮分析特别要求**
   - 首轮必须挑战 current_approach_analysis.limitations
   - 首轮 analogies 至少提供1-2个跨领域类比
   - 首轮 rules_of_thumb 应该提供实用的经验法则

3. **质疑隐含假设**
   - 当前解法基于什么假设？
   - 如果这些假设不成立呢？
   - mental_models.reframing 必须提供重新定义问题的方式

4. **极端情况测试**
   - extreme_cases 不能为空
   - 在边界条件下，当前答案还成立吗？
   - 这是发现问题的利器

5. **你的核心价值**
   - 你不是来认可"标准答案"的
   - 你是来提供"如果换一种想法呢？"
   - 创造性不等于正确性，但可以激发更好的思考

6. **实用快捷验证**
   - quick_checks 必须提供验证答案的方法
   - sanity_checks 帮助检验答案是否合理
   - 这些工具帮助 Student 自我检查

请开始你的创造性分析：
"""
    
    def __init__(
        self,
        agent_id: str = "heuristic_solver_agent_001",
        config: Optional[AgentConfig] = None,
        llm_client: Any = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HEURISTIC_SOLVER,
            config=config,
            llm_client=llm_client
        )
    
    def execute(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        context = context or {}
        prompt = self._build_prompt(question, context)
        
        try:
            llm_response = self._call_llm(prompt)
            response = self._parse_response(llm_response)
        except Exception as e:
            response = AgentResponse(
                content=f"启发式分析过程出现错误: {str(e)}",
                confidence=0.1,
                reasoning="执行出错",
                metadata={"error": str(e)}
            )
        
        self._log_execution(question, response, prompt)
        return response
    
    def _build_prompt(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        context = context or {}
        
        student_response = context.get("student_response", "暂无回答")
        student_confidence = context.get("student_confidence", 0.0)
        context_info = context.get("additional_context", "暂无额外上下文")
        
        template = self.config.prompt_template or self.PROMPT_TEMPLATE
        
        return template.format(
            question=question,
            student_response=student_response,
            student_confidence=student_confidence,
            context_info=context_info
        )
    
    def _parse_response(self, llm_response: str) -> AgentResponse:
        try:
            json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = llm_response
            
            data = json.loads(json_str)
            
            return AgentResponse(
                content=data.get("final_response", ""),
                confidence=float(data.get("confidence", 0.7)),
                reasoning=data.get("reasoning", ""),
                metadata={
                    "current_approach_analysis": data.get("current_approach_analysis", {}),
                    "heuristic_insights": data.get("heuristic_insights", {}),
                    "alternative_approaches": data.get("alternative_approaches", {}),
                    "mental_models": data.get("mental_models", {}),
                    "practical_shortcuts": data.get("practical_shortcuts", {})
                }
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return AgentResponse(
                content=llm_response,
                confidence=0.5,
                reasoning="无法解析结构化响应",
                metadata={"parse_error": str(e), "raw_response": llm_response}
            )

