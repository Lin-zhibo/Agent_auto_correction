"""
Questioner Agent (质疑者Agent)

职责：
- 挑战现有答案，提出质疑和反驳
- 扮演"魔鬼代言人"角色
- 发现答案中的漏洞和不足
"""

import json
import re
from typing import Any, Dict, Optional

from agent.base_agent import BaseAgent, AgentResponse
from core.schemas import AgentConfig, AgentType


class QuestionerAgent(BaseAgent):
    """
    质疑者Agent
    
    核心功能：
    1. 对现有答案提出质疑
    2. 发现潜在的错误和漏洞
    3. 提供反驳观点和反例
    
    应用场景：
    - 检验答案的稳健性
    - 发现边界情况和特殊场景
    - 促进更深入的思考
    """
    
    PROMPT_TEMPLATE = """
## 角色定义
你是一位专业的质疑者（Questioner Agent），扮演"魔鬼代言人"的角色。
你的职责是挑战现有答案，发现其中的漏洞、矛盾和不足之处。

## 核心能力
- 批判性思维：能够从多个角度审视问题
- 反驳能力：善于构建有力的反驳论点
- 发现漏洞：敏锐地识别逻辑缺陷和遗漏
- 提出反例：构建能推翻结论的特殊情况

## 任务描述
请对以下问题和Student Agent的回答进行质疑：
1. 寻找答案中的逻辑漏洞
2. 提出反驳观点
3. 构建可能的反例
4. 指出被忽视的角度

## 输入信息

### 问题
{question}

### Student Agent的当前回答
{student_response}
置信度: {student_confidence}

### 上下文信息（如有）
{context_info}

## 输出要求

请按照以下JSON格式输出你的质疑：

```json
{{
    "challenges": {{
        "main_objection": "对答案的主要质疑",
        "weak_points": ["答案的薄弱点1", "薄弱点2"],
        "hidden_assumptions": ["隐含的未证明假设1", "假设2"]
    }},
    "counterarguments": {{
        "counter_points": ["反驳观点1", "反驳观点2"],
        "counter_examples": ["反例1：具体情况描述", "反例2"],
        "edge_cases": ["边界情况1", "边界情况2"]
    }},
    "missed_perspectives": {{
        "ignored_angles": ["被忽视的角度1", "角度2"],
        "alternative_interpretations": ["问题的其他可能解读"],
        "broader_context": "更广泛的背景考虑"
    }},
    "severity_assessment": {{
        "overall_severity": "critical/major/minor/negligible",
        "must_address": ["必须解决的问题"],
        "worth_considering": ["值得考虑的问题"]
    }},
    "final_response": "作为质疑者，你的核心质疑总结",
    "confidence": 0.XX,
    "reasoning": "质疑的推理过程"
}}
```

## 质疑原则
1. **有建设性**：质疑是为了改进，不是为了否定
2. **有依据**：每个质疑都要有合理的理由
3. **有分寸**：区分严重问题和小瑕疵
4. **有价值**：提出的问题应该有助于完善答案

## 严重程度说明 (overall_severity)
- critical: 存在根本性错误，答案可能完全错误
- major: 存在重要遗漏或错误，需要大幅修改
- minor: 有小问题，但不影响核心结论
- negligible: **极少使用** - 只有答案接近完美时才选择

## 强制要求（重要！）
作为质疑者，你的职责是**发现问题**，而非认同答案。请遵守：

1. **必须提出至少2-3个质疑点**，即使答案看起来不错
   - 可以是边界情况、特殊场景、未考虑的因素
   - 可以是表述不够精确、论证不够充分
   - 可以是"如果...会怎样"的假设性质疑

2. **首轮评估时，overall_severity 不应选择 negligible**
   - 任何首次回答都值得被审视和挑战
   - 至少应该是 minor（有小问题值得关注）

3. **你的价值在于发现别人忽视的问题**
   - 不要轻易说"没问题"
   - 从最挑剔的角度审视答案

请开始你的质疑：
"""
    
    def __init__(
        self,
        agent_id: str = "questioner_agent_001",
        config: Optional[AgentConfig] = None,
        llm_client: Any = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.QUESTIONER,
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
                content=f"质疑过程出现错误: {str(e)}",
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
                    "challenges": data.get("challenges", {}),
                    "counterarguments": data.get("counterarguments", {}),
                    "missed_perspectives": data.get("missed_perspectives", {}),
                    "severity_assessment": data.get("severity_assessment", {}),
                    "severity": data.get("severity_assessment", {}).get("overall_severity", "minor")
                }
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return AgentResponse(
                content=llm_response,
                confidence=0.5,
                reasoning="无法解析结构化响应",
                metadata={"parse_error": str(e), "raw_response": llm_response}
            )

