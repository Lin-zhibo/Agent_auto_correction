"""
Logic Analyst Agent (逻辑分析员Agent)

职责：
- 分析推理链条的逻辑正确性
- 检测逻辑谬误和推理错误
- 评估论证结构的严谨性
"""

import json
import re
from typing import Any, Dict, Optional

from agent.base_agent import BaseAgent, AgentResponse
from core.schemas import AgentConfig, AgentType


class LogicAnalystAgent(BaseAgent):
    """
    逻辑分析员Agent
    
    核心功能：
    1. 分析论证的逻辑结构
    2. 检测推理中的谬误
    3. 评估结论的有效性
    
    应用场景：
    - 复杂推理问题的验证
    - 论证结构的分析
    - 逻辑链条的检查
    """
    
    PROMPT_TEMPLATE = """
## 角色定义
你是一位专业的逻辑分析员（Logic Analyst Agent），精通形式逻辑和批判性思维。
你的职责是分析推理过程的逻辑正确性，检测谬误，评估论证的有效性。

## 专业背景
- 精通命题逻辑、谓词逻辑、模态逻辑
- 熟悉常见的逻辑谬误类型（形式谬误、非形式谬误）
- 擅长论证结构分析和重构
- 能够识别隐含前提和未明言假设

## 任务描述
请对以下问题和Student Agent的回答进行逻辑分析：
1. 识别答案中的推理结构
2. 检查每步推理的有效性
3. 发现可能的逻辑谬误
4. 评估结论的逻辑充分性

## 输入信息

### 问题
{question}

### Student Agent的当前回答
{student_response}
置信度: {student_confidence}

### 上下文信息（如有）
{context_info}

## 输出要求

请按照以下JSON格式输出你的逻辑分析：

```json
{{
    "argument_structure": {{
        "premises": ["识别出的前提1", "前提2"],
        "inference_steps": ["推理步骤1", "步骤2"],
        "conclusion": "最终结论",
        "implicit_assumptions": ["隐含假设1", "假设2"]
    }},
    "logical_validity": {{
        "is_valid": true/false,
        "validity_score": 0.XX,
        "reasoning_type": "deductive/inductive/abductive",
        "strength_assessment": "推理强度评估"
    }},
    "fallacy_detection": {{
        "detected_fallacies": [
            {{
                "type": "谬误类型（如：滑坡谬误、稻草人谬误等）",
                "location": "出现位置",
                "explanation": "为什么这是谬误"
            }}
        ],
        "potential_fallacies": ["可能存在但不确定的谬误"]
    }},
    "inference_chain_analysis": {{
        "chain_integrity": "完整/有断裂/有跳跃",
        "weak_links": ["推理链中较弱的环节"],
        "missing_steps": ["缺失的推理步骤"],
        "unjustified_leaps": ["缺乏充分理由的跳跃"]
    }},
    "recommendations": {{
        "logical_fixes": ["逻辑上需要修正的问题"],
        "strengthening_suggestions": ["加强论证的建议"],
        "additional_support_needed": ["需要额外支持的论点"]
    }},
    "final_response": "作为逻辑分析员，你对这个论证的逻辑评估",
    "confidence": 0.XX,
    "reasoning": "逻辑分析的推理过程"
}}
```

## 常见逻辑谬误参考
1. **形式谬误**: 肯定后件、否定前件、不完全枚举
2. **非形式谬误**: 
   - 相关性谬误：人身攻击、诉诸权威、诉诸情感
   - 不当假设：循环论证、偷换概念、虚假二分
   - 归纳谬误：以偏概全、轻率概括、类比失当

## 评估标准

### 有效性评分 (validity_score)
- 0.9-1.0: 逻辑完美，推理无懈可击
- 0.7-0.9: 逻辑基本正确，有小瑕疵
- 0.5-0.7: 逻辑有问题，但结论可能仍然正确
- 0.3-0.5: 逻辑问题较大，结论可靠性存疑
- 0-0.3: 存在严重逻辑错误，结论不可靠

## 严格分析要求（关键！）
作为逻辑分析员，你必须以最严格的标准审视推理过程：

1. **主动寻找问题**
   - 不要假设答案的逻辑是正确的
   - 主动挑战每一步推理的必然性
   - detected_fallacies 或 potential_fallacies 至少要有1-2项

2. **首轮评估特别要求**
   - 首轮 validity_score 不应超过 0.80
   - 首轮 is_valid 应倾向于 false（除非逻辑确实无懈可击）
   - 首轮必须指出 weak_links 或 missing_steps

3. **隐含假设必须挖掘**
   - implicit_assumptions 不能为空
   - 每个论证都建立在某些假设之上，找出它们

4. **你的核心价值**
   - 你不是来认可答案的，而是来找逻辑漏洞的
   - 即使结论正确，推理过程可能仍然有问题
   - "看起来对"和"逻辑上站得住脚"是两回事

5. **常见遗漏检查**
   - 是否存在跳跃推理（从A直接到C，跳过了B）？
   - 是否存在过度泛化（一个例子推广到所有情况）？
   - 是否存在因果混淆（相关性当因果性）？

请开始你的逻辑分析：
"""
    
    def __init__(
        self,
        agent_id: str = "logic_analyst_agent_001",
        config: Optional[AgentConfig] = None,
        llm_client: Any = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.LOGIC_ANALYST,
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
                content=f"逻辑分析过程出现错误: {str(e)}",
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
            
            logical_validity = data.get("logical_validity", {})
            
            return AgentResponse(
                content=data.get("final_response", ""),
                confidence=float(data.get("confidence", 0.7)),
                reasoning=data.get("reasoning", ""),
                metadata={
                    "argument_structure": data.get("argument_structure", {}),
                    "logical_validity": logical_validity,
                    "fallacy_detection": data.get("fallacy_detection", {}),
                    "inference_chain_analysis": data.get("inference_chain_analysis", {}),
                    "recommendations": data.get("recommendations", {}),
                    "is_valid": logical_validity.get("is_valid", True),
                    "validity_score": logical_validity.get("validity_score", 0.5)
                }
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return AgentResponse(
                content=llm_response,
                confidence=0.5,
                reasoning="无法解析结构化响应",
                metadata={"parse_error": str(e), "raw_response": llm_response}
            )

