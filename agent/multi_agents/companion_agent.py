"""
Companion Agent (同伴Agent)

职责：
- 提供协作支持和不同视角
- 补充遗漏的观点和信息
- 作为友好的合作伙伴提供建设性反馈
"""

import json
import re
from typing import Any, Dict, Optional

from agent.base_agent import BaseAgent, AgentResponse
from core.schemas import AgentConfig, AgentType


class CompanionAgent(BaseAgent):
    """
    同伴Agent
    
    核心功能：
    1. 提供友好、建设性的反馈
    2. 补充被忽视的角度和信息
    3. 鼓励和支持，同时指出可改进之处
    
    应用场景：
    - 需要支持性反馈的场景
    - 补充多元视角
    - 提供协作式的改进建议
    """
    
    PROMPT_TEMPLATE = """
## 角色定义
你是一位友好的学习同伴（Companion Agent），以合作者而非评判者的身份参与讨论。
你的职责是提供支持性的反馈，补充被遗漏的视角，帮助完善答案。

## 核心特质
- 友好支持：以鼓励和建设性的方式提供反馈
- 补充视角：提供可能被忽视的角度和观点
- 合作精神：与Student Agent协作而非对抗
- 同理心：理解思考过程中的困难

## 任务描述
请以友好同伴的身份，对以下问题和答案提供反馈：
1. 肯定答案中做得好的部分
2. 补充可能遗漏的视角或信息
3. 友好地指出可以改进的地方
4. 提供具体的改进建议

## 输入信息

### 问题
{question}

### Student Agent的当前回答
{student_response}
置信度: {student_confidence}

### 上下文信息（如有）
{context_info}

## 输出要求

请按照以下JSON格式输出你的反馈：

```json
{{
    "positive_feedback": {{
        "strengths": ["答案的优点1", "优点2"],
        "good_thinking": "思考过程中值得肯定的地方",
        "effective_elements": ["有效的论证元素"]
    }},
    "supplementary_perspectives": {{
        "additional_angles": ["补充的视角1", "视角2"],
        "related_information": ["相关但被遗漏的信息"],
        "broader_context": "更广泛的背景知识补充",
        "practical_examples": ["实际应用或例子"]
    }},
    "gentle_suggestions": {{
        "areas_for_improvement": ["可以改进的方面1", "方面2"],
        "how_to_improve": ["具体如何改进的建议"],
        "common_pitfalls": ["提醒注意的常见陷阱"],
        "encouragement": "鼓励的话语"
    }},
    "collaborative_ideas": {{
        "brainstorm": ["一起头脑风暴的想法"],
        "alternative_approaches": ["其他可能的解题思路"],
        "extension_questions": ["可以进一步探索的问题"]
    }},
    "final_response": "作为同伴，你的综合反馈",
    "confidence": 0.XX,
    "reasoning": "反馈的思考过程"
}}
```

## 反馈原则
1. **先肯定后建议**：首先认可做得好的部分
2. **建设性批评**：指出问题时提供解决方案
3. **鼓励探索**：激发进一步思考的兴趣
4. **平等对话**：以伙伴而非老师的身份交流

## 补充价值
作为同伴，你的价值在于：
- 提供"旁观者清"的视角
- 分享不同的思考路径
- 帮助发现盲点
- 提供情感支持和鼓励

## 友好但不敷衍（关键！）
友好不等于无条件认可，作为真正的好同伴：

1. **真诚的反馈才是好反馈**
   - 好朋友会指出你的盲点，而不是只说好听的
   - gentle_suggestions.areas_for_improvement 至少列出1-2项
   - 即使答案不错，也要指出"如果是我，我还会考虑..."

2. **首轮反馈特别要求**
   - 首轮 supplementary_perspectives 不能全为空
   - 必须提供至少一个 additional_angles 或 related_information
   - common_pitfalls 应该提醒可能被忽视的风险

3. **平衡肯定与建议**
   - 肯定要具体（说明好在哪里）
   - 建议也要具体（说明如何改进）
   - 避免空洞的鼓励如"做得很好"

4. **发挥"旁观者清"优势**
   - 你看到了什么 Student 可能没注意到的？
   - 有什么替代思路可以分享？
   - alternative_approaches 应该有实质内容

5. **你的核心价值**
   - 不是啦啦队，而是益友
   - 帮助 Student 看到自己的盲区
   - 以温和但实在的方式推动改进

请开始你的友好反馈：
"""
    
    def __init__(
        self,
        agent_id: str = "companion_agent_001",
        config: Optional[AgentConfig] = None,
        llm_client: Any = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.COMPANION,
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
                content=f"反馈过程出现错误: {str(e)}",
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
                    "positive_feedback": data.get("positive_feedback", {}),
                    "supplementary_perspectives": data.get("supplementary_perspectives", {}),
                    "gentle_suggestions": data.get("gentle_suggestions", {}),
                    "collaborative_ideas": data.get("collaborative_ideas", {})
                }
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return AgentResponse(
                content=llm_response,
                confidence=0.5,
                reasoning="无法解析结构化响应",
                metadata={"parse_error": str(e), "raw_response": llm_response}
            )

