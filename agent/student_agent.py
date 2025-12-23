"""
Student Agent (学生Agent)

职责：
- 作为主答题者，首先对问题进行作答
- 结合长期记忆中的知识经验
- 生成带有置信度的答案
"""

import json
import re
from typing import Any, Dict, Optional

from agent.base_agent import BaseAgent, AgentResponse
from core.schemas import AgentConfig, AgentType


class StudentAgent(BaseAgent):
    """
    学生Agent
    
    核心功能：
    1. 对输入问题进行初步作答
    2. 结合LTM中的相关知识
    3. 根据反馈进行自我修正
    
    应用场景：
    - 通用问答任务
    - 数学/逻辑推理
    - 知识检索与应用
    """
    
    # 提示词模板
    PROMPT_TEMPLATE = """
## 角色定义
你是一个严谨的问题解答者（Student Agent），负责对给定的问题进行深入分析和作答。
你需要展现出"学习者"的特质：谨慎思考、承认不确定性、基于证据推理。

## 任务描述
请针对以下问题进行分析并给出答案。你需要：
1. 仔细理解问题的含义和要求
2. 运用相关知识进行推理
3. 给出明确的答案
4. 评估自己答案的置信度

## 输入信息

### 问题
{question}

### 相关背景知识（来自长期记忆）
{ltm_context}

### 历史反馈（如有）
{feedback_context}

## 输出要求

请按照以下JSON格式输出你的回答：

```json
{{
    "analysis": "对问题的理解和分析过程",
    "reasoning": "详细的推理步骤，展示你的思考过程",
    "answer": "最终答案（简洁明确）",
    "confidence": 0.XX,
    "uncertainty_factors": ["可能影响答案准确性的因素1", "因素2"],
    "knowledge_gaps": ["需要补充的知识点"]
}}
```

## 置信度评估标准（重要！请严格遵守）

confidence（置信度）取值范围为0-1，请**保守评估**：
- 0.9-1.0：**极少使用** - 仅当答案有明确的公理/定义支撑，且无任何争议时
- 0.7-0.9：比较确定，有较强的推理支持，但仍可能遗漏某些角度
- 0.5-0.7：中等确定（**这是大多数首次回答的合理范围**）
- 0.3-0.5：较不确定，推理过程可能存在漏洞
- 0-0.3：非常不确定，仅为初步猜测

**关键原则**：
1. 首次回答（无反馈时）的置信度通常**不应超过0.75**，因为你还没有经过专家审视
2. 只有在整合反馈并确认无误后，置信度才应提升到0.8以上
3. 保持谦逊：承认"我可能遗漏了某些角度"是好的学习态度

## 其他注意事项
1. 即使不确定，也要给出你认为最可能的答案，而不是拒绝回答
2. reasoning部分要详细，展示完整的推理链条
3. 如果发现与之前的反馈存在矛盾，请明确说明并解释你的选择
4. uncertainty_factors 至少列出1-2个可能影响答案的因素

请开始你的分析：
"""
    
    def __init__(
        self,
        agent_id: str = "student_agent_001",
        config: Optional[AgentConfig] = None,
        llm_client: Any = None
    ):
        """
        初始化Student Agent
        
        Args:
            agent_id: Agent唯一标识
            config: Agent配置
            llm_client: LLM客户端
        """
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.STUDENT,
            config=config,
            llm_client=llm_client
        )
    
    def execute(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        执行问题解答
        
        Args:
            question: 输入问题
            context: 上下文信息，可包含：
                - ltm_knowledge: 长期记忆中的相关知识
                - previous_feedback: 之前的反馈信息
                - round_index: 当前轮次
                
        Returns:
            AgentResponse: 包含答案、置信度和推理过程
        """
        context = context or {}
        
        # 构建提示词
        prompt = self._build_prompt(question, context)
        
        # 调用LLM
        try:
            llm_response = self._call_llm(prompt)
            response = self._parse_response(llm_response)
        except Exception as e:
            # 异常处理：返回低置信度的默认响应
            response = AgentResponse(
                content=f"处理过程中出现错误: {str(e)}",
                confidence=0.1,
                reasoning="执行出错，无法完成推理",
                metadata={"error": str(e)}
            )
        
        # 记录执行历史
        self._log_execution(question, response, prompt)
        
        return response
    
    def _build_prompt(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        构建提示词
        
        Args:
            question: 输入问题
            context: 上下文信息
            
        Returns:
            格式化后的提示词
        """
        context = context or {}
        
        # 处理长期记忆上下文
        ltm_context = self._format_ltm_context(context.get("ltm_knowledge"))
        
        # 处理反馈上下文
        feedback_context = self._format_feedback_context(
            context.get("previous_feedback"),
            context.get("round_index", 1)
        )
        
        # 使用模板或配置中的提示词
        template = self.config.prompt_template or self.PROMPT_TEMPLATE
        
        return template.format(
            question=question,
            ltm_context=ltm_context,
            feedback_context=feedback_context
        )
    
    def _format_ltm_context(self, ltm_knowledge: Optional[Dict[str, Any]]) -> str:
        """格式化长期记忆上下文"""
        if not ltm_knowledge:
            return "暂无相关背景知识"
        
        parts = []
        
        # 纠正策略
        if ltm_knowledge.get("correction_strategies"):
            strategies = ltm_knowledge["correction_strategies"]
            parts.append("【纠正策略】\n" + "\n".join(f"- {s}" for s in strategies))
        
        # 思考角度
        if ltm_knowledge.get("thinking_angles"):
            angles = ltm_knowledge["thinking_angles"]
            parts.append("【推荐思考角度】\n" + "\n".join(f"- {a}" for a in angles))
        
        # 历史错误案例
        if ltm_knowledge.get("error_examples"):
            examples = ltm_knowledge["error_examples"][:3]  # 最多3个
            examples_text = []
            for ex in examples:
                examples_text.append(
                    f"  问题: {ex.get('question', 'N/A')}\n"
                    f"  错误答案: {ex.get('wrong_answer', 'N/A')}\n"
                    f"  正确答案: {ex.get('correct_answer', 'N/A')}\n"
                    f"  错误分析: {ex.get('error_analysis', 'N/A')}"
                )
            parts.append("【历史错误案例参考】\n" + "\n\n".join(examples_text))
        
        return "\n\n".join(parts) if parts else "暂无相关背景知识"
    
    def _format_feedback_context(
        self,
        previous_feedback: Optional[Dict[str, Any]],
        round_index: int
    ) -> str:
        """格式化反馈上下文"""
        if round_index == 1 or not previous_feedback:
            return "这是第一轮回答，暂无历史反馈"
        
        parts = [f"当前是第{round_index}轮回答，请参考以下反馈进行改进："]
        
        # 上一轮答案
        if previous_feedback.get("previous_answer"):
            parts.append(f"【上一轮答案】\n{previous_feedback['previous_answer']}")
        
        # 冲突点
        if previous_feedback.get("conflicts"):
            conflicts = previous_feedback["conflicts"]
            parts.append("【检测到的冲突点】\n" + "\n".join(f"- {c}" for c in conflicts))
        
        # 互补观点
        if previous_feedback.get("complementary_points"):
            points = previous_feedback["complementary_points"]
            parts.append("【其他Agent的互补观点】\n" + "\n".join(f"- {p}" for p in points))
        
        # Insight Agent的综合反馈
        if previous_feedback.get("insight_feedback"):
            parts.append(f"【综合反馈】\n{previous_feedback['insight_feedback']}")
        
        return "\n\n".join(parts)
    
    def _parse_response(self, llm_response: str) -> AgentResponse:
        """
        解析LLM响应
        
        Args:
            llm_response: LLM原始响应文本
            
        Returns:
            解析后的AgentResponse
        """
        try:
            # 尝试提取JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # 尝试直接解析
                json_str = llm_response
            
            data = json.loads(json_str)
            
            return AgentResponse(
                content=data.get("answer", ""),
                confidence=float(data.get("confidence", 0.5)),
                reasoning=data.get("reasoning", ""),
                metadata={
                    "analysis": data.get("analysis", ""),
                    "uncertainty_factors": data.get("uncertainty_factors", []),
                    "knowledge_gaps": data.get("knowledge_gaps", [])
                }
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # 解析失败，使用原始响应
            return AgentResponse(
                content=llm_response,
                confidence=0.5,
                reasoning="无法解析结构化响应",
                metadata={"parse_error": str(e), "raw_response": llm_response}
            )
    
    def execute_with_refinement(
        self,
        question: str,
        feedback: Dict[str, Any],
        original_response: AgentResponse
    ) -> AgentResponse:
        """
        基于反馈进行答案修正
        
        Args:
            question: 原始问题
            feedback: 反馈信息
            original_response: 原始响应
            
        Returns:
            修正后的AgentResponse
        """
        context = {
            "previous_feedback": {
                "previous_answer": original_response.content,
                "conflicts": feedback.get("conflicts", []),
                "complementary_points": feedback.get("complementary_points", []),
                "insight_feedback": feedback.get("insight_feedback", "")
            },
            "round_index": feedback.get("round_index", 2),
            "ltm_knowledge": feedback.get("ltm_knowledge")
        }
        
        return self.execute(question, context)

