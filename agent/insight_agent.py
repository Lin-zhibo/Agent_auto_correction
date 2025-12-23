"""
Insight Agent (洞察Agent)

职责：
- 汇总各Agent的输出
- 提取角度、证据、冲突、互补等信息
- 生成反馈提示词指导Student Agent改进
"""

import json
import re
from typing import Any, Dict, List, Optional

from agent.base_agent import BaseAgent, AgentResponse
from core.schemas import AgentConfig, AgentOutput, AgentType


class InsightAgent(BaseAgent):
    """
    洞察Agent
    
    核心功能：
    1. 分析多Agent输出，识别共识和分歧
    2. 提取关键角度和证据
    3. 生成综合性反馈
    
    应用场景：
    - 多Agent协作的汇总阶段
    - 冲突检测与解决
    - 反馈生成
    """
    
    # 提示词模板
    PROMPT_TEMPLATE = """
## 角色定义
你是一个Meta-Synthesizer（元综合者），也称为Insight Agent。
你的核心任务是：
1. 分析多个Agent的输出，提取有价值的洞察
2. **评估Student Agent答案的质量和正确性**
3. 生成指导性反馈

## 任务描述
请分析以下多个Agent针对同一问题的回答，完成以下任务：
1. 识别各Agent观点的共识和分歧
2. 提取各角度的关键证据和推理
3. 检测潜在的逻辑冲突
4. 发现互补的观点
5. **综合评估Student Agent答案的质量（0-1分）**
6. 生成综合性的反馈提示词

## 输入信息

### 原始问题
{question}

### Student Agent当前答案
{student_response}
置信度: {student_confidence}

### 当前轮次
第 {current_round} 轮

### 各专家Agent的反馈
{agent_outputs}

## 输出要求

请按照以下JSON格式输出你的分析结果：

```json
{{
    "answer_evaluation": {{
        "quality_score": 0.XX,
        "correctness": "correct/partially_correct/incorrect/uncertain",
        "completeness": "complete/partial/incomplete",
        "accuracy_issues": ["准确性问题1", "问题2"],
        "missing_points": ["遗漏的要点1", "要点2"]
    }},
    "consensus_points": [
        "多数Agent认同的观点1",
        "多数Agent认同的观点2"
    ],
    "divergence_points": [
        {{
            "topic": "分歧话题",
            "perspectives": [
                {{"agent": "agent_name", "view": "观点描述"}}
            ]
        }}
    ],
    "conflicts_detected": [
        "检测到的逻辑冲突或矛盾1",
        "检测到的逻辑冲突或矛盾2"
    ],
    "complementary_insights": [
        "互补观点1：可以丰富答案的角度",
        "互补观点2"
    ],
    "key_evidence": [
        "关键证据或论据1",
        "关键证据或论据2"
    ],
    "thinking_angles": [
        "建议考虑的思考角度1",
        "建议考虑的思考角度2"
    ],
    "feedback_prompt": "针对Student Agent的综合反馈，指导其如何改进答案",
    "recommended_action": "建议的下一步行动：refine(继续修正)/accept(接受当前答案)/escalate(需要人工介入)",
    "confidence_adjustment": 0.XX,
    "reasoning": "得出上述结论的推理过程"
}}
```

## 答案评估标准

### quality_score 评分标准（0-1）- 请严格评估！
- 0.9-1.0: 优秀 - **极少给出** - 答案不仅正确，还要完整、精确、考虑周全
- 0.7-0.9: 良好 - 基本正确，但仍有改进空间
- 0.5-0.7: 中等 - **这是大多数首轮答案的合理评分**
- 0.3-0.5: 较差 - 错误较多，需要大幅修正
- 0.0-0.3: 很差 - 根本性错误

**评分校准原则**：
- 首轮答案（round=1）的 quality_score 通常**不应超过0.75**
- 只有经过反馈改进后的答案，才有可能达到0.8以上
- 综合考虑各专家Agent的反馈，如果有任何Agent提出重要问题，应降低评分

### correctness 判断标准
- correct: 答案正确，与问题要求相符
- partially_correct: 部分正确，有些要点正确但存在错误
- incorrect: 答案错误
- uncertain: 无法确定（问题本身有歧义或缺乏足够信息）

### recommended_action 决策标准
- **accept**: 仅当满足以下所有条件时:
  1. quality_score >= 0.85
  2. correctness 为 correct
  3. 当前轮次 >= 2（首轮不应直接accept，应给Student机会改进）
- **refine**: 以下任一情况:
  1. 当前是第1轮（即使答案看起来不错，也建议refine以确保充分考虑）
  2. quality_score < 0.85
  3. correctness 为 partially_correct 或 uncertain
  4. 存在明显的改进空间
- **escalate**: 问题过于复杂或存在根本性分歧，需要人工介入

## 分析原则

1. **客观公正**：不偏向任何单一Agent，综合考虑所有观点
2. **证据优先**：基于具体证据和推理，而非主观判断
3. **建设性**：反馈应具有可操作性，帮助Student Agent改进
4. **谨慎判断**：对于不确定的内容，明确标注不确定性
5. **质量优先**：只有当答案真正达到可接受标准时才建议accept

## 注意事项
1. conflicts_detected应只包含实质性的逻辑冲突，不要将不同表述误判为冲突
2. feedback_prompt应简洁有力，直接指出改进方向
3. confidence_adjustment是对Student Agent置信度的调整建议（-0.3到+0.3）
4. **quality_score 是核心评估指标，请认真评估**

请开始你的分析：
"""
    
    def __init__(
        self,
        agent_id: str = "insight_agent_001",
        config: Optional[AgentConfig] = None,
        llm_client: Any = None
    ):
        """
        初始化Insight Agent
        
        Args:
            agent_id: Agent唯一标识
            config: Agent配置
            llm_client: LLM客户端
        """
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.INSIGHT,
            config=config,
            llm_client=llm_client
        )
    
    def execute(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        执行洞察分析
        
        Args:
            question: 原始问题
            context: 上下文信息，应包含：
                - student_response: Student Agent的回答
                - student_confidence: Student Agent的置信度
                - agent_outputs: 各Agent的输出列表
                
        Returns:
            AgentResponse: 包含综合分析和反馈
        """
        context = context or {}
        
        # 构建提示词
        prompt = self._build_prompt(question, context)
        
        # 调用LLM
        try:
            llm_response = self._call_llm(prompt)
            response = self._parse_response(llm_response)
        except Exception as e:
            response = AgentResponse(
                content="分析过程出现错误",
                confidence=0.3,
                reasoning=f"执行出错: {str(e)}",
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
            question: 原始问题
            context: 上下文信息
            
        Returns:
            格式化后的提示词
        """
        context = context or {}
        
        # Student Agent信息
        student_response = context.get("student_response", "暂无")
        student_confidence = context.get("student_confidence", 0.0)
        current_round = context.get("round_index", 1)
        
        # 格式化Agent输出
        agent_outputs = self._format_agent_outputs(context.get("agent_outputs", []))
        
        template = self.config.prompt_template or self.PROMPT_TEMPLATE
        
        return template.format(
            question=question,
            student_response=student_response,
            student_confidence=student_confidence,
            current_round=current_round,
            agent_outputs=agent_outputs
        )
    
    def _format_agent_outputs(self, outputs: List[AgentOutput]) -> str:
        """格式化Agent输出列表"""
        if not outputs:
            return "暂无其他Agent的反馈"
        
        formatted = []
        for i, output in enumerate(outputs, 1):
            agent_type = output.agent_type.value if hasattr(output.agent_type, 'value') else str(output.agent_type)
            formatted.append(
                f"### Agent {i}: {agent_type} ({output.agent_id})\n"
                f"**回答**: {output.response}\n"
                f"**置信度**: {output.confidence}\n"
                f"**推理过程**: {output.reasoning}"
            )
        
        return "\n\n".join(formatted)
    
    def _parse_response(self, llm_response: str) -> AgentResponse:
        """
        解析LLM响应
        
        Args:
            llm_response: LLM原始响应
            
        Returns:
            解析后的AgentResponse
        """
        try:
            # 提取JSON
            json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = llm_response
            
            data = json.loads(json_str)
            
            # 构建反馈内容
            feedback_content = data.get("feedback_prompt", "")
            
            # 提取答案评估
            answer_evaluation = data.get("answer_evaluation", {})
            quality_score = answer_evaluation.get("quality_score", 0.5)
            correctness = answer_evaluation.get("correctness", "uncertain")
            
            return AgentResponse(
                content=feedback_content,
                confidence=0.8,  # Insight Agent的输出置信度
                reasoning=data.get("reasoning", ""),
                metadata={
                    # 答案评估（新增）
                    "answer_evaluation": answer_evaluation,
                    "quality_score": quality_score,
                    "correctness": correctness,
                    # 原有字段
                    "consensus_points": data.get("consensus_points", []),
                    "divergence_points": data.get("divergence_points", []),
                    "conflicts_detected": data.get("conflicts_detected", []),
                    "complementary_insights": data.get("complementary_insights", []),
                    "key_evidence": data.get("key_evidence", []),
                    "thinking_angles": data.get("thinking_angles", []),
                    "recommended_action": data.get("recommended_action", "refine"),
                    "confidence_adjustment": data.get("confidence_adjustment", 0.0)
                }
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return AgentResponse(
                content=llm_response,
                confidence=0.5,
                reasoning="无法解析结构化响应",
                metadata={"parse_error": str(e), "raw_response": llm_response}
            )
    
    def synthesize(
        self,
        question: str,
        student_response: str,
        student_confidence: float,
        agent_outputs: List[AgentOutput]
    ) -> Dict[str, Any]:
        """
        便捷方法：执行综合分析并返回结构化结果
        
        Args:
            question: 原始问题
            student_response: Student Agent的回答
            student_confidence: Student Agent的置信度
            agent_outputs: 各Agent的输出
            
        Returns:
            包含完整分析结果的字典
        """
        context = {
            "student_response": student_response,
            "student_confidence": student_confidence,
            "agent_outputs": agent_outputs
        }
        
        response = self.execute(question, context)
        
        return {
            "feedback_prompt": response.content,
            # 答案评估（新增）
            "answer_evaluation": response.metadata.get("answer_evaluation", {}),
            "quality_score": response.metadata.get("quality_score", 0.5),
            "correctness": response.metadata.get("correctness", "uncertain"),
            # 原有字段
            "conflicts": response.metadata.get("conflicts_detected", []),
            "complementary_points": response.metadata.get("complementary_insights", []),
            "thinking_angles": response.metadata.get("thinking_angles", []),
            "recommended_action": response.metadata.get("recommended_action", "refine"),
            "confidence_adjustment": response.metadata.get("confidence_adjustment", 0.0),
            "reasoning": response.reasoning
        }

