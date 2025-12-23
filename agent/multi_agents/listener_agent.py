"""
Listener Agent (倾听者Agent)

职责：
- 综合各方Agent的意见
- 提取共识和分歧点
- 平衡不同观点，形成综合建议
"""

import json
import re
from typing import Any, Dict, List, Optional

from agent.base_agent import BaseAgent, AgentResponse
from core.schemas import AgentConfig, AgentType


class ListenerAgent(BaseAgent):
    """
    倾听者Agent
    
    核心功能：
    1. 汇总多个Agent的反馈
    2. 识别共识和分歧
    3. 提出平衡各方的综合建议
    
    应用场景：
    - 多Agent协作后的意见整合
    - 解决Agent间的冲突
    - 形成统一的改进方向
    """
    
    PROMPT_TEMPLATE = """
## 角色定义
你是一位善于倾听的协调者（Listener Agent），擅长综合不同意见。
你的职责是认真倾听每位专家的观点，找出共识与分歧，提出平衡各方的综合建议。

## 核心能力
- 全面倾听：公正地理解每个Agent的观点
- 提炼共识：识别各方意见中的共同点
- 分析分歧：理清分歧的本质和原因
- 综合协调：提出能被各方接受的解决方案

## 任务描述
请综合分析以下各Agent的反馈意见：
1. 识别各Agent观点的共同点
2. 分析存在分歧的地方及原因
3. 评估每个观点的价值和合理性
4. 提出综合性的改进建议

## 输入信息

### 原始问题
{question}

### Student Agent的回答
{student_response}
置信度: {student_confidence}

### 各专家Agent的反馈
{expert_feedbacks}

### 上下文信息（如有）
{context_info}

## 输出要求

请按照以下JSON格式输出你的综合分析：

```json
{{
    "consensus_analysis": {{
        "agreed_points": ["各方一致认同的观点1", "观点2"],
        "agreement_strength": "strong/moderate/weak",
        "core_consensus": "核心共识总结"
    }},
    "divergence_analysis": {{
        "disagreement_points": [
            {{
                "topic": "分歧主题",
                "positions": ["Agent A认为...", "Agent B认为..."],
                "root_cause": "分歧的根本原因"
            }}
        ],
        "irreconcilable_differences": ["难以调和的根本分歧"],
        "reconcilable_differences": ["可以调和的分歧"]
    }},
    "opinion_evaluation": {{
        "most_valuable_insights": ["最有价值的见解1", "见解2"],
        "questionable_opinions": ["需要谨慎对待的观点"],
        "weight_distribution": {{
            "agent_name": "该Agent观点的权重评估和理由"
        }}
    }},
    "synthesis": {{
        "integrated_answer": "综合各方意见后的改进答案",
        "key_improvements": ["关键改进点1", "改进点2"],
        "remaining_uncertainties": ["仍存在的不确定性"],
        "recommended_focus": "建议的重点关注方向"
    }},
    "action_plan": {{
        "immediate_actions": ["立即需要做的修改"],
        "optional_enhancements": ["可选的优化"],
        "further_investigation": ["需要进一步研究的问题"]
    }},
    "final_response": "作为倾听者，你的综合建议总结",
    "confidence": 0.XX,
    "reasoning": "综合分析的推理过程"
}}
```

## 倾听原则
1. **公平公正**：不偏袒任何一方，客观评估每个观点
2. **求同存异**：尊重分歧的存在，但优先寻找共识
3. **实用导向**：综合建议应该是可操作的
4. **价值判断**：敢于指出哪些意见更有价值

## 协调优先级
- 首先：解决影响正确性的核心问题
- 其次：处理影响完整性的遗漏
- 最后：考虑表述和细节优化

## 客观综合要求（关键！）
作为倾听者，你不是来"和稀泥"的，而是来客观整合的：

1. **不要过早达成共识**
   - 如果专家们有分歧，不要试图抹平分歧
   - 分歧本身可能揭示问题的复杂性
   - divergence_analysis.disagreement_points 必须真实反映分歧

2. **首轮综合特别要求**
   - 首轮 remaining_uncertainties 不能为空
   - 首轮 action_plan.immediate_actions 应该具体且可操作
   - 不要过早下"已经很好"的结论

3. **权重分配要有依据**
   - weight_distribution 必须给出理由
   - 不是所有专家意见权重相同
   - 某些领域的专家意见应该更受重视

4. **综合不等于妥协**
   - integrated_answer 应该是"取长补短"而非"折中"
   - 指出各方意见的强项和弱项
   - synthesis.key_improvements 至少列出2-3项

5. **你的核心价值**
   - 帮助 Student 理解：哪些反馈最重要
   - 帮助 Student 理解：下一步应该优先改什么
   - 避免给 Student 造成"大家都说好"的错觉

请开始你的综合分析：
"""
    
    def __init__(
        self,
        agent_id: str = "listener_agent_001",
        config: Optional[AgentConfig] = None,
        llm_client: Any = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.LISTENER,
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
                content=f"综合分析过程出现错误: {str(e)}",
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
        
        # 格式化专家反馈
        expert_feedbacks = context.get("expert_feedbacks", [])
        if expert_feedbacks:
            feedbacks_text = "\n\n".join([
                f"【{fb.get('agent_name', '未知Agent')}】\n"
                f"观点: {fb.get('content', '无')}\n"
                f"置信度: {fb.get('confidence', 0)}"
                for fb in expert_feedbacks
            ])
        else:
            feedbacks_text = "暂无专家反馈"
        
        template = self.config.prompt_template or self.PROMPT_TEMPLATE
        
        return template.format(
            question=question,
            student_response=student_response,
            student_confidence=student_confidence,
            expert_feedbacks=feedbacks_text,
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
            
            synthesis = data.get("synthesis", {})
            
            return AgentResponse(
                content=data.get("final_response", ""),
                confidence=float(data.get("confidence", 0.7)),
                reasoning=data.get("reasoning", ""),
                metadata={
                    "consensus_analysis": data.get("consensus_analysis", {}),
                    "divergence_analysis": data.get("divergence_analysis", {}),
                    "opinion_evaluation": data.get("opinion_evaluation", {}),
                    "synthesis": synthesis,
                    "action_plan": data.get("action_plan", {}),
                    "integrated_answer": synthesis.get("integrated_answer", "")
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
        expert_feedbacks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        便捷方法：综合多个Agent的反馈
        
        Args:
            question: 原始问题
            student_response: Student Agent的回答
            student_confidence: Student Agent的置信度
            expert_feedbacks: 专家反馈列表，每个元素包含 agent_name, content, confidence
            
        Returns:
            综合分析结果
        """
        context = {
            "student_response": student_response,
            "student_confidence": student_confidence,
            "expert_feedbacks": expert_feedbacks
        }
        
        response = self.execute(question, context)
        
        return {
            "synthesis": response.content,
            "confidence": response.confidence,
            "integrated_answer": response.metadata.get("integrated_answer", ""),
            "consensus": response.metadata.get("consensus_analysis", {}),
            "divergence": response.metadata.get("divergence_analysis", {}),
            "action_plan": response.metadata.get("action_plan", {})
        }

