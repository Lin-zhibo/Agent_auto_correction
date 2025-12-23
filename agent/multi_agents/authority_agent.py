"""
Authority Agent (权威者Agent)

职责：
- 从权威、专业的角度审视问题
- 提供基于专业知识的判断
- 对Student Agent的答案进行专业评估
"""

import json
import re
from typing import Any, Dict, Optional

from agent.base_agent import BaseAgent, AgentResponse
from core.schemas import AgentConfig, AgentType


class AuthorityAgent(BaseAgent):
    """
    权威者Agent
    
    核心功能：
    1. 从专业领域角度分析问题
    2. 基于权威知识进行判断
    3. 提供专业级别的建议
    
    应用场景：
    - 需要专业知识支撑的问题
    - 需要权威背书的答案验证
    - 专业术语和概念的解释
    
    这是多Agent阶段的第一个Agent实现，作为示例展示完整的提示词设计。
    """
    
    # 提示词模板 - 具有实际可落地性的完整设计
    PROMPT_TEMPLATE = """
## 角色定义
你是一位资深的领域权威专家（Authority Agent），拥有深厚的专业知识背景。
你的职责是从专业角度审视问题和答案，提供权威性的判断和建议。

## 专业背景
- 你具备多个学科领域的专业知识：数学、物理、计算机科学、工程学、经济学、哲学等
- 你熟悉各领域的核心概念、理论框架和最佳实践
- 你能够识别常见的概念混淆和逻辑谬误
- 你了解领域内的权威来源和标准答案

## 任务描述
请从专业权威的角度，对以下问题和Student Agent的回答进行评估：
1. 评估答案的专业准确性
2. 识别可能的概念错误或不精确之处
3. 提供权威性的补充说明
4. 给出专业角度的改进建议

## 输入信息

### 问题
{question}

### Student Agent的当前回答
{student_response}
置信度: {student_confidence}

### 上下文信息（如有）
{context_info}

## 输出要求

请按照以下JSON格式输出你的专业评估：

```json
{{
    "professional_assessment": {{
        "accuracy_score": 0.XX,
        "accuracy_analysis": "对答案准确性的专业评估",
        "identified_domain": "识别出的问题所属领域",
        "key_concepts": ["涉及的核心专业概念1", "概念2"]
    }},
    "expert_opinion": {{
        "stance": "agree/partially_agree/disagree",
        "reasoning": "专业角度的判断理由",
        "authoritative_sources": ["支持判断的权威来源或理论"],
        "professional_supplement": "专业补充说明，可能是学生答案中遗漏的重要内容"
    }},
    "error_analysis": {{
        "concept_errors": ["发现的概念错误1", "错误2"],
        "precision_issues": ["不够精确的表述1", "表述2"],
        "logical_gaps": ["逻辑漏洞1", "漏洞2"]
    }},
    "recommendations": {{
        "must_correct": ["必须修正的问题"],
        "should_improve": ["建议改进的方面"],
        "optional_enhancement": ["可选的优化建议"]
    }},
    "final_response": "作为权威专家，你对这个问题的简洁观点",
    "confidence": 0.XX,
    "reasoning": "得出上述评估的推理过程"
}}
```

## 评估标准

### 准确性评分标准 (accuracy_score)
- 0.9-1.0: 专业准确，无明显错误
- 0.7-0.9: 基本正确，有小瑕疵
- 0.5-0.7: 部分正确，存在明显问题
- 0.3-0.5: 错误较多，需要大幅修正
- 0-0.3: 根本性错误，需要重写

### 立场说明 (stance)
- agree: 完全认同Student Agent的答案
- partially_agree: 部分认同，但需要补充或修正
- disagree: 不认同，存在重要错误

## 注意事项
1. 保持专业客观，避免过于苛刻或宽松
2. 明确区分"错误"和"不够完美"
3. 建议应具体可操作，避免空泛
4. 如果问题超出专业范围，诚实说明

## 评分校准（重要！）
作为权威专家，你的评估应该**严格且有建设性**：

1. **accuracy_score 评分原则**：
   - 首次看到的答案，很少能达到 0.9 以上
   - 大多数"看起来正确"的答案应该在 0.6-0.8 之间
   - 即使答案正确，如果不够完整/精确，也应扣分

2. **stance 选择原则**：
   - **agree**：极少使用，仅当答案专业、完整、无可挑剔时
   - **partially_agree**：这是最常见的情况，意味着"方向对但可以更好"
   - **disagree**：存在明显的专业性错误

3. **必须指出改进空间**：
   - 即使答案基本正确，也要指出可以更专业、更严谨的地方
   - recommendations 的 should_improve 至少包含1-2项

## 首轮评估特别要求（关键！）
如果这是对Student首次回答的评估，请特别注意：

1. **首轮 accuracy_score 上限为 0.80**
   - 再好的首次回答也有优化空间
   - 给 Student 留出改进的机会
   
2. **首轮 stance 不应为 agree**
   - 应该选择 partially_agree 或 disagree
   - 即使答案看起来完美，也要找到可以更专业的地方

3. **首轮必须提供实质性反馈**
   - error_analysis 的三个列表不能都为空
   - recommendations.must_correct 或 should_improve 必须有内容

4. **你的价值体现**
   - 作为权威专家，你的价值不在于简单认可
   - 而在于提供深度专业见解，帮助答案达到更高水准
   - 宁可"严格但建设性"，不可"宽松但无价值"

请开始你的专业评估：
"""
    
    def __init__(
        self,
        agent_id: str = "authority_agent_001",
        config: Optional[AgentConfig] = None,
        llm_client: Any = None
    ):
        """
        初始化Authority Agent
        
        Args:
            agent_id: Agent唯一标识
            config: Agent配置
            llm_client: LLM客户端
        """
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.AUTHORITY,
            config=config,
            llm_client=llm_client
        )
    
    def execute(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        执行专业评估
        
        Args:
            question: 输入问题
            context: 上下文信息，应包含：
                - student_response: Student Agent的回答
                - student_confidence: Student Agent的置信度
                - additional_context: 额外上下文
                
        Returns:
            AgentResponse: 包含专业评估结果
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
                content=f"专业评估过程出现错误: {str(e)}",
                confidence=0.1,
                reasoning="执行出错",
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
            
            # 提取核心内容
            final_response = data.get("final_response", "")
            expert_opinion = data.get("expert_opinion", {})
            
            return AgentResponse(
                content=final_response,
                confidence=float(data.get("confidence", 0.7)),
                reasoning=data.get("reasoning", ""),
                metadata={
                    "professional_assessment": data.get("professional_assessment", {}),
                    "expert_opinion": expert_opinion,
                    "error_analysis": data.get("error_analysis", {}),
                    "recommendations": data.get("recommendations", {}),
                    "stance": expert_opinion.get("stance", "partially_agree")
                }
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return AgentResponse(
                content=llm_response,
                confidence=0.5,
                reasoning="无法解析结构化响应",
                metadata={"parse_error": str(e), "raw_response": llm_response}
            )
    
    def evaluate(
        self,
        question: str,
        student_response: str,
        student_confidence: float,
        additional_context: str = ""
    ) -> Dict[str, Any]:
        """
        便捷方法：执行专业评估并返回结构化结果
        
        Args:
            question: 问题
            student_response: Student Agent的回答
            student_confidence: Student Agent的置信度
            additional_context: 额外上下文
            
        Returns:
            包含评估结果的字典
        """
        context = {
            "student_response": student_response,
            "student_confidence": student_confidence,
            "additional_context": additional_context
        }
        
        response = self.execute(question, context)
        
        return {
            "opinion": response.content,
            "confidence": response.confidence,
            "stance": response.metadata.get("stance", "partially_agree"),
            "accuracy_score": response.metadata.get("professional_assessment", {}).get("accuracy_score", 0.5),
            "errors": response.metadata.get("error_analysis", {}).get("concept_errors", []),
            "recommendations": response.metadata.get("recommendations", {}),
            "reasoning": response.reasoning
        }

