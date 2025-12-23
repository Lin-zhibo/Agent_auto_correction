"""
Concept Analogist Agent (概念类比者Agent)

职责：
- 通过类比帮助理解复杂概念
- 在不同领域间建立概念映射
- 发现深层的结构相似性
"""

import json
import re
from typing import Any, Dict, List, Optional

from agent.base_agent import BaseAgent, AgentResponse
from core.schemas import AgentConfig, AgentType


class ConceptAnalogistAgent(BaseAgent):
    """
    概念类比者Agent
    
    核心功能：
    1. 为复杂概念寻找恰当的类比
    2. 建立跨领域的概念映射
    3. 通过结构对应加深理解
    
    应用场景：
    - 抽象概念的具象化
    - 跨学科知识的迁移
    - 深化对概念本质的理解
    """
    
    PROMPT_TEMPLATE = """
## 角色定义
你是一位概念类比专家（Concept Analogist Agent），擅长在不同事物间发现深层联系。
你的职责是通过类比帮助理解复杂概念，建立知识间的桥梁。

## 核心能力
- 结构映射：识别不同领域间的结构相似性
- 类比生成：创造恰当且有洞察力的类比
- 概念迁移：将熟悉领域的知识迁移到新领域
- 局限识别：明确类比的边界和局限

## 类比的艺术
1. **结构类比**：映射关系和结构，而非表面特征
2. **渐进类比**：从简单类比递进到复杂类比
3. **多元类比**：提供多个角度的类比选择
4. **负面类比**：说明什么不是好的类比

## 任务描述
请为以下问题和答案中的概念生成有洞察力的类比：
1. 识别核心概念和难点
2. 生成多个不同角度的类比
3. 详细解释类比的映射关系
4. 指出类比的局限性

## 输入信息

### 问题
{question}

### Student Agent的当前回答
{student_response}
置信度: {student_confidence}

### 需要类比的重点概念（如有）
{focus_concepts}

### 上下文信息（如有）
{context_info}

## 输出要求

请按照以下JSON格式输出你的类比分析：

```json
{{
    "concept_identification": {{
        "core_concepts": ["需要类比的核心概念1", "概念2"],
        "difficult_points": ["理解难点1", "难点2"],
        "abstract_elements": ["抽象元素1", "元素2"]
    }},
    "primary_analogy": {{
        "source_domain": "类比来源领域",
        "target_domain": "目标概念领域",
        "analogy_statement": "完整的类比表述",
        "mapping": {{
            "元素A": "对应的类比元素A'",
            "元素B": "对应的类比元素B'",
            "关系R": "对应的类比关系R'"
        }},
        "why_it_works": "为什么这个类比有效",
        "insight_gained": "通过类比获得的洞察"
    }},
    "alternative_analogies": [
        {{
            "analogy": "备选类比1",
            "source_domain": "来源领域",
            "strength": "这个类比的优势",
            "best_for": "最适合解释的方面"
        }},
        {{
            "analogy": "备选类比2",
            "source_domain": "来源领域",
            "strength": "这个类比的优势",
            "best_for": "最适合解释的方面"
        }}
    ],
    "analogy_limitations": {{
        "breaks_down_when": ["类比失效的情况1", "情况2"],
        "misleading_aspects": ["可能造成误导的方面"],
        "important_differences": ["源领域和目标领域的重要差异"]
    }},
    "progressive_understanding": {{
        "level_1": "最基础的类比理解",
        "level_2": "进阶的类比理解",
        "level_3": "深度的结构理解",
        "transcend_analogy": "超越类比，直接理解概念本质"
    }},
    "cross_domain_connections": {{
        "related_concepts": ["其他领域的相关概念"],
        "universal_patterns": ["跨领域的普遍模式"],
        "transfer_opportunities": ["知识迁移的机会"]
    }},
    "final_response": "综合性的类比解释",
    "confidence": 0.XX,
    "reasoning": "选择这些类比的理由"
}}
```

## 好类比的标准
1. **结构对应**：关系和结构有清晰映射
2. **熟悉度高**：源领域是受众熟悉的
3. **洞察力强**：能揭示目标概念的本质
4. **边界清晰**：明确说明类比的局限

## 类比来源领域参考
- 日常生活：家庭、烹饪、交通、运动
- 自然现象：水流、天气、生态系统
- 社会系统：城市、公司、学校
- 人体：器官、神经、免疫系统
- 技术系统：机器、网络、建筑

## 类比质量审查（关键！）
作为类比专家，你需要严格评估类比的有效性：

1. **指出类比的局限性**
   - 没有完美的类比，每个类比都有失效的地方
   - analogy_limitations 的三个字段都不能为空
   - breaks_down_when 必须具体说明什么情况下类比不再适用

2. **首轮评估特别要求**
   - 首轮必须提供 alternative_analogies 至少2个
   - 首轮 progressive_understanding 要展示理解的层次
   - 首轮 misleading_aspects 必须指出可能的误导

3. **挑战现有类比**
   - 如果原答案中有类比，评估其质量
   - 好的类比要结构对应，不只是表面相似
   - 指出原有类比的不足之处

4. **映射必须清晰**
   - mapping 不能只是一句话
   - 每个关键元素都要有对应
   - 关系和结构的映射比表面特征更重要

5. **你的核心价值**
   - 帮助 Student 用更好的类比来解释概念
   - 指出哪些类比是"看起来像但实际误导"的
   - 提供更准确、更有洞察力的类比选择

6. **负面类比很重要**
   - 告诉读者什么是不好的类比
   - 避免常见的错误类比
   - important_differences 帮助防止过度类比

请开始生成你的类比：
"""
    
    def __init__(
        self,
        agent_id: str = "concept_analogist_agent_001",
        config: Optional[AgentConfig] = None,
        llm_client: Any = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.CONCEPT_ANALOGIST,
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
                content=f"类比生成过程出现错误: {str(e)}",
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
        focus_concepts = context.get("focus_concepts", "自动识别")
        context_info = context.get("additional_context", "暂无额外上下文")
        
        template = self.config.prompt_template or self.PROMPT_TEMPLATE
        
        return template.format(
            question=question,
            student_response=student_response,
            student_confidence=student_confidence,
            focus_concepts=focus_concepts,
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
            
            primary = data.get("primary_analogy", {})
            
            return AgentResponse(
                content=data.get("final_response", ""),
                confidence=float(data.get("confidence", 0.7)),
                reasoning=data.get("reasoning", ""),
                metadata={
                    "concept_identification": data.get("concept_identification", {}),
                    "primary_analogy": primary,
                    "alternative_analogies": data.get("alternative_analogies", []),
                    "analogy_limitations": data.get("analogy_limitations", {}),
                    "progressive_understanding": data.get("progressive_understanding", {}),
                    "cross_domain_connections": data.get("cross_domain_connections", {}),
                    "best_analogy": primary.get("analogy_statement", "")
                }
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return AgentResponse(
                content=llm_response,
                confidence=0.5,
                reasoning="无法解析结构化响应",
                metadata={"parse_error": str(e), "raw_response": llm_response}
            )
    
    def find_analogy(
        self,
        concept: str,
        target_audience: str = "一般成年人",
        preferred_domains: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        便捷方法：为特定概念寻找类比
        
        Args:
            concept: 需要类比的概念
            target_audience: 目标受众
            preferred_domains: 偏好的类比来源领域
            
        Returns:
            包含类比结果的字典
        """
        context = {
            "student_response": f"这个概念是：{concept}",
            "student_confidence": 0.8,
            "focus_concepts": concept,
            "additional_context": f"目标受众: {target_audience}"
        }
        
        if preferred_domains:
            context["additional_context"] += f"\n偏好的类比领域: {', '.join(preferred_domains)}"
        
        response = self.execute(f"请解释概念：{concept}", context)
        
        return {
            "concept": concept,
            "primary_analogy": response.metadata.get("primary_analogy", {}),
            "alternatives": response.metadata.get("alternative_analogies", []),
            "limitations": response.metadata.get("analogy_limitations", {}),
            "explanation": response.content
        }

