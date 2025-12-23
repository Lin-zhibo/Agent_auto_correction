"""
Explanation Generator Agent (解释生成者Agent)

职责：
- 生成易于理解的解释说明
- 将复杂概念转化为通俗易懂的表述
- 提供多层次的解释（简单/中等/详细）
"""

import json
import re
from typing import Any, Dict, Optional

from agent.base_agent import BaseAgent, AgentResponse
from core.schemas import AgentConfig, AgentType


class ExplanationGeneratorAgent(BaseAgent):
    """
    解释生成者Agent
    
    核心功能：
    1. 将复杂答案转化为易懂的解释
    2. 提供多层次深度的解释版本
    3. 使用生动的例子和比喻
    
    应用场景：
    - 复杂概念的通俗化
    - 教学场景的解释生成
    - 不同受众的内容适配
    """
    
    PROMPT_TEMPLATE = """
## 角色定义
你是一位优秀的解释者（Explanation Generator Agent），擅长将复杂内容转化为易懂的解释。
你的职责是让任何人都能理解答案，不论其专业背景。

## 核心能力
- 简化复杂性：将专业术语转化为日常语言
- 层次化解释：根据不同深度需求提供解释
- 生动举例：用贴近生活的例子辅助理解
- 结构清晰：组织有条理的解释逻辑

## 解释技巧
1. **费曼技巧**：用最简单的语言解释
2. **类比映射**：用熟悉事物解释陌生概念
3. **递进深入**：从浅入深，逐步展开
4. **可视化描述**：用图像化语言辅助理解

## 任务描述
请为以下问题和答案生成易懂的解释：
1. 提供多层次的解释版本
2. 使用生动的例子和类比
3. 确保非专业人士也能理解
4. 保持解释的准确性

## 输入信息

### 问题
{question}

### Student Agent的当前回答
{student_response}
置信度: {student_confidence}

### 目标受众（如有）
{target_audience}

### 上下文信息（如有）
{context_info}

## 输出要求

请按照以下JSON格式输出你的解释：

```json
{{
    "eli5_explanation": {{
        "simple_answer": "五岁小孩也能懂的超简单解释",
        "analogy": "生活中的类比",
        "key_takeaway": "最重要的一句话总结"
    }},
    "intermediate_explanation": {{
        "main_explanation": "中等深度的解释，适合一般成年人",
        "key_concepts": ["需要理解的关键概念1", "概念2"],
        "examples": ["具体例子1", "例子2"],
        "common_misconceptions": ["常见误解及澄清"]
    }},
    "detailed_explanation": {{
        "comprehensive_explanation": "详细的完整解释",
        "technical_details": "技术细节（如适用）",
        "background_context": "背景知识",
        "further_reading": ["深入学习的方向"]
    }},
    "visual_aids": {{
        "mental_model": "帮助理解的思维模型描述",
        "diagram_suggestion": "建议的图示说明",
        "step_by_step": ["分步骤理解: 步骤1", "步骤2", "步骤3"]
    }},
    "engagement_elements": {{
        "interesting_facts": ["有趣的相关事实"],
        "real_world_applications": ["实际应用场景"],
        "thought_experiment": "帮助理解的思想实验"
    }},
    "final_response": "综合性的清晰解释",
    "confidence": 0.XX,
    "reasoning": "选择这种解释方式的理由"
}}
```

## 解释原则
1. **准确性优先**：简化但不能歪曲
2. **受众导向**：根据受众调整复杂度
3. **具体胜于抽象**：多用具体例子
4. **检验理解**：解释应能让读者自行验证

## 好解释的标准
- 读完后能用自己的话复述
- 能回答"这有什么用"的问题
- 能将概念应用到新情境
- 能识别相关的错误理解

## 解释质量审查（关键！）
作为解释生成者，你不仅要生成解释，还要评估原答案的解释质量：

1. **指出解释不清的地方**
   - 原答案哪些部分对普通人来说难以理解？
   - common_misconceptions 必须指出可能造成的误解
   - 如果答案有术语，是否有解释？

2. **首轮评估特别要求**
   - 首轮必须指出 eli5_explanation 和原答案的差距
   - 首轮 visual_aids.step_by_step 必须提供分步理解
   - 首轮 engagement_elements 应该增加吸引力

3. **多层次覆盖**
   - 三个层次的解释不能太相似
   - eli5 应该是真正简单的，不是"稍微简化"
   - detailed 应该有深度，不是"稍微展开"

4. **实际例子必须有**
   - examples 不能为空
   - 好的例子比抽象解释更有力
   - real_world_applications 帮助理解"为什么重要"

5. **你的核心价值**
   - 帮助 Student 改进答案的可理解性
   - 指出哪些地方"只有专家才懂"
   - 提供让答案更易懂的具体建议

6. **避免假设读者背景**
   - 不要假设读者知道某些术语
   - 不要假设读者有某些领域知识
   - 站在"完全外行"的角度审视

请开始生成你的解释：
"""
    
    def __init__(
        self,
        agent_id: str = "explanation_generator_agent_001",
        config: Optional[AgentConfig] = None,
        llm_client: Any = None
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.EXPLANATION_GENERATOR,
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
                content=f"解释生成过程出现错误: {str(e)}",
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
        target_audience = context.get("target_audience", "一般成年人")
        context_info = context.get("additional_context", "暂无额外上下文")
        
        template = self.config.prompt_template or self.PROMPT_TEMPLATE
        
        return template.format(
            question=question,
            student_response=student_response,
            student_confidence=student_confidence,
            target_audience=target_audience,
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
                    "eli5_explanation": data.get("eli5_explanation", {}),
                    "intermediate_explanation": data.get("intermediate_explanation", {}),
                    "detailed_explanation": data.get("detailed_explanation", {}),
                    "visual_aids": data.get("visual_aids", {}),
                    "engagement_elements": data.get("engagement_elements", {})
                }
            )
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            return AgentResponse(
                content=llm_response,
                confidence=0.5,
                reasoning="无法解析结构化响应",
                metadata={"parse_error": str(e), "raw_response": llm_response}
            )
    
    def explain(
        self,
        question: str,
        answer: str,
        level: str = "intermediate",
        target_audience: str = "一般成年人"
    ) -> str:
        """
        便捷方法：生成指定层次的解释
        
        Args:
            question: 问题
            answer: 需要解释的答案
            level: 解释层次 (eli5/intermediate/detailed)
            target_audience: 目标受众
            
        Returns:
            对应层次的解释文本
        """
        context = {
            "student_response": answer,
            "student_confidence": 0.8,
            "target_audience": target_audience
        }
        
        response = self.execute(question, context)
        
        level_map = {
            "eli5": "eli5_explanation",
            "intermediate": "intermediate_explanation",
            "detailed": "detailed_explanation"
        }
        
        explanation_key = level_map.get(level, "intermediate_explanation")
        explanation_data = response.metadata.get(explanation_key, {})
        
        if level == "eli5":
            return explanation_data.get("simple_answer", response.content)
        elif level == "intermediate":
            return explanation_data.get("main_explanation", response.content)
        else:
            return explanation_data.get("comprehensive_explanation", response.content)

