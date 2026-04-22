# 实验

### 数据集：B1、B2、B3

### （B1）

|                         数据集名称                         |    任务类型    |                        数据规模                         |                          主要功能                          |                        获取方式                        |
| :--------------------------------------------------------: | :------------: | :-----------------------------------------------------: | :--------------------------------------------------------: | :----------------------------------------------------: |
|             **TruthfulQA** (Lin et al., 2022)              |   事实性问答   |                        817 问题                         | 测试模型避免生成“听起来合理但错误”的回答（典型幻觉测试集） |              Hugging Face: `truthful_qa`               |
|    **MMLU (Massive Multitask Language Understanding)**     | 多领域知识问答 |                      15k+，57 科目                      |  通用认知与专业知识评估，常用于衡量“general intelligence”  |               Hugging Face: `cais/mmlu`                |
|               **GSM8K** (Cobbe et al., 2021)               |    数学推理    |                        8.5k 问题                        |                 评估多步数学和逻辑推理能力                 |                 Hugging Face: `gsm8k`                  |
| **HaluEval 2.0** (Cheng et al., 2024, 来自 HaluAgent 论文) |    幻觉检测    |                     多领域汉英双语                      |                 用于评估幻觉识别/检测能力                  |         https://github.com/RUCAIBox/HaluAgent          |
|              **OlympicArena (NeurIPS 2024)**               |  综合认知推理  | 11,163 跨学科问题（数学、物理、语言、化学、生物、人文） |        测试 agent 在多学科和多模态推理任务中的能力         | 官方开源平台：https://gair-nlp.github.io/OlympicArena/ |

### 对比工具：

#### Reflexion（ICLR 2026 under review）

**论文**：REFLEXION: Language Models That Think Twice for Internalized Self-Correction
**模型**：LLaMA‑3‑8B‑Instruct（开源）

**所用数据集**：

#### （B2）

|              数据集               |      类型      |               用途               |     来源     | 结果     |
| :-------------------------------: | :------------: | :------------------------------: | :----------: | -------- |
| **TruthfulQA (Lin et al., 2022)** | 事实一致性评估 |     测试幻觉识别与事实正确性     | Hugging Face | **61.0** |
|  **GSM8K (Cobbe et al., 2021)**   |    数学推理    |   检验逻辑推理和多步计算正确性   | Hugging Face | **78.2** |
|   **OpenBookQA, ARC‑Challenge**   |    常识推理    | 对比 reasoning step 的自反思效果 | Hugging Face | **72.8** |

#### MCF（Multi-agent Collaborative Filtering, *Expert Systems With Applications*, 2025）

**论文**：Mitigating reasoning hallucination through Multi-agent Collaborative Filtering

**模型**：ChatGPT‑3.5 与 LLaMA‑2‑13B‑Chat（开源）

**所用数据集**：

#### （B3）

|        数据集名称        |   类型   |         用途         |     来源     | **LLaMA‑2‑13B‑Chat** | **ChatGPT‑3.5** |
| :----------------------: | :------: | :------------------: | :----------: | -------------------- | --------------- |
|        **GSM8K**         | 算术推理 | 验证跨步骤逻辑正确性 | Hugging Face | **79.6**(+8.8)****   | **82.3 (+4.0)** |
| **CSQA (CommonsenseQA)** | 常识推理 |  检查多步思维一致性  | Hugging Face | **75.1**(+6.6)****   | **83.5 (+3.4)** |
|    **ARC‑Challenge**     | 科学推理 | 检查 reasoning depth | Hugging Face | **72.8**(+7.5)****   | **78.6 (+4.0)** |

### 实验设置

1. **主实验**（**`RQ1，RQ2，RQ3，RQ4`**）**：**\sys使用、GPT-4系列、**GPT-4mini系列、LLaMA‑3模型（开源）**，在B1上进行实验。统计B1的准确率、提升准确率、花费的时间、token

2. **对比工作（RQ2）：**\sys在B2、B3（和B1有重复的）进行实验，准确率、提升准确率、花费的时间、token。（其他的对比工作使用论文中的结果。）

3. **消融实验（RQ4）：**

   1. Ablation-1：去掉memory系统
   2. Ablation-2：去掉多agent系统

   统计识别+最终结果的成功率。

4. **真实系统运用（RQ5）：**插入到EvoPoC提升效果。

### RQ1：

***How effective is it on the general dataset, and to what extent does the capability of the underlying LLM influence \sys’s overall effectiveness?***

**实验目的：**

验证本方案\sys在通用数据集上的有效性，并比较不同的LLM底座对于结果的影响。

**实验细节：**

（待补充）

**实验结果展示：**

- \sys使用多种LLM在B1数据集上的实验结果。实验结果为：使用系统前后的准确率提高

### RQ2：

***How does \sys perform compared to state-of-the-art approaches?***

**实验目的**：

对比本方案\sys与对比工作在上的效果表现。

**实验细节**：

（待补充）

**实验结果展示**：

- \sys使用**LLaMA‑3‑8B‑Instruct**在B2上的结果和**Reflexion**的对比。
- \sys使用**ChatGPT‑3.5 与 LLaMA‑2‑13B‑Chat（开源）**在B3上的结果和**MCF**的对比。

### RQ3：

***How efficient is \sys in terms of runtime performance and token consumption?***

**实验目的**：

评估本方案\sys在时间和token方面的开销。

**实验细节**：

（待补充）

**实验结果展示**：

- \sys在B1数据集上测试时，使用的时间、token消耗。

### RQ4：

***How dose \sys's key design affect its overall performance?***

**实验目的**：

消融实验验证关键组件的作用。

**实验细节**：

（待补充）

##### **实验结果展示**：

- 在通用数据集上的识别率和最终成功率与\sys进行比较。使用最好效果的LLM基座

### RQ5：

***How dose \sys's key design affect its overall performance?***

**实验目的**：

真实系统中\sys插件的应用。

**实验细节**：

（待补充）

##### **实验结果展示**：

- 对于EovPoC系统的结果提升。

# TimeLine