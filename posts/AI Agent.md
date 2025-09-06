---
title: AI Agent
description: AI Agent
date: 2025-09-06
tags:
  - Agent
---

https://www.youtube.com/watch?v=M2Yg1kwPpts

* 今天使用AI的方式：人类给予明确的口令，AI根据这一个口令做一个动作
* AI Agent：人类给予目标，AI自动想办法完成

### 如何打造AI Agent？
#### RL
<img src="/public/agent1.png">

* 缺点：每一个任务都需要一个rl模型去训练
#### LLM
<img src="/public/agent2.png">

#### 从LLM的角度看Agent要解决的问题

<img src="/public/agent3.png">

#### 以LLM运行Agent的优势
* 原来以Alphago为例，它局限于事先设定好的有限的行为，只能在棋盘上落子；使用LLM，它有近乎无限的可能，而且可以使用外部工具
* 例如都是一个AI programmer出现了Compile error,如果是Typical Agent的话，就会给Agent一个Reward=-1，但是为什么是-1呢？  如果是LLM Agent我们可以直接把错误的log发给agent，它获得更多资讯，就会给出更好的结果
#### AI Agent举例
* Minecraft中的AI NPC
* 让AI使用电脑
* 用AI做科学研究
#### 对于更加真实的互动情景
* 回合制互动到即时互动，外部环境变了，就立即采取新的行动

### AI Agent关键能力
#### AI 如何根据经验调整行为
* 很多语言模型可以根据回馈来改变行为，不用调整或更新参数
<img src="/public//agent4.png">

* Read模块其实相当于一个RAG模块，只不过检索的是自己的记忆
* StreamBench 正面的反馈比负面的反馈更有用，也就是说你要告诉ai要去做什么，而不是不要做什么
<img src="/public/agent5.png">

* Write就是决定目前的对话要不要存入Memory里
* Reflection就是对记忆做出更高层次的总结和抽象，可以形成知识图谱

##### 有记忆的GPT
#### AI如何使用工具
* 工具可以看做Function，使用工具就是调用这些Function，使用工具又叫Function Call
* 模型不必在意工具内部是怎么样运作的，只需要知道给它什么样的输入，可以得到什么样的输出
##### 如何使用工具？
<img src="/public/agent6.png">

* 最常使用的就是搜索引擎
* 可以使用其他AI作为工具，例如一个只能识别文本的模型，可以使用语音识别的模型来得到文字，或者用户的情绪分析等
##### 非常多工具怎么办？
* 做一个Tool Selection 模块来选择工具，其实跟RAG很像
<img src="/public/agent7.png">

* 而且模型可以自己写一个function当做工具自己来用
##### 工具也会出错？
* 例如调用温度的function,如果得到100度，他会说这个温度不合理
* 内部的knowledge和外部的knowledge在做竞争
##### 什么样的外部资讯比较容易说服AI？
* 跟自己内部知识比较相近的
* 相比于人类，更相信AI同类
##### 就算工具可靠，不代表AI就不会犯错
#### AI能不能做计划
* 计划赶不上变化

PlanBench

* LLM做计划？会不会是从资料里拿出来的？
* 创造一个新规则体系来测试

##### 方法
* Tree Search for Language Model Agents
* 做出的动作覆水难收，那就把制作计划当做“梦境”，找出一个成功的solution再做出行动