---
title: Prompt Engineering
description: Prompt Engineering
date: 2025-10-12
tags:
  - 大模型
---
### How to use prompts?
#### Rule 1:Few-shot is likely bettter than zero-shot 少样本学习可能比无样本学习更好
* 在prompt中给出更多示例帮助prompt思考
* tips：
   * 使用和target相似的示例 kNN
   * 输入所有的示例然后做一些排序
* 局限：
   * 如果给出的示例的label是严重不平衡的，可能会影响结果
   * 越后来输入的示例的label对结果的影响越大
   * 在prompt中出现频率越多的token越容易影响结果

#### Rule 2:Give your model a certain role 角色扮演
#### Rule 3:Chain-of-thought for reasoning tasks 
* 也就是让大模型think ste by step
#### Rule 4:Clear and specific promopts
* 用一些特殊的符号把重要的句子标出来 ``` '''''' --- <> 等
* 明确告诉大模型需要的输出格式 例如html python等等
* 增加条件，例如如果你可以找到这个名字和位置就打印出来，找不到就返回nothing，这样是为了防止大模型一味地去满足用户的要求去胡言乱语
#### Rule 5：Know the model temperature
* Set temp=0 for determined tasks
* Set temp>0 to introduce randomness
### How to learn prompts?
* Prompt tuning
* Tuning-free prompting
* Fixed-LM prompt tuning
* Visual prompt   VPT