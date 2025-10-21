The original repository for paper collection is [[Link](https://github.com/141forever/inductive-reasoning-papers)].

# Inductive Reasoning

Inductive reasoning involves drawing general conclusions from specific observations.
The main characteristics of inductive reasoning are its particular-to-general thinking process and the non-uniqueness of its answers. 
Considering how humans perceive the world, they typically make judgments by drawing analogies from past experiences to current situations, rather than always going through a strictly logical process as in deductive reasoning. 
We can assume that the inductive mode is key to knowledge generalization and better aligns with human cognition.
We give two examples of inductive reasoning in the figure below.

<p align="center">
  <img src="https://github.com/141forever/inductive-reasoning-papers/blob/main/Figures/two_examples.jpg" width="50%">
</p>



# About the Survey

Among all the papers shown in the repository for paper collection ([[Link](https://github.com/141forever/inductive-reasoning-papers)]), our survey only focuses on synthesizing and analyzing those related to inductive tasks and language models, while also including additional relevant works beyond this repository.
This is the link to our survey [[ARXIV](https://arxiv.org/abs/2510.10182)].

If you find any points in our survey worth discussing or notice any mistakes, feel free to open an issue and share your thoughts!


# Citation
If you think this survey helps, please cite our paper.
```
@misc{chen2025surveyinductivereasoninglarge,
      title={A Survey of Inductive Reasoning for Large Language Models}, 
      author={Kedi Chen and Dezhao Ruan and Yuhao Dan and Yaoting Wang and Siyu Yan and Xuecheng Wu and Yinqi Zhang and Qin Chen and Jie Zhou and Liang He and Biqing Qi and Linyang Li and Qipeng Guo and Xiaoming Shi and Wei Zhang},
      year={2025},
      eprint={2510.10182},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.10182}, 
}
```



# The Selected Papers

## Introduction and Background

- **Multitask Learning: A Knowledge-Based Source of Inductive Bias** [ICML1993] [[paper link](https://api.semanticscholar.org/CorpusID:18522085)]

- **Inductive Reasoning and Bounded Rationality** [The American Economic Review, 1994] [[paper link](https://www.semanticscholar.org/paper/Inductive-Reasoning-and-Bounded-Rationality-Arthur/8f1f0e79365e75c5ba5d513355b32b03d6940c97)]

- **Inductive policy: The pragmatics of bias selection** [Machine Learning, 20, 1995] [[paper link](https://link.springer.com/content/pdf/10.1007/BF00993474.pdf)] 

- **Properties ofinductive reasoning** [Psychonomic Bulletin & Review, 2000] [[paper link](https://link.springer.com/content/pdf/10.3758/BF03212996.pdf)]

- **The perception of the environment: essays on livelihood, dwelling and skill** [Book, 2000] [[paper link](https://leiaarqueologia.wordpress.com/wp-content/uploads/2017/08/the-perception-of-the-environment-tim-ingold.pdf)]

- **A General Inductive Approach for Qualitative Data Analysis** [American Journal of Evaluation, 27(2), 2003] [[paper link](https://www.researchgate.net/profile/David-Thomas-57/publication/228620846_A_General_Inductive_Approach_for_Qualitative_Data_Analysis/links/0f31753b32a98e30f9000000/A-General-Inductive-Approach-for-Qualitative-Data-Analysis.pdf?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19)]

- **Essentials of Logic** [Book, 2004] [[paper link](https://www.semanticscholar.org/paper/Essentials-of-Logic-Copi-Cohen/d90b699a2d858544908955f27a06d4fe106fc7f9)] 

- **When Is Inductive Inference Possible?** [NIPS2024] [[paper link](https://papers.nips.cc/paper_files/paper/2024/file/a8808b75b299d64a23255bc8d30fb786-Paper-Conference.pdf)]

- **Inductive reasoning in humans and large language models**  [Cognitive Systems Research, Volume 83, 2024] [[paper link](https://arxiv.org/pdf/2306.06548)]

## Enhancement

###  Post-training

#### Synthetic Data

- **LIME: Learning Inductive Bias for Primitives of Mathematical Reasoning** [ICML2021] [[paper link](http://proceedings.mlr.press/v139/wu21c/wu21c.pdf)]

- **Knowledge Base Question Answering for Space Debris Queries** [ACL2023] [[paper link](https://aclanthology.org/2023.acl-industry.47.pdf)] 

- **CESAR: Automatic Induction of Compositional Instructions for Multi-turn Dialogs** [EMNLP2023] [[paper link](https://aclanthology.org/2023.emnlp-main.717.pdf)]

- **Linguistic Rule Induction Improves Adversarial and OOD Robustness in Large Language Models** [COLING2024] [[paper link](https://aclanthology.org/2024.lrec-main.924/)]

- **ItD: Large Language Models Can Teach Themselves Induction through Deduction** [ACL2024] [[paper link](https://aclanthology.org/2024.acl-long.150.pdf)]

- **Code-Driven Inductive Synthesis: Enhancing Reasoning Abilities of Large Language Models with Sequences** [Arxiv2025] [[paper link](https://arxiv.org/abs/2503.13109)]

- **In the LLM era, Word Sense Induction remains unsolved** [ACL2025] [[paper link](https://aclanthology.org/2025.findings-acl.882.pdf)] 

#### IRL-style Optimization

- **A survey of inverse reinforcement learning: Challenges, methods and progress** [Artificial Intelligence, 2021] [[paper link](https://arxiv.org/pdf/1806.06877)]

- **Query-Dependent Prompt Evaluation and Optimization with Offline Inverse RL** [ICLR2024] [[paper link](https://openreview.net/forum?id=N6o0ZtPzTg)]

- **Solving the Inverse Alignment Problem for Efficient RLHF** [Arxiv2024] [[paper link](https://arxiv.org/pdf/2412.10529)]

- **Approximated Variational Bayesian Inverse Reinforcement Learning for Large Language Model Alignment** [AAAI2025] [[paper link](https://arxiv.org/pdf/2411.09341)]

- **Inverse Reinforcement Learning Meets Large Language Model Post-Training: Basics, Advances, and Opportunities** [Arxiv2025] [[paper link](https://arxiv.org/pdf/2507.13158)]

- **Your Reward Function for RL is Your Best PRM for Search: Unifying RL and Search-Based TTS** [Arxiv2025] [[paper link](https://arxiv.org/pdf/2508.14313)]

###  Test-time Scaling

- **From Reasoning to Learning: A Survey on Hypothesis Discovery and Rule Learning with Large Language Models** [Transactions on Machine Learning Research, 2025] [[paper link](https://openreview.net/forum?id=d7W38UzUg0)]

#### Hypothesis Selection

- **Feature selection and hypothesis selection models of induction** [Proceedings of the Annual Meeting of the Cognitive Science Society, volume 12, 1990] [[paper link](https://escholarship.org/uc/item/7253w5x5)]

- **Private hypothesis selection** [Advances in Neural Information Processing Systems, 32, 2019] [[paper link](https://arxiv.org/pdf/1905.13229)]

- **Hypothesis Search: Inductive Reasoning with Language Models** [ICLR2024] [[paper link](https://arxiv.org/pdf/2309.05660)]

- **Hypothesis Search: Inductive Reasoning with Language Models** [ICLR2024] [[paper link](https://arxiv.org/pdf/2309.05660)]

- **Generating Diverse Hypotheses for Inductive Reasoning** [NAACL2025] [[paper link](https://aclanthology.org/2025.naacl-long.429.pdf)]

- **Measuring What Matters: Evaluating Ensemble LLMs with Label Refinement in Inductive Coding**  [ACL2025] [[paper link](https://aclanthology.org/2025.findings-acl.563.pdf)]

#### Hypothesis Iteration


