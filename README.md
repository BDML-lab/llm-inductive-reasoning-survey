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

- **Feature selection and hypothesis selection models of induction** [CogSci, volume 12, 1990] [[paper link](https://escholarship.org/uc/item/7253w5x5)]

- **Private hypothesis selection** [Advances in Neural Information Processing Systems, 32, 2019] [[paper link](https://arxiv.org/pdf/1905.13229)]

- **Hypothesis Search: Inductive Reasoning with Language Models** [ICLR2024] [[paper link](https://arxiv.org/pdf/2309.05660)]

- **Hypothesis Search: Inductive Reasoning with Language Models** [ICLR2024] [[paper link](https://arxiv.org/pdf/2309.05660)]

- **Generating Diverse Hypotheses for Inductive Reasoning** [NAACL2025] [[paper link](https://aclanthology.org/2025.naacl-long.429.pdf)]

- **Measuring What Matters: Evaluating Ensemble LLMs with Label Refinement in Inductive Coding**  [ACL2025] [[paper link](https://aclanthology.org/2025.findings-acl.563.pdf)]

#### Hypothesis Iteration

- **From methodology to practice: Inductive iteration in comparative research** [Comparative Political Studies, 2015] [[paper link](https://journals.sagepub.com/doi/full/10.1177/0010414014554685)]

- **Phenomenal Yet Puzzling: Testing Inductive Reasoning Capabilities of Language Models with Hypothesis Refinement** [ICLR2024] [[paper link](https://arxiv.org/pdf/2310.08559)]

- **ARISE: Iterative Rule Induction and Synthetic Data Generation for Text Classification** [NAACL2025] [[paper link](https://aclanthology.org/2025.findings-naacl.359.pdf)]

- **Patterns Over Principles: The Fragility of Inductive Reasoning in LLMs under Noisy Observations**  [ACL2025] [[paper link](https://aclanthology.org/2025.findings-acl.1006.pdf)]

- **IDEA: Enhancing the Rule Learning Ability of Large Language Model Agent through Induction, Deduction, and Abduction** [ACL2025] [[paper link](https://aclanthology.org/2025.findings-acl.698.pdf)]

#### Hypothesis Evolution

- **The hypothesis of interactive evolution** [Kybernetes, 2011] [[paper link](https://www.emerald.com/k/article-abstract/40/7-8/1021/440854/The-hypothesis-of-interactive-evolution?redirectedFrom=fulltext)]

- **Towards continuous scientific data analysis and hypothesis evolution** [AAAI2017] [[paper link](https://doi.org/10.1609/aaai.v31i1.11157)]

- **Neuro-Symbolic Hierarchical Rule Induction** [ICML2022] [[paper link](https://proceedings.mlr.press/v162/glanois22a/glanois22a.pdf)]

- **Zero-Shot On-the-Fly Event Schema Induction** [EACL2023] [[paper link](https://aclanthology.org/2024.findings-eacl.103.pdf)]

- **Open-Domain Hierarchical Event Schema Induction by Incremental Prompting and Verification** [ACL2023] [[paper link](https://aclanthology.org/2023.acl-long.312.pdf)]

- **PRIMO: Progressive Induction for Multi-hop Open Rule Generation** [COLING2024] [[paper link](https://aclanthology.org/2024.lrec-main.1137.pdf)]

- **Exploring the Evolution-Coupling Hypothesis: Do Enzymes’ Performance Gains Correlate with Increased Dissipation?** [Entropy, 2025] [[paper link](https://www.mdpi.com/1099-4300/27/4/365)]


###  Data Augmentation

#### Human Intervention

- **Semi-supervised New Event Type Induction and Event Detection** [EMNLP2020] [[paper link](https://aclanthology.org/2020.emnlp-main.53.pdf)]

- **Large Scale Substitution-based Word Sense Induction** [ACL2022] [[paper link](https://aclanthology.org/2022.acl-long.325.pdf)]

- **Human-in-the-Loop Schema Induction** [ACL2023] [[paper link](https://aclanthology.org/2023.acl-demo.1.pdf)]

- **Semi-supervised New Event Type Induction and Description via Contrastive Loss-Enforced Batch Attention** [EACL 2023] [[paper link](https://aclanthology.org/2024.eacl-long.42.pdf)]

- **A(More) Realistic Evaluation Setup for Generalisation of Community Models on Malicious Content Detection** [NAACL2024] [[paper link](https://aclanthology.org/2024.findings-naacl.30.pdf)] 

#### External Knowledge

- **“A Little Birdie Told Me ... ”- Inductive Biases for Rumour Stance Detection on Social Media** [EMNLP2020] [[paper link](https://aclanthology.org/2020.wnut-1.31.pdf)]

- **Knowledge-Enriched Event Causality Identification via Latent Structure  Induction Networks** [ACL2021] [[paper link](https://aclanthology.org/2021.acl-long.376v2.pdf)]

- **Bilingual Lexicon Induction via Unsupervised Bitext Construction and Word Alignment** [ACL2021] [[paper link](https://aclanthology.org/2021.acl-long.67.pdf)]

- **Video-aided Unsupervised Grammar Induction** [NAACL2021] [[paper link](https://aclanthology.org/2021.naacl-main.119.pdf)]

- **Fire Burns, Sword Cuts: Commonsense Inductive Bias for Exploration in Text-based Games**  [ACL2022] [[paper link](https://aclanthology.org/2022.acl-short.56.pdf)] 

- **IAG: Induction-Augmented Generation Framework for Answering Reasoning Questions** [EMNLP2023] [[paper link](https://aclanthology.org/2023.emnlp-main.1.pdf)]

- **Limitations of Language Models in Arithmetic and Symbolic Induction** [ACL2023] [[paper link](https://aclanthology.org/2023.acl-long.516.pdf)]

- **Re-evaluating the Need for Multimodal Signals in Unsupervised Grammar Induction** [NAACL2024] [[paper link](https://aclanthology.org/2024.findings-naacl.70.pdf)]

- **How Lexicalis Bilingual Lexicon Induction?** [NAACL2024] [[paper link](https://aclanthology.org/2024.findings-naacl.273.pdf)]

- **INDUCT-LEARN: Short Phrase Prompting with Instruction Induction** [EMNLP2024] [[paper link](https://aclanthology.org/2024.emnlp-main.297.pdf)]

- **Inductive Linguistic Reasoning with Large Language Models**  [ACL2025] [[paper link](https://aclanthology.org/2025.findings-acl.1171.pdf)]

- **Decision Tree Induction Through LLMs via Semantically-Aware Evolution** [ICLR2025] [[paper link](https://arxiv.org/pdf/2503.14217)]


#### Structured Signals

- **Probing as Quantifying Inductive Bias** [ACL2022] [[paper link](https://aclanthology.org/2022.acl-long.129.pdf)]

- **On Bilingual Lexicon Induction with Large Language Models** [EMNLP2023] [[paper link](https://aclanthology.org/2023.emnlp-main.595.pdf)]

- **Learning Query Adaptive Anchor Representation for Inductive Relation Prediction** [ACL2023] [[paper link](https://aclanthology.org/2023.findings-acl.882.pdf)]

- **Learning Rule-Induced Subgraph Representations for Inductive Relation Prediction** [NIPS2023] [[paper link](https://papers.nips.cc/paper_files/paper/2023/file/0b06c8673ebb453e5e468f7743d8f54e-Paper-Conference.pdf)]

- **Leveraging Grammar Induction for Language Understanding and Generation** [EMNLP2024] [[paper link](https://aclanthology.org/2024.findings-emnlp.259.pdf)]

## Benchmarks

| **Benchmark Name** | **Paper Name** |**Paper Link** |
|----------------|----------------|----------------|
| **SCAN**| **Human few-shot learning of compositional instructions** |[paper link](https://mindmodeling.org/cogsci2019/papers/0123/index.html) |
| **ARC**| **On the Measure of Intelligence** |[paper link](https://arxiv.org/abs/1911.01547) |
| **List Functions**| **The child as hacker: building more human-like models of learning** |[paper link](https://dspace.mit.edu/handle/1721.1/129232) |
| **PROGES** |**A large-scale benchmark for few-shot program induction and synthesis** |[paper link](https://proceedings.mlr.press/v139/alet21a.html) |
| **SyGuS**|**BUSTLE: bottom-up program synthesis through learningguided exploration** |[paper link](https://openreview.net/forum?id=yHeg4PbFHh) |
| **ACRE** |**ACRE: Abstract Causal REasoning Beyond Covariation** |[paper link](https://ieeexplore.ieee.org/document/9578352/) |
| **ILP**|**Neuro-Symbolic Hierarchical Rule Induction** |[paper link](https://proceedings.mlr.press/v162/glanois22a.html) |
| **Instructions**|**Instruction Induction: From Few Examples to Natural Language Task Description** |[paper link](https://arxiv.org/abs/2205.10782) |
| **Arithmetics**|**Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks** |[paper link](https://doi.org/10.18653/V1/2024.NAACL-LONG.102) |
| **Levy/Holt**|**Explicit Inductive Inference using Large Language Models** |[paper link](https://doi.org/10.18653/V1/2024.FINDINGS-EMNLP.926) |
| **NutFrame**|**NutFrame: Frame-based Conceptual Structure Induction with LLMs** |[paper link](https://aclanthology.org/2024.lrec-main.1079) |
| **DEER**|**Language Models as Inductive Reasoners** |[paper link](https://aclanthology.org/2024.eacl-long.13/) |
| **RULEARN** |**IDEA: Enhancing the Rule Learning Ability of Large Language Model Agent through Induction, Deduction, and Abduction** |[paper link](https://arxiv.org/abs/2408.10455) |
| **Cryptography**|**Patterns Over Principles: The Fragility of Inductive Reasoning in LLMs under Noisy Observations** |[paper link](https://aclanthology.org/2025.findings-acl.1006/) |
| **GeoILP**|**Geoilp: A synthetic dataset to guide large-scale rule induction** |[paper link](https://openreview.net/forum?id=cfGpIcOIa5) |
| **InductionBench**|**InductionBench: LLMs Fail in the Simplest Complexity Class** |[paper link](https://aclanthology.org/2025.acl-long.1287/) |
| **CodeSeq**|**Code-driven Number Sequence Calculation: Enhancing the inductive Reasoning Abilities of Large Language Models** |[paper link](https://www.arxiv.org/abs/2510.14620) |

## Analysis

- **Demystifying Inductive Biases for (Beta-)VAE Based Architectures** [ICML2021] [[paper link](http://proceedings.mlr.press/v139/zietlow21a/zietlow21a.pdf)]

- **Effects of Parameter Norm Growth During Transformer Training: Inductive Bias from Gradient Descent** [EMNLP2021] [[paper link](https://aclanthology.org/2021.emnlp-main.133.pdf)] 

- **Examining the Inductive Bias of Neural Language Models with Artificial Languages** [ACL2021] [[paper link](https://aclanthology.org/2021.acl-long.38.pdf)]

- **What they do when in doubt: a study of inductive biases in seq2seq learners** [ICLR2021] [[paper link](https://arxiv.org/abs/2006.14953)]

- **Predicting Inductive Biases of Pre-Trained Models** [ICLR2021] [[paper link](https://openreview.net/pdf?id=mNtmhaDkAr)]

-  **The Inductive Bias of ReLU Networks on Orthogonally Separable Data** [ICLR2021] [[paper link](https://openreview.net/pdf?id=krz7T0xU9Z_)]

- **Inductive Biases and Variable Creation in Self-Attention Mechanisms** [ICML2022] [[paper link](https://proceedings.mlr.press/v162/edelman22a/edelman22a.pdf)]

- **Fast Rates for Noisy Interpolation Require Rethinking the Effects of Inductive Bias** [ICML2022] [[paper link](https://proceedings.mlr.press/v162/donhauser22a/donhauser22a.pdf)]

- **The SSL Interplay: Augmentations, Inductive Bias, and Generalization** [ICML2023] [[paper link](https://openreview.net/pdf?id=d2aohFmZoB)]

- **Injecting structural hints: Using language models to study inductive biases in language learning** [EMNLP2023] [[paper link](https://aclanthology.org/2023.findings-emnlp.563.pdf)]

- **Inductive Bias Is in the Eye of the Beholder** [EMNLP2023] [[paper link](https://aclanthology.org/2023.genbench-1.12.pdf)]

- **Measuring Inductive Biases of In-Context Learning with Underspecified Demonstrations**  [ACL2023] [[paper link](https://aclanthology.org/2023.acl-long.632.pdf)]

- **Instruction Induction: From Few Examples  to Natural Language Task Descriptions** [ACL2023] [[paper link](https://aclanthology.org/2023.acl-long.108.pdf)]

- **How to Plant Trees in LMs: Data and Architectural Effects on the Emergence of Syntactic Inductive Biases**  [ACL2023] [[paper link](https://aclanthology.org/2023.acl-long.629.pdf)] 

- **A theoretical study of inductive biases in contrastive learning** [ICLR2023] [[paper link](https://arxiv.org/pdf/2211.14699)]

- **Strong inductive biases provably prevent harmless interpolation** [ICLR2023] [[paper link](https://arxiv.org/pdf/2301.07605)]

- **Identifying Semantic Induction Heads to Understand In-Context Learning** [ACL2024] [[paper link](https://aclanthology.org/2024.findings-acl.412.pdf)] 

- **The Evolution of Statistical Induction Heads: In-Context Learning Markov Chains** [NIPS2024] [[paper link](https://papers.nips.cc/paper_files/paper/2024/file/75b0edb869e2cd509d64d0e8ff446bc1-Paper-Conference.pdf)]

- **Unveiling Induction Heads: Provable Training Dynamics and Feature Learning in Transformers** [NIPS2024] [[paper link](https://papers.nips.cc/paper_files/paper/2024/file/7aae9e3ec211249e05bd07271a6b1441-Paper-Conference.pdf)]

- **Inductive biases of multi-task learning and finetuning: multiple regimes of feature reuse** [NIPS2024] [[paper link](https://papers.nips.cc/paper_files/paper/2024/file/d7346ff79699b5bba26f8af89e700a95-Paper-Conference.pdf)]

- **What needs to go right for an induction head? A mechanistic study of in-context learning circuits and their formation** [ICML2024] [[paper link](https://openreview.net/pdf?id=O8rrXl71D5)]

- **Position: The No Free Lunch Theorem, Kolmogorov Complexity, and the Role of Inductive Biases in Machine Learning** [ICML2024] [[paper link](https://openreview.net/pdf?id=EaJ7nqJ2Fa)]

- **Induction Heads as an Essential Mechanism for Pattern Matching in  In-context Learning** [NAACL2025] [[paper link](https://aclanthology.org/2025.findings-naacl.283.pdf)]

- **Can Input Attributions Explain Inductive Reasoning  in In-Context Learning?** [ACL2025] [[paper link](https://aclanthology.org/2025.findings-acl.1092.pdf)]

- **Beyond Induction Heads: In-Context Meta Learning Induces Multi-Phase Circuit Emergence** [ICML2025] [[paper link](https://openreview.net/pdf?id=Xw01vF13aV)]

- **When Diffusion Models Memorize: Inductive Biases in Probability Flow of Minimum-Norm Shallow Neural Nets** [ICML2025] [[paper link](https://openreview.net/pdf?id=WD2CKUrxmx)]

- **Geometric Inductive Biases of Deep Networks: The Role of Data and Architecture** [ICLR2025] [[paper link](https://arxiv.org/pdf/2410.12025)]
