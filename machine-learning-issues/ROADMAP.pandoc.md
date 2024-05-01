---
title: "Roadmap of Machine Learning Issues"
author: "Yuxuan Jiang"
bibliography: [./citations.bib]
csl: ieee.csl
---

# Roadmap of *Machine Learning Issues*

From a high-level perspective, this project aims to help developers detect silent/latent correctness issues (bugs) in machine learning pipelines before those issues result in unbearable costs. This file documents the storyline of the project, the milestones, and the goals of each milestone. It is a living document that will be updated as the project progresses.

## 1. Key Research Questions

The following are the key research questions that we will answer in this project:

1. What are the silent/latent correctness issues of machine learning pipelines?

    Reasons why this question is important:
    1. The insights from this question will help us understand the problem space and guide our design decisions.
    2. The insights might also help future researchers to design better abstractions to prevent those issues from happening in the first place.
2. How can we detect those silent/latent correctness issues early on?

    Reasons why this question is important:
    1. A tool that can detect those issues early on will help developers save a lot of time and money.

The meaning of "silent/latent" is that the issues are either completely silent (i.e., no error message is reported) or only raise exceptions when the pipeline is executed for a long time (e.g., hours).

For now, we are focusing on user level issues (i.e., issues that are caused by the user code or the user data). We are not focusing on issues that are caused by the environment (e.g., GPU drivers, CUDA, etc.) or issues that are caused by the ML library itself (e.g., PyTorch, TensorFlow, etc.).

Please see [CONTRIBUTING.md#bug-selection](https://github.com/OrderLab/machine-learning-issues/blob/main/CONTRIBUTING.md#bug-selection) for more details.

### What are not (necessarily) the scope of this project?

1. **Hyperparameter tuning**: This is a very important yet heavily studied problem. As a result, we want to focus on other problems that are less studied but still matters a lot.
2. **Solving environment issues**: Engineering-wise, this is a very hard problem to solve due to the complexity of the Python ecosystem, lack of standardization in the ML community, and the lack of control over the user's environment. We feel like this problem is better solved by the ML community as a whole or by the Software Engineering community.
3. **Issues that are not silent/latent**: Issues that raise exceptions early on are usually easy to detect and fix. Though they still troubles developers a lot, the cost of such issues and effort to debug them are usually not as high as silent/latent issues.
4. **Finding issues in GPU Drivers, CUDA, or other non-user-level code**: Quality assurance of those components is very important, and our project should be able to help with that. **Our technique is indeed aimed at detecting general silent/latent correctness issues in ML pipelines due to various root causes despite their location no matter it is in user level code or the environment. However, we choose to focus on user issues for now as they are easier to control.** For example, a PyTorch API raising an exception might be caused by a bug in PyTorch, a bug in CUDA, or a bug in the user code or an environment installation issue. As a result, we feel like for now it is better to focus on user issues as they are easier to control. We might consider expanding our scope in the future.

## 2. Introduction & Expected Contribution

As machine learning (ML) gains popularity in both academia and industry, the correctness of ML pipelines becomes increasingly important. A silent or latent correctness issue in an ML pipeline often goes unnoticed, which adds to the challenges in timely detection and debugging of such issues. The cost of detecting and fixing such issues is further amplified as the scale of ML projects increases astronomically with the recent LLM arms race [@huggingface2022bloom, @googleai2023gemini, @metaai2023llama], usually involving hours or days of downtime and hours or days of wasted machine hours [@bigscience2022hanging].

<!-- Silent Issue Example From Bloom -->
For example, a gradient clipping bug in [Deepspeed's BF16Optmizer](https://github.com/microsoft/DeepSpeed/pull/1801/commits/e24814a10de04ce280efe2adb027b023e3336493) caused the certain parts of the model to silently diverge during training. Though the bug was detected before the divergence became too huge, the developers still had to spend 12 days to fix the bug [@bigscience2022hanging]. On the other hand, if the bug was not detected early on, the developers might have to face the cost of retraining the model from scratch as the end model weights will be inconsistent.

<!-- Latent Issue Example -->
Another example is that a model expecting input of certain batch size might raise an exception at the end of the training when the size of the training dataset is not a multiple of the batch size. This latent issue will cause the entire training to fail after hours of training [@jhoo2021static].

Reproducing and fixing silent/latent issues imposes a huge burden on developers. Manifestation of such issues usually requires long execution time to surface in the form of exceptions or noticeable discrepancies in the metrics. Specific input might also be required for debugging, which is hardly available. As reported in CMU SEI's Interview with data scientists [@lewis2021characterizing], ”A typical thing that might happen is that in the production environment, something would happen. We would have a bad prediction, some sort of anomalous event. And we were asked to investigate that. Well, unless we have the same input data in our development environment, we can’t reproduce that event.”. In addition, even if the bug has been detected, it is also hard to estimate the time needed to fix it, as ML engineering still largely relies on trial and error [@8987482]. Consequently, a tool that can detect such issues early on will help developers save a lot of time and money.

<!-- What's the challenges of finding silent/latent correctness issues? -->
Silent or latent correctness issues are easy to make and hard to detect. Here we try to more formally define the underlying reasons:

<!-- CMU SEI Blog https://insights.sei.cmu.edu/blog/detecting-mismatches-machine-learning-systems/
developing an AI/ML model is a statistical problem that is relatively fast and cheap; but deploying, evolving, and maintaining models and the systems that contain them is an engineering problem that is hard and expensive. -->

1. Flexible and dynamic development process of ML pipelines. Thus, ML pipelines are less formally managed [@lewis2021characterizing, @10.5555/2969442.2969519].

2. Complexity of the ML stack. ML libraries usually have a large number of APIs and complex configuration knobs. Thus, it is hard for developers to correctly understand API usage and to formally verify the correctness of their pipelines [@humbatova2019taxonomy]. Dynamic types and the implicit broadcasting and type conversion rules also makes it harder to detect issues at compile time.

3. Data-dependent nature of ML pipelines. Thus, it is harder to apply traditional software engineering techniques (e.g., unit testing) to ML pipelines. For example, a dataset-model mismatch can lead to silent accuracy decrease ([PyTorch#FORUM84911](https://github.com/OrderLab/machine-learning-issues/blob/main/PyTorch-FORUM84911/README.md#pytorch-forum84911)), Additionally, certain correctness issues might only be triggered by certain data [@li2023reliability, @humbatova2019taxonomy].  

4. No explicit manifestation of the issue (e.g., no error message or discrepancy needs time to accumulate).

<!-- What's missing in existing work, from a high-level perspective? -->
Existing works fall short in the following ways:

1. Existing work tend to focuse on ensuring quality of ML models (DeepXplore [@Pei_2017]), DL frameworks (CRADLE [@8812095] and DeepREL [@deng2022fuzzing]), and compilers (NNSmith [Liu_2023]) rather than the quality of ML pipelines where bugs can reside in user-level code.
2. Existing work focusing on ML pipelines focuses on a very specific type of issues (e.g. PyTea [@jhoo2021static], RANUM [@li2023reliability]) or are limited because they need prior runs of the same pipeline to pinpoint the issue (e.g. MLDebugger [@Louren_o_2019]).
3. Existing work struggle to scale beyond simple ML pipelines due to the enormous amount of Python syntax and API to support and the inherent limitation of static analysis [@jhoo2021static].
<!-- “Real-world machine learning applications heavily utilize third-party libraries, external datasets, and configuration parameters, and handle their controls with subtle branch conditions and loops, but the existing tools still lack in supporting some of these elements and thus they fail to analyze even a simple ML application” ([Jhoo et al., 2021, p. 3](zotero://select/library/items/3BQZ8HNH)) ([pdf](zotero://open-pdf/library/items/TPTNR77J?page=3&annotation=UMVSK2LS)) -->

<!-- What's our approach? i.e. our hypothesis -->
We want to create a general-purpose tool that can detect a wide range of silent/latent correctness issues in ML pipelines. Our high-level approach is to leverage dynamic analysis to infer the likely invariants of ML pipelines and then use those invariants to detect correctness issues.

A few directions we are considering:

1. **Repository Mining + Static Analysis**:
Infer library API specifications (pre-conditions and post conditions) from examples [@pytorch-examples]. Then, use the inferred specifications to statically verify the user code connecting two API calls in the new ML pipeline.
    - Pro: The inferred specifications are likely to be correct and comprehensive as they are mined from a huge number well-maintained examples.
    - Con: The inferred invariants might be best for detecting issues that raises exceptions. It might not be as good for detecting issues that are completely silent as these are usually caused by the data or inappropriately configuration parameters.

2. **Predictive Analysis**: The hypothesis is that although ML pipelines are expensive to run, we can use a small number of executions to predict the correctness of future executions.
    - Pro: The cost of running ML pipelines is reduced.
    - Con: The correctness of the predictions is not guaranteed.

3. **Online Anomaly Detection**: The hypothesis is that the silent/latent correctness issues will cause the execution of the ML pipeline to violate certain invariants. We can use online anomaly detection to detect such violations.

<!-- What's our expected contribution? -->
We expect to make the following contributions:

1. In-depth understanding of the silent/latent correctness issues of ML pipelines.

2. A tool that can detect a wide range of correctness issues in ML pipelines at compile time or early on during runtime.

## 3. Stages

1. Understanding Real-World Silent/Latent Correctness Issues [*Current Stage*]
    - [x] Collecting real-world issues
        - [x] Collecting issues from GitHub issues
        - [x] Collecting issues from StackOverflow
        - [x] Collecting issues from other sources (e.g., Reddit, Twitter, etc.)
    - [x] Analyzing the issues
    - [x] Summarizing the issues
2. Testing the Hypothesis [*Current Stage*]
    - [ ] Testing all the hypothesis we have
        - [ ] Implementing prototypes
        - [ ] Evaluating prototypes
3. Iterating on the Hypothesis
    - [ ] Iterating on the hypothesis
        - [ ] Desiding on one potential direction to focus on
        - [ ] Finish iterating on the direction with data collected from Stage 1
        - [ ] Repeating Stage 2 and 3
    - [ ] Large-Scale Evaluation
        - [ ] Collecting more real-world issues (probably from industry collaborators)
        - [ ] Evaluating the tool on the issues
4. Large-Scale Evaluation
    - [ ] Collecting more real-world issues (probably from industry collaborators)
    - [ ] Evaluating the tool on the issues
5. Writing Paper
    - [ ] Writing the paper
    - [ ] Submitting the paper

## 4. Milestones

### 4.1. Milestone 1: Understanding Real-World Silent/Latent Correctness Issues

- **Outcome**: A list of 10 real-world silent/latent correctness issues that 1) meet our bug-choosing criteria, 2) are reproducible, and 3) are analyzed. We will also analyze the issues and summarize the findings, to guide our design decisions.
- **Satisfying Criteria**: This milestone will be considered as satisfied if we have several issues that meet our bug-choosing criteria, are reproducible, and are analyzed.
- **Deadline**: No deadline. We will keep collecting issues even after this milestone is satisfied.

### 4.2. Milestone 2: Have a working prototype

- **Outcome**: A working prototype built from our hypothesis that can detect a wide range of silent/latent correctness issues in ML pipelines. The minimum requirement is being able to infer certain types of constraints from the pipeline.
- **Satisfying Criteria**:
  - This milestone will be considered as satisfied if the working prototype shows promising results on the issues collected in Milestone 1.
  - A working prototype is considered as "working" if it can detect some silent/latent correctness issues in ML pipelines at a satisfying practicality (e.g., the cost of running the pipeline is not too high).
- **Deadline**: 2024 Winter

### 4.3. Milestone 3: Reaching a satisfying utility

- **Outcome**: A tool that can detect a wide range of silent/latent correctness issues in ML pipelines with a satisfying utility.
- **Satisfying Criteria**: This milestone will be considered as satisfied if the tool can detect a wide range of silent/latent correctness issues in ML pipelines with a satisfying utility. We also need a very large set of real-world issues to evaluate the tool.
- **Deadline**: 2024 Summer

### 4.4. Milestone 4: Writing Paper

- **Outcome**: A paper that describes the tool and the findings.
- **Satisfying Criteria**: This milestone will be considered as satisfied if the paper is accepted by a top-tier conference.
- **Deadline**: 2024 Fall (Before the OSDI deadline)

## 5. Goals by the end of this semester (2024 Winter)

- Finish Milestone 1 and 2.
- Have a plan for Milestone 3.
- Have a revised introduction, background, and related work section for the paper (Milestone 4).

## 6. Existing Flaws
<!-- Mostly on understanding the issues and how far has the research community gone. -->

Since this project aims to leverage dynamic invariant inference to detect silent/latent correctness issues in ML pipelines, the following are the existing flaws that we need to address threats to validity related to this approach:

1. Input:
    - Whether the input (e.g. collected trace, ml pipelines) is representative of real-world ML pipelines?
    - Whether the input itself is correct? If not, the invariants inferred from the input might be incorrect.
    - Noise in the input (e.g. irrelevant variables, or indeterminism, etc.) might affect the quality of the inferred invariants.
2. Performance:
    - Whether the performance of the tool is acceptable? If not, the tool might not be practical.
3. Accuracy:
    - Whether the inferred invariants are accurate? If not, the tool might not be able to detect the issues.

## References
