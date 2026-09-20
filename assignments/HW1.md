---
layout: page
title: HW1
parent: Assignments
nav_order: 1
---

# Homework 1: Exploratory Analysis of a Music Catalog
{:.no_toc}

[Handout (PDF)](/IntroML-f26/assets/pdf/HW1.pdf) · [Starter notebook](/IntroML-f26/assets/notebooks/hw1_starter.ipynb)

| | |
|:--|:--|
| Pre-Registration Report Submission Deadline | **9/30 11:59 PM** |
| Final Report and Code Submission Deadline | **10/8 11:59 PM** |

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Introduction

You are given a music catalog containing **114,000 track records**. In this assignment, you will treat this catalog as a dataset and investigate its quality, structure, and predictive value.

Suppose a team ultimately wants to use this dataset to build machine-learning systems for tasks such as **genre tagging** and **predicting track popularity**. Before deciding what models to build, however, the team first needs to understand the dataset itself:

- Is the dataset reliable enough to use as training data?
- What structure is present in the audio-feature space?
- Which patterns appear to reflect meaningful properties of the tracks, and which may instead result from the way the dataset was collected or represented?
- Which target variables can be predicted meaningfully from the available audio features, and which cannot?

Your goal in this assignment is to answer these questions with evidence.

You will first audit the dataset for data-quality problems and make explicit decisions about how those problems should be handled. You will then use **principal component analysis** to study the geometry of the audio-feature space, use **clustering methods** to investigate structure among genres and individual tracks, and **fit regression models** for two prediction targets and compare their predictive performance. Finally, you will summarize what your results imply about the **strengths and limitations** of this dataset for machine-learning applications.

The purpose of the assignment is **not to obtain the highest possible predictive accuracy**. Instead, you are expected to connect each conclusion to appropriate evidence, **explain the reasoning behind your analysis choices, and recognize when the data do not support a strong conclusion**. A weak or unsuccessful predictive result can therefore be a correct and important finding if you demonstrate it carefully and explain what it means.

A later assignment will build on the groundwork established here. For Homework 1, all empirical analyses must use only the variables provided in this catalog. **Do not add external datasets, external features, or information collected from other sources.** Your conclusions should be supported by evidence from your analysis of this dataset.

## Rules

1. This is an individual assignment. You may discuss course concepts and general approaches with classmates, but the analysis you perform, the code you submit, and the writing in your report must be your own. Do not copy another student's code, results, figures, or written explanations. **Any submission found to contain plagiarism or copied work will receive a score of zero for the assignment.**
2. **AI tools**, including coding assistants and coding agents, are **permitted**. If you use them, you are still responsible for understanding, checking, and being able to explain or modify everything you submit. You must also submit the AI-use log described in [Section 7](#section-7-ai-use-log).
3. **Write the report yourself.** Do not use AI tools to generate the content of your report. We want to read your own reasoning in your own words. Plain language is fine: a short, clear explanation of what you did and why is worth more than a polished paragraph you cannot defend at the demo. Reports that read as AI-generated will be checked at the demo, and you will be asked to explain each conclusion.
4. The report may be written in **either Traditional Chinese or English**. There are no specific requirements for the report format.
5. Google Colab is sufficient for this assignment to be accomplished, no GPU is required. You may use standard Python packages including `pandas`, `numpy`, `scipy`, `matplotlib`, and `scikit-learn`. There are **no restrictions on the Python libraries** you may use.
6. **Submit your work on eeclass**. The pre-registration report mentioned in [Section 2](#section-2-pre-registration) is submitted separately and has an earlier deadline. **No late submission will be accepted.** The main submission consists of:

   | Submission | Deadline |
   |:--|:--|
   | A **pre-registration report** in the PDF format, up to **2 pages.** File name: `HW1_pre_report_{student_id}.pdf` | 9/30 11:59 PM |
   | A **final report** in the PDF format, up to **10 pages.** File name: `HW1_final_report_{student_id}.pdf` | 10/8 11:59 PM |
   | An **.ipynb notebook** containing all code needed to reproduce the results reported in your submission. File name: `HW1_code_{student_id}.ipynb` | 10/8 11:59 PM |
   | The **AI-use log** document. File name: `HW1_ai_use_{student_id}.md` | 10/8 11:59 PM |

## Materials

The starter notebook (`hw1_starter.ipynb`) and raw dataset (`tracks.csv`) for Homework 1 have been posted on eeclass. Please head over to the platform for full details and instructions.

## Section 1: The Data

The raw dataset is available from <https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset>.

For this assignment, the dataset has been downloaded and provided to you as `tracks.csv`.

The file `tracks.csv` contains 114,000 rows and 20 columns. Each row represents one track-genre listing. The same track may therefore appear in more than one row if it is associated with multiple genre listings.

The dataset contains the following columns:

| Columns | Type | Description |
|:--|:--|:--|
| `track_id`, `artists`, `album_name`, `track_name` | String | Track identifiers and metadata |
| `track_genre` | String | The genre category associated with the track |
| `popularity` | Integer | A popularity score provided by the streaming platform (0~100) |
| `duration_ms` | Integer | Track duration in milliseconds |
| `explicit` | Boolean | Indicator for explicit lyrics |
| `danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence` | Float | Audio features automatically extracted from the track's audio |
| `loudness` | Float | Estimated loudness of the track |
| `tempo` | Float | Estimated tempo in BPM |
| `key`, `mode`, `time_signature` | Integer | Musical attributes stored as integer-coded values |

Do not assume that every field was measured correctly. Part of your task is to examine the dataset and determine whether these assumptions are justified.

## Section 2: Pre-Registration (5%)

Submit this section report **before 9/30 11:59 PM**.

The purpose of this task is to make your expectations explicit before you analyze the full dataset. Later, you will compare these expectations with your actual results and reflect on which expectations were supported, which were not, and what evidence changed your mind.

You may read the column descriptions in [Section 1](#section-1-the-data) and inspect some rows of `tracks.csv`. **Do not compute summary statistics, create plots, search the full dataset, or fit any models before submitting this section.**

In at most two pages, provide the following inside the report:

1. (2%) **Two testable expectations** about the dataset. For each expectation, state:
   1. What you expect to observe?
   2. What result would show that your expectation was not supported by the data?
2. (1%) **One data-quality problem you think might be in the dataset.** Base your guesses only on the information available at this stage. They do not need to be correct.
3. (1%) **Predicted held-out error for two regression tasks.** Before fitting any model, predict the held-out root-mean-square error (RMSE) you expect for each task. Report each prediction in the same units as the target and give one sentence explaining your reasoning. Later in Section 6, you will use the available features to predict:
   1. A track's `energy`
   2. A track's `popularity`.
4. (1%) **One question about the dataset, labels, or data-collection process** that you would want answered before relying on a model trained on it.

Your pre-registration is graded on **completeness and specificity**, not on whether your predictions are correct. You will compare these expectations with your actual findings later in the assignment and discuss in the report.

- Submitted by September 30: Up to 5 points.
- Submitted between October 1 and October 8: Up to 2 points.
- Not submitted: Section 2 receives 0 points, and Question 1 in Section 6 cannot receive credit since there is no prediction to restate.

## Section 3: Auditing the Dataset (12%)

Real-world datasets often contain problems that can affect a machine-learning analysis, for example, missing or invalid values, repeated records, inconsistent entries, or variables whose meaning is easy to misinterpret.

Your task in this section is to **identify** the most important **data-quality** problems in `tracks.csv`, explain why they matter, and decide how you will handle them before continuing with the rest of the assignment.

1. (8%) Identify and prioritize **at least four data-quality findings**, ordered from most to least important based on how strongly you believe each one could affect the analyses or models in this assignment. For each finding, include:
   1. **Evidence**: one number, table, or figure showing that the issue is present.
   2. **Why it matters**: explain how the issue could affect a later analysis or model.
   3. **Your decision**: state what you will do about it.
   4. **Trade-off:** explain what information, data, or reliability you may lose because of that decision.
2. (2%) **Justify your ranking.** Explain why you ranked your first finding as more important than your second.
3. (2%) **Apply your cleaning decisions.** In your submitted notebook, include the code that transforms the raw dataset into the cleaned dataset used in your analysis. The code should be reproducible from the original `tracks.csv` dataset.

If you identify a problem but decide not to change the data because of it, explain that decision in your report and treat the issue as a limitation.

Simply listing unusual values or potential problems is not sufficient. Your findings should connect **evidence, impact, and a justified decision**.

Use the cleaned dataset for Sections 4 to 6, and state any place where you deliberately use the raw data instead.

## Section 4: The Geometry of the Feature Space (10%)

The numeric audio features place each track at a point in a multidimensional feature space. In this section, use **principal component analysis (PCA)** to understand the main directions in which the tracks vary, how many of those directions are important, and how feature scaling affects the result:

1. (3%) **Effective dimensionality.** Use PCA to estimate how many principal components are needed to represent most of the variation in the audio features. Clearly state the criterion or metrics you use to decide how many components are sufficient.
2. (2%) **Interpret the leading principal components.** For the first few principal components, explain in plain language what kind of variation each component appears to capture.
3. (4%) **Effect of standardization.** Repeat the PCA without standardizing the features first. Compare the result with the standardized PCA and explain why they differ. Your explanation should refer to the quantity that PCA maximizes. Support your explanation with either:
   1. A short mathematical derivation.
   2. A variance calculation using your own data.
4. (1%) **Choose appropriate variables.** Identify which columns in [Section 1](#section-1-the-data) should not be treated as continuous numeric features in this PCA analysis. Briefly explain why, and state how you handled them.

## Section 5: Structure in the Data (10%)

The goal of this section is to investigate whether tracks with similar audio features naturally form groups, and whether those groups reflect meaningful musical structure rather than artifacts of how the dataset was collected.

You will examine this question at two levels: **genres** and **individual tracks**:

1. (4%) **Clustering genres.** Create one representative audio-feature profile for each genre, and use hierarchical clustering to group similar genres together. Show how the resulting genre groupings differ, and explain which result you find more reasonable based on evidence from the data. Compare either:
   1. Two different measures of dissimilarity.
   2. Two different linkage methods.
2. (3%) **Clustering individual tracks.** Using the standardized audio features, cluster the individual tracks (for example, with k-means). To choose the number of clusters k, compute at least two quantitative criteria (such as the elbow of the inertia curve and the silhouette score) over a range of k, show them, and explain your final choice. Then describe what each resulting cluster represents in terms of its audio characteristics (for example, by reporting the mean of each feature per cluster). Some clustering methods or criteria may be expensive to run on all 114,000 rows. If you use a subset of the data for clustering or evaluation, clearly state how the subset was selected and why.
3. (3%) **Are the clusters meaningful?** A clustering algorithm always returns k clusters, even on data with no real structure. Provide evidence that the clusters you found in Question 2 reflect real differences between tracks, rather than being an artifact of the algorithm, the sampling, or the way genres were assigned. Provide at least two evidence. For example:
   1. Plot the clusters in the space of the first two principal components and explain whether the cluster boundaries align with the directions of variation you interpreted there.
   2. **Stability.** Re-run the clustering with a different random seed, a different subsample, or after applying your cleaning decisions from Section 3, and report how much the cluster assignments change.

## Section 6: Two Regression Targets (13%)

The goal of this section is to compare how well the available features can predict two different targets, and to understand why the predictive performance differs.

You will build regression models for `energy` and `popularity` features.

Use a **held-out test set** to evaluate predictive performance. Unless otherwise stated, use the remaining appropriate features in the dataset as predictors, and do not include the target itself among the input features. You must deliver:

1. (3%) **Compare your predictions with the actual results.** Restate the RMSE values you predicted in [Section 2](#section-2-pre-registration). Then fit a **linear regression model** for each target and report its **held-out RMSE**. For each target, also report the RMSE of a simple baseline model that ignores the input features and always predicts the mean target value from the training set.
2. (3%) **Explain the difference in predictive performance**. Compare the two regression tasks and explain why one target is easier or harder to predict from the available features. Support your explanation with evidence from this dataset, such as relationships between the target and the input features, rather than only making a general statement such as "the data are noisy."
3. (3%) **Look for additional predictive information.** If one target is difficult to predict from the audio features, look for **one other variable in the dataset that may contain useful information about that target**. For example, pick one non-audio column in `tracks.csv` that you think carries information about the target. Support your choice with an appropriate measurement or plot.
4. (4%) **Study the effect of regularization.** Use `energy` as the target for this question. Identify several input features that are strongly correlated with one another. Standardize these features, then fit a **regularized linear regression model** using these features, and **vary the regularization strength** over several orders of magnitude. Report how the coefficients and held-out RMSE change as regularization becomes stronger.

   If regularization does not improve held-out performance on the full dataset, say so. Then **create a setting in which regularization has a clearer effect,** for example by using a smaller training set, a richer set of basis functions (such as polynomial features), or both.

## Section 7: AI-Use Log

AI tools are permitted in this assignment. The purpose of this log is to document how you used them and how you verified their outputs.

Submit an AI-use log written in the markdown file. If you used any AI tool during the assignment, briefly describe:

1. **What you used AI for.** List the main parts of the assignment for which you used AI, such as writing or debugging code, explaining a concept, suggesting an analysis, interpreting a result, or revising your writing.
2. **How you checked the output.** For each major use, briefly explain how you verified that the AI-generated output was correct or appropriate. For example, you might compare it with lecture material, inspect the code, run an additional experiment, or check the result against the data.

If you did not use any AI tools, simply state that in the log.

You remain responsible for all code, analyses, figures, and explanations in your submission. **You should be able to explain any AI-assisted work during the demo session.**

A 5-point deduction will be applied for missing submissions or conspicuously low-effort responses (e.g., merely writing "I used ChatGPT").

## Grading Policy

**Report (50%)**

| Tier | Description | Percentage |
|:--:|:--|--:|
| 4 | Provides evidence, explanation, and decisions, along with a discussion of limitations or alternative interpretations. | 100% |
| 3 | Provides evidence, explanation, and decisions, with coherent logic linking all three. | 80% |
| 2 | Presents results or figures, but the explanation is weak or disconnected from the evidence. | 50% |
| 1 | Only presents results without explanation, or the methodology is clearly flawed. | 20% |
| 0 | No answer provided / Blank. | 0% |

To prevent grader bias, each section is assigned to a dedicated TA who evaluates all students' submissions for that specific part. Rest assured that your work will be assessed with a fair and consistent standard.

**Demo (50%)**

The demo will take place on **Tuesday, November 10, in the evening**, together with the HW2 demo. Further details, including the exact time, location, and demo format, will be announced later.

Feel free to ask questions! For homework-related issues, please post on the eeclass discussion forum, and the TAs will reply promptly. For personal matters or private inquiries, you can reach the TA at <george930502@gapp.nthu.edu.tw>.
