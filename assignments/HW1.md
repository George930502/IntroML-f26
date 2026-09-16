---
layout: page
title: Lab 1
parent: Assignments
nav_order: 1
---

# Lab 1: Exploratory Analysis of a Music Catalog
{:.no_toc}

[Handout (PDF)](/IntroML-f26/assets/pdf/HW1.pdf) · [Starter notebook](/IntroML-f26/assets/notebooks/hw1_starter.ipynb)

| | |
|:--|:--|
| Out | **Thursday, September 24** |
| Pre-registration due | **Monday, September 28** |
| Report due | **Thursday, October 8** |
| Demo | **Week of November 10** |
| Points | Report 50, demo 50 |
| Work | Individual |

**Summary.** You are given a catalog of 114,000 music tracks. Each track is
described by audio features that an automatic analyzer produced from the
recording, together with a genre label. The team you are working with wants to
build a genre tagger and a hit predictor on top of this catalog. Your task is to
establish what the catalog can support before anyone trains such a model on it.
You will audit the file for defects, describe the geometry of the feature space
with PCA, test whether the catalog contains real structure, fit regression
models to two targets of very different difficulty, and state what the catalog
does and does not support. The later assignment builds on the ground prepared
here, but everything you hand in for this one concerns this catalog and nothing
else. Points are awarded for evidence and justification, not for model accuracy.
Several questions below have little or no signal in this data; establishing
that, and explaining why, earns full credit.

> **Deadlines and deliverables**
>
> | | |
> |:--|:--|
> | **Monday, September 28** | Pre-registration, at most one page ([Section 2](#2-pre-registration-4-points)). |
> | **Thursday, October 8** | Report (PDF, at most **6 pages** and **8 figures**), notebook, and AI-use log ([Section 8](#8-ai-use-log)). |
> | **Week of November 10** | Demo ([Section 9](#9-demo-50-points)). |

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Start here: instructions

- **Collaboration policy.** Work individually. You may discuss concepts with
  classmates, but the analysis, the code, and the writing you submit must be
  your own.
- **Use of AI tools.** Permitted, including coding agents. You must submit the
  AI-use log described in [Section 8](#8-ai-use-log), and you must be able to
  explain and modify any part of your submission at the demo.
- **Submitting your work.** Submit on eeclass. The pre-registration in [Section
  2](#2-pre-registration-4-points) is a separate, earlier submission. The main
  submission consists of three items: the report as a PDF of at most six pages
  and at most eight figures; the notebook, which must run from top to bottom;
  and the AI-use log, which does not count toward the page limit. Give your report the same section titles as this handout, in
  the same order.
- **Late submission.** See the course policy on eeclass. The pre-registration is
  the one exception: it earns no credit after its deadline, for the reason given
  in [Section 2](#2-pre-registration-4-points).
- **Materials.** The data file `tracks.csv` is posted on eeclass. The [starter
  notebook](/IntroML-f26/assets/notebooks/hw1_starter.ipynb) loads the file and
  works through one audit finding as an example of the level of evidence
  expected in [Section 3](#3-auditing-the-file-12-points).
- **Environment.** Google Colab is sufficient; no GPU is needed. You may use
  `pandas`, `numpy`, `scipy`, `matplotlib`, and `scikit-learn`.
- **Scope.** Nothing in this assignment requires material taught after October
8. Cross-validation, ROC analysis, and the classifiers introduced later in the
   course are neither required nor rewarded here.

## 1. The data

The file `tracks.csv` contains one row per track and genre listing, with the
following columns.

| Column | Type | Description |
|:--|:--|:--|
| `track_id`, `track_name`, `artists`, `album_name` | text | identifiers and metadata |
| `track_genre` | text | the playlist category from which the row was collected |
| `popularity` | 0–100 | a score computed by the streaming platform |
| `duration_ms` | integer | track length in milliseconds |
| `explicit` | boolean | explicit lyrics flag |
| `danceability`, `energy`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `valence` | 0–1 | audio descriptors produced by an automatic analyzer |
| `loudness` | dB | typically negative |
| `tempo` | BPM | estimated beats per minute |
| `key`, `mode`, `time_signature` | integer | musical attributes stored as integer codes |

The provider states neither that every value in the file was measured
successfully, nor that every row describes a different track. Do not assume
either.

## 2. Pre-registration (4 points)

Submit this section by **Monday, September 28**, before you begin the analysis.
Read the column descriptions in [Section 1](#1-the-data) and inspect no more
than the first 50 rows of the file. In at most one page, state the following.

1. (2 points) Three claims you expect to hold in this catalog. For each claim,
   state the observation that would show it to be false.
2. (1 point) Your predicted held-out root-mean-square error for the two
   regression targets of [Section 6](#6-two-regression-targets-10-points), in
   the units of each target, with one sentence of reasoning for each.
3. (1 point) Two defects you expect to find in the file, and one question you
   would want answered before trusting a model trained on it.

Points are awarded for completeness and specificity, not for being right. You
will compare these statements with your results in [Section
6](#6-two-regression-targets-10-points) and [Section
7](#7-conclusions-4-points), and you will be asked about them at the demo. **A
pre-registration submitted after the deadline earns no credit**, because its
purpose is to record your expectations before you know the answers.

## 3. Auditing the file (12 points)

Automatically collected catalogs contain defects: values that were never
measured, rows that appear more than once, and fields that mean something other
than what their name suggests. Find the defects in this file that would mislead
someone who trains a model on it.

1. (8 points) Report at most eight findings, ordered by how much each would
   distort a model trained on the file. For each finding, give the evidence, in
   the form of one number or one figure; the mechanism you believe produced it;
   the action you take; and what that action costs you.
2. (2 points) Explain why your first finding outranks your second.
3. (2 points) Carry out the decisions you took, as code in your notebook that
   turns the raw table into the table the rest of your analysis uses. Findings
   you decided not to act on belong in the report as stated limitations of your
   analysis, rather than in the code.

An unordered list of every unusual value in the file, with no mechanism and no
ranking, earns partial credit at best.

## 4. The geometry of the feature space (10 points)

The numeric audio columns define a feature space. Describe its shape.

1. (3 points) How many dimensions does this space effectively have? State the
   criterion you applied and the evidence for it.
2. (3 points) Interpret the leading principal components. Name each one in plain
   language and give the loadings that support the name.
3. (3 points) Report what happens when the features are not standardized before
   PCA. Account for the result in terms of the quantity that PCA maximizes,
   rather than in terms of intuition: a short derivation, or a variance
   calculation on your own numbers, is expected here.
4. (1 point) Identify the columns in [Section 1](#1-the-data) that should not
   enter this analysis as numeric variables, and state what you did with them.

## 5. Structure in the catalog (10 points)

The team believes that the catalog falls into natural kinds of music. Test that
belief at two levels of aggregation.

1. (4 points) **Genres.** Represent each genre by the profile of its tracks and
   group the genres by hierarchical clustering. Compare two dissimilarity
   measures, or two linkage rules. Report where the two groupings disagree, and
   state which of them you would defend.
2. (3 points) **Tracks.** Cluster individual tracks. Justify the number of
   clusters with at least two criteria, and describe in plain language what each
   cluster contains. Not every method scales to 114,000 rows; if you cluster a
   sample, or evaluate on one, say which sample and why.
3. (3 points) Answer the following objection with evidence: *these clusters are
   not musical structure; they are an artifact of the way the file was
   assembled*. A comparison against the genre labels is one form of evidence,
   and not the only one.

## 6. Two regression targets (10 points)

The team wants to know which of the following questions the audio features can
answer: how energetic a track is, and how popular a track will be. Fit a linear
model for each.

1. (2 points) Restate your predictions from [Section
   2](#2-pre-registration-4-points), then report the held-out root-mean-square
   error for both targets, together with the error of a model that ignores the
   features and always predicts the training mean.
2. (3 points) Account for the difference between the two targets. Explain what
   makes the harder one hard, using evidence from this data rather than a
   general statement about noise.
3. (3 points) If the audio features do not explain the harder target, look for a
   variable in the file that does. Identify a candidate and support it with a
   measurement.
4. (2 points) Fit the model again with regularization, on a set of features that
   you have reason to believe are related to one another. Vary the penalty over
   several orders of magnitude and report what happens to the coefficients and to
   the held-out error. If the penalty does not improve the error, say so and
   explain why not, then find a setting in which it does: a smaller training
   sample, a richer set of basis functions, or both.

Close this section with one sentence of the following form: the audio features
support X, and do not support Y.

## 7. Conclusions (4 points)

Close the report with a summary that a reader could act on without reading the
rest of it. At most one page.

1. (1 point) State what your cleaning keeps and what it discards.
2. (1 point) Give three claims about this catalog that your analysis supports,
   each with the evidence behind it and the observation that would refute it.
3. (1 point) Name the property of this file that would make an evaluation based
   on a random split of rows look better than the model deserves, and give the
   evidence for that property from your audit.
4. (1 point) Fit a logistic regression for a two-class question of your choice,
   using a training and test split. Choose a question whose two classes are of
   comparable size, so that the comparison below carries information. Report the
   fraction of test cases it classifies correctly, next to the fraction obtained
   by always predicting the more common class, and state what the comparison
   tells you about the catalog.

State which of your claims from [Section 2](#2-pre-registration-4-points) did
not survive the analysis, and what changed your mind.

## 8. AI-use log

Submit at most one page recording which parts of the work you delegated to an AI
tool, how you checked its output, and at least one case in which it produced
something wrong or misleading, including how you noticed. The log carries no
points of its own. It is the basis of one of the demo questions, so it must
describe what you actually did.

## 9. Demo (50 points)

The demo takes place in the week of **November 10**, in a session that also
covers the next assignment. The schedule and the length of each slot are
announced on eeclass. Bring your report and a notebook you can run. Questions
are drawn from a bank and refer to your own submission. Typical questions are:

- Take one decision from your audit. What changes further down the analysis if
  you decide the other way? Show the effect in your notebook.
- Your report interprets the first principal component. Suppose one column were
  multiplied by 1000. What happens to that interpretation, and why?
- You reported two regression results. Which of them would you rely on, and what
  evidence would change your mind?
- Reproduce the error recorded in your AI-use log.
- A TA changes one element of your notebook, such as a random seed, a scaling
  step, or a filter. State what you expect to happen and why, then run it.

Points are awarded for explaining, defending, and modifying your own work, and
not for the work being free of mistakes.

## 10. Grading

**Report (50 points)**

| Section | Points |
|:--|--:|
| 2 · Pre-registration | 4 |
| 3 · Auditing the file | 12 |
| 4 · Geometry of the feature space | 10 |
| 5 · Structure in the catalog | 10 |
| 6 · Two regression targets | 10 |
| 7 · Conclusions | 4 |

**Demo (50 points)**

| Criterion | Points |
|:--|--:|
| Explains and defends own decisions | 15 |
| Predicts the effect of a change, with reasons | 15 |
| Argues from definitions when questioned | 10 |
| Accounts for the AI tools used | 10 |

**Open-ended answers are marked in four bands.**

| Band | Description | Credit |
|:--|:--|--:|
| 1 | A result or a number is reported. | 25% |
| 2 | Several results are reported without a connection between them. | 50% |
| 3 | Evidence, mechanism, and decision are connected into one argument. | 85% |
| 4 | The argument is also tested against an alternative explanation, or its limits are stated. | 100% |

Correct observations that are neither ordered nor explained fall in band 2,
regardless of how many a report contains.

## 11. Reading

The lecture slides are the primary material for every section of this
assignment. The textbook sections below cover the same topics in more depth and
are optional; read the named sections only.

| Section | Lecture slides | Kubat, *An Introduction to Machine Learning* | Bishop, *Pattern Recognition and Machine Learning* |
|:--|:--|:--|:--|
| 3 · Audit | L1, L12 | §1.3 Problems with Available Data, p. 6: attribute types, irrelevant and redundant attributes, missing values, inconsistent data, and noise. §11.5 Unknown Attribute Values, p. 222. | — |
| 4 · Geometry | L12 | §11.6 Attribute Selection, p. 224, on selection against extraction. Kubat does not cover PCA. | §12.1.1 and §12.1.2, p. 561–565, the two formulations of PCA used in the slides. §1.4 The Curse of Dimensionality, p. 33. |
| 5 · Structure | L11 | §15.1 Cluster Analysis, p. 297. §15.2 k-Means, p. 301, including the normalization of attributes. §15.3, p. 305. §15.4 Hierarchical Aggregation, p. 307. | §9.1 K-means Clustering, p. 424. |
| 6 · Regression | L1, L2 | — | §1.1, p. 4–11, where held-out error is measured by *E*<sub>RMS</sub>, equation (1.3). §3.1.4 Regularized Least Squares, p. 144. |
| 7 · Conclusions | L2 | — | §4.3.2 Logistic Regression, p. 205. |

## 12. Data source

*Spotify Tracks Dataset*, distributed on Hugging Face as
`maharshipandya/spotify-tracks-dataset`, used here for coursework. Cite this
source in your report.
