---
layout: page
title: Assignments
nav_order: 4
has_children: true
has_toc: false
description: Lab assignments.
---

# Assignments
{:.no_toc}

There are 4 labs, worth 40% of your grade in total.

- Google Colab is recommended.
- The use of AI tools is welcome.
- A demo with the TAs is required for each lab.
- **Avoid plagiarism. Cheating or plagiarism on labs will result in a zero score.**

## Table of Contents
{: .no_toc .text-delta }

{% assign hws = site.pages | where: 'parent', 'Assignments' | sort: 'nav_order' %}
{% for hw in hws %}
- [{{ hw.title }}]({{ hw.url | relative_url }})
{% endfor %}
