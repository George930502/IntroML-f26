---
layout: page
title: Assignments
nav_order: 4
has_children: true
has_toc: false
description: Lab assignments.
---

# Assignments {:.no_toc}

There are 4 labs, worth 40% of your grade in total.

- Each lab is due 3 weeks after its announcement.
- Google Colab is recommended.
- The use of AI tools is welcome.
- A demo with the TAs is required for each lab.
- Avoid plagiarism. Cheating or plagiarism on labs will result in a zero score.

## Table of Contents {: .no_toc .text-delta }

<ul class="nav-list">
{% assign hws = site.pages | where: 'parent', 'Assignments' | sort: 'nav_order' %}
{% for hw in hws %}
<li class="nav-list-item"><a href="{{ hw.url | relative_url }}" class="nav-list-link">{{ hw.title }}</a></li>
{% endfor %}
</ul>
