---
layout: distill
title: "What Does a World Model Choose to Remember?"
date: 2026-09-21
published: false
tags: ["SSL", "Theory", "World Model"]
citation: false
related_posts: false

authors:
  - name: Hugues Van Assel
    url: "https://huguesva.github.io/"
---

<link rel="stylesheet" href="{{ '/assets/css/site.css' | relative_url }}">

{% comment %}
Working title; draft one paragraph at a time for review.
Keep the writing plain and direct, in the style of the existing research notes.
Use a capacity of k features. Put derivations in collapsible proof blocks.
Local preview: bundle exec jekyll serve --unpublished
{% endcomment %}

World models come in many forms, and joint-embedding predictive architectures (JEPAs) have recently attracted a lot of attention. In these notes, we use simple mathematical models to connect JEPA to other families of world models. The goal is to understand the tradeoffs of each approach: what it learns, what it leaves out, and where it can fail.
