---
layout: page
permalink: /talks/
title: talks
description: Invited talks, conference presentations, and seminars on representation learning, optimal transport, and dimensionality reduction.
nav: true
nav_order: 1
---

<style>
.talk-card {
  display: flex;
  margin-bottom: 2em;
  padding: 1em;
  border-radius: 8px;
  background: #fafafa;
  border: 1px solid #eee;
}
.talk-thumbnail {
  flex: 0 0 180px;
  margin-right: 1.5em;
}
.talk-thumbnail img {
  width: 100%;
  border-radius: 4px;
  box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.talk-content {
  flex: 1;
}
.talk-title {
  font-weight: 600;
  font-size: 1.1em;
  margin-bottom: 0.5em;
  color: #333;
}
.talk-buttons {
  margin: 0.5em 0;
}
.talk-buttons a {
  display: inline-block;
  padding: 0.25em 0.75em;
  margin-right: 0.5em;
  margin-bottom: 0.3em;
  font-size: 0.85em;
  border-radius: 4px;
  text-decoration: none;
  background: var(--global-theme-color, #2698BA);
  color: white !important;
  transition: opacity 0.2s;
}
.talk-buttons a:hover {
  opacity: 0.8;
}
.talk-venues {
  font-size: 0.9em;
  color: #666;
  margin-top: 0.5em;
}
.talk-venues ul {
  margin: 0.3em 0 0 1.2em;
  padding: 0;
}
.talk-venues li {
  margin-bottom: 0.2em;
}
.year-header {
  font-size: 1.5em;
  font-weight: 600;
  margin: 1.5em 0 1em 0;
  padding-bottom: 0.3em;
  border-bottom: 2px solid var(--global-theme-color, #2698BA);
}
.section-header {
  font-size: 1.2em;
  font-weight: 600;
  margin: 2em 0 1em 0;
  color: #666;
}
@media (max-width: 600px) {
  .talk-card { flex-direction: column; }
  .talk-thumbnail { margin-right: 0; margin-bottom: 1em; flex: none; width: 100%; max-width: 200px; }
}
</style>

<div class="year-header">2025</div>

<div class="talk-card">
  <div class="talk-thumbnail">
    <img src="/assets/img/talks/je_vs_rc_thumb.png" alt="JE vs RC talk" onerror="this.src='/assets/img/talks/default_thumb.png'">
  </div>
  <div class="talk-content">
    <div class="talk-title">Joint Embedding vs Reconstruction: Which Paradigm to Choose for Pretraining?</div>
    <div class="talk-buttons">
      <a href="/assets/pdf/slides/JE_vs_RC.pdf">Slides</a>
      <a href="/assets/pdf/posters/JE_vs_RC.pdf">Poster</a>
    </div>
    <div class="talk-venues">
      <ul>
        <li>12/2025: NeurIPS 2025, San Diego</li>
        <li>11/2025: Prescient Design Machine Learning seminar, South San Francisco</li>
      </ul>
    </div>
  </div>
</div>

<div class="year-header">2024</div>

<div class="talk-card">
  <div class="talk-thumbnail">
    <img src="/assets/img/talks/msk_thumb.png" alt="DR with OT talk" onerror="this.src='/assets/img/talks/default_thumb.png'">
  </div>
  <div class="talk-content">
    <div class="talk-title">Revisiting Dimensionality Reduction with Optimal Transport: Within and Across Spaces</div>
    <div class="talk-buttons">
      <a href="/assets/pdf/slides/msk_pres.pdf">Slides</a>
    </div>
    <div class="talk-venues">
      <ul>
        <li>11/2024: Genentech, South San Francisco</li>
        <li>10/2024: Memorial Sloan Kettering Cancer Center, New York</li>
      </ul>
    </div>
  </div>
</div>

<div class="talk-card">
  <div class="talk-thumbnail">
    <img src="/assets/img/talks/distR_thumb.png" alt="GW DR talk" onerror="this.src='/assets/img/talks/default_thumb.png'">
  </div>
  <div class="talk-content">
    <div class="talk-title">Unifying Dimensionality Reduction and Clustering with Gromov-Wasserstein</div>
    <div class="talk-buttons">
      <a href="/assets/pdf/slides/distR_pres.pdf">Slides</a>
    </div>
    <div class="talk-venues">
      <ul>
        <li>06/2024: Cambridge University Machine Learning group seminar</li>
        <li>05/2024: CANUM 2024, Île de Ré</li>
        <li>04/2024: LaMME seminar, Univ. Paris-Saclay</li>
        <li>04/2024: Inria Mind team seminar</li>
      </ul>
    </div>
  </div>
</div>

<div class="year-header">2023</div>

<div class="talk-card">
  <div class="talk-thumbnail">
    <img src="/assets/img/talks/snekhorn_thumb.png" alt="SNEkhorn talk" onerror="this.src='/assets/img/talks/default_thumb.png'">
  </div>
  <div class="talk-content">
    <div class="talk-title">Affinity Matrices with Symmetric Optimal Transport</div>
    <div class="talk-buttons">
      <a href="/assets/pdf/slides/snekhorn_long.pdf">Slides</a>
      <a href="/assets/pdf/posters/SNEkhorn.pdf">Poster</a>
    </div>
    <div class="talk-venues">
      <ul>
        <li>12/2023: NeurIPS 2023, New Orleans</li>
        <li>11/2023: RT MIA thematic day on Dimensionality Reduction, Lyon</li>
        <li>11/2023: PariSanté campus, Paris</li>
        <li>06/2023: GeMSS 2023, Copenhagen</li>
      </ul>
    </div>
  </div>
</div>

<div class="year-header">2022</div>

<div class="talk-card">
  <div class="talk-thumbnail">
    <img src="/assets/img/talks/gc_thumb.png" alt="Graph Coupling talk" onerror="this.src='/assets/img/talks/default_thumb.png'">
  </div>
  <div class="talk-content">
    <div class="talk-title">Graph Coupling for Dimensionality Reduction</div>
    <div class="talk-buttons">
      <a href="/assets/pdf/slides/mlsp_GC.pdf">Slides</a>
      <a href="/assets/pdf/posters/Poster_Neurips2022.pdf">Poster</a>
    </div>
    <div class="talk-venues">
      <ul>
        <li>11/2022: NeurIPS 2022, New Orleans</li>
        <li>10/2022: Workshop "Statistical challenges in scRNA-seq data analysis", Toulouse</li>
        <li>05/2022: EPIT workshop, CIRM, Marseille</li>
        <li>04/2022: Statlearn workshop, Cargèse</li>
        <li>03/2022: MLSP seminar, ENS Lyon</li>
      </ul>
    </div>
  </div>
</div>

<div class="section-header">Other Talks</div>

<div class="talk-card">
  <div class="talk-thumbnail">
    <img src="/assets/img/talks/default_thumb.png" alt="Other talks" onerror="this.style.display='none'">
  </div>
  <div class="talk-content">
    <div class="talk-title">Miscellaneous & Popularization Talks</div>
    <div class="talk-buttons">
      <a href="/assets/pdf/slides/HD_GGM_network_pres.pdf">HD Graphical Models</a>
      <a href="/assets/pdf/slides/KL_presentation.pdf">KL Divergence</a>
      <a href="/assets/pdf/slides/pres_convexity.pdf">Convexity</a>
      <a href="/assets/pdf/slides/HD_data.pdf">High-Dimensional Data</a>
    </div>
    <div class="talk-venues">
      <ul>
        <li>05/2021: SingleStatOmics meeting - HD Gaussian graphical models (Li et al.)</li>
        <li>Popularization: Kullback-Leibler divergence, Convexity, Curses of Dimensionality</li>
      </ul>
    </div>
  </div>
</div>
