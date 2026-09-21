---
layout: distill
title: "What Does a World Model Choose to Remember?"
date: 2026-09-21
published: false
tags: ["SSL", "Theory", "World Model"]
citation: false
related_posts: false
bibliography: 2026-09-21-world-models.bib

authors:
  - name: Hugues Van Assel
    url: "https://huguesva.github.io/"
---

<link rel="stylesheet" href="{{ '/assets/css/site.css' | relative_url }}">

LLMs have made substantial progress on text-based tasks. Extending this progress to the physical world calls for AI that can anticipate how its surroundings evolve and how its actions affect them. World models promise to provide this capability, and years of active research have produced a wide range of approaches. In this note, we study several of these approaches from first principles, using simple mathematical models to understand their tradeoffs and blind spots.

## Five approaches to learning a world model

We will compare five approaches: VAE world models<d-cite key="ha2018worldmodels"></d-cite>, action-conditioned observation prediction<d-cite key="oh2015action"></d-cite>, Embed to Control (E2C)<d-cite key="watter2015e2c"></d-cite>, Dreamer<d-cite key="hafner2020dreamer"></d-cite>, and joint-embedding predictive architectures (JEPAs), with LeWorldModel as a recent example<d-cite key="maes2026leworldmodel"></d-cite>. To make the comparison explicit, we use linear models with a capacity of $k$ features and one-step predictions conditioned on the action. We leave out recurrent memory and reward-learning losses. For the variational methods, we retain only the mean-dependent terms of Gaussian objectives with fixed covariance weights, which gives squared reconstruction and prediction errors. These simplified objectives let us study what each approach encourages the representation to retain; they do not describe the full agents.

### VAE world models: reconstruct first, predict later

In the World Models pipeline<d-cite key="ha2018worldmodels"></d-cite>, a VAE first learns to compress individual observations. Its encoder is then fixed, and a separate model learns the latent dynamics. To understand what this first stage retains, let $\mathbf X_t\in\mathbb R^d$ be a centered observation with covariance $\boldsymbol\Sigma\succ0$. Our linear encoder is $\mathbf Z_t=\mathbf E\mathbf X_t\in\mathbb R^r$, where $0\leq r\leq k\leq d$, and $\mathbf D\mathbf Z_t$ reconstructs the observation. We normalize the active latent coordinates so that $\mathbf E\boldsymbol\Sigma\mathbf E^\top=\mathbf I_r$. With decoder variance $\sigma_{\mathrm{dec}}^2>0$ and regularization weight $\beta>0$, the simplified objective is

$$
\begin{aligned}
\mathcal L_{\mathrm{VAE}}
&=\frac{1}{2\sigma_{\mathrm{dec}}^2}
\mathbb E\!\left[\|\mathbf D\mathbf E\mathbf X_t-\mathbf X_t\|_2^2\right]\\
&\quad+\frac{\beta}{2}\mathbb E\!\left[\|\mathbf E\mathbf X_t\|_2^2\right],\\
&\mathbf E\boldsymbol\Sigma\mathbf E^\top=\mathbf I_r,
\qquad 0\leq r\leq k.
\end{aligned}
$$

The first term rewards reconstruction. The second is the encoder-mean contribution of the VAE's KL penalty; under our normalization, it costs $\beta/2$ per active feature. The normalization is part of the simplified model: it prevents the encoder from shrinking its outputs while the decoder rescales them. We are studying these normalized means, not optimizing the full stochastic VAE objective.

Write $\boldsymbol\Sigma\boldsymbol\phi_j=p_j\boldsymbol\phi_j$, where the $\boldsymbol\phi_j$ form an orthonormal basis and $p_j$ is the variance along direction $j$. The solution keeps up to $k$ directions with the largest positive scores

$$
\boxed{h_j^{\mathrm{VAE}}=\frac{p_j}{\sigma_{\mathrm{dec}}^2}-\beta.}
$$

A direction improves the objective exactly when $p_j>\beta\sigma_{\mathrm{dec}}^2$. For each retained direction, we can choose the encoder row $\boldsymbol\phi_j^\top/\sqrt{p_j}$ and the corresponding decoder column $\sqrt{p_j}\boldsymbol\phi_j$. Ties can be resolved arbitrarily, and zero-score directions can be omitted. In this reduction, selection depends on variance alone: neither temporal predictability nor the effect of actions enters the score.

<details markdown="1">
<summary>Proof: which directions does the VAE retain?</summary>

**Optimal decoder.** Fix an admissible encoder $\mathbf E$. Expanding the reconstruction error and using $\mathbf E\boldsymbol\Sigma\mathbf E^\top=\mathbf I_r$ gives

$$
\begin{aligned}
\mathbb E\!\left[\|\mathbf D\mathbf E\mathbf X_t-\mathbf X_t\|_2^2\right]
={}&\operatorname{tr}(\boldsymbol\Sigma)\\
&-2\operatorname{tr}(\mathbf D\mathbf E\boldsymbol\Sigma)
+\operatorname{tr}(\mathbf D\mathbf D^\top).
\end{aligned}
$$

Differentiating with respect to $\mathbf D$ gives $\mathbf D^\star=\boldsymbol\Sigma\mathbf E^\top$. Substituting this decoder yields

$$
\min_{\mathbf D}\mathcal L_{\mathrm{VAE}}
=\frac{\operatorname{tr}(\boldsymbol\Sigma)
-\operatorname{tr}(\mathbf E\boldsymbol\Sigma^2\mathbf E^\top)}
{2\sigma_{\mathrm{dec}}^2}
+\frac{\beta r}{2}.
$$

**Encoder directions.** Set $\mathbf W=\mathbf E\boldsymbol\Sigma^{1/2}$. The normalization becomes $\mathbf W\mathbf W^\top=\mathbf I_r$, and the trace being maximized is $\operatorname{tr}(\mathbf W\boldsymbol\Sigma\mathbf W^\top)$. To see why the leading eigenvectors are optimal, write

$$
\begin{aligned}
\operatorname{tr}(\mathbf W\boldsymbol\Sigma\mathbf W^\top)
&=\sum_{j=1}^d p_j\alpha_j,\\
\alpha_j:=\|\mathbf W\boldsymbol\phi_j\|_2^2,
\qquad 0\leq\alpha_j\leq1,
&\qquad \sum_{j=1}^d\alpha_j=r.
\end{aligned}
$$

For fixed $r$, this sum is bounded by the sum of the $r$ largest eigenvalues. Taking the rows of $\mathbf W$ to be their eigenvectors attains that bound.

**Number of retained features.** For an eigenvector set $S$ with $\lvert S\rvert\leq k$, the resulting loss is

$$
\mathcal L_{\mathrm{VAE}}^\star(S)
=\frac{\operatorname{tr}(\boldsymbol\Sigma)}{2\sigma_{\mathrm{dec}}^2}
-\frac12\sum_{j\in S}
\left(\frac{p_j}{\sigma_{\mathrm{dec}}^2}-\beta\right).
$$

Adding a direction lowers the loss only if its score is positive. We therefore retain the largest positive scores, up to capacity $k$. The empty set gives the zero representation when no score is positive.

</details>
