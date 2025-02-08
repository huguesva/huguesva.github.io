---
layout: distill
title: Inverse optimal transport does not require unrolling
description: A note on the equivalence between inverse OT and minimizing the Monge gap.
tags: ['OT', 'DR']
giscus_comments: false
date: 2024-02-25
featured: false
citation: true

authors:
  - name: Hugues Van Assel
    url: "https://huguesva.github.io/"
    affiliations:
      name: Ecole Normale Superieure de Lyon

bibliography: 2024-02-25-distill.bib

---

<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

This blog is about an elegant and practical reformulation of inverse Optimal Transport (OT) that enables efficient computations. It is based on a derivation found in <d-cite key="ma2020learning"></d-cite>. In the last part, we apply this trick to efficiently learn low dimensional data representations.

### Background on (Entropic) Optimal Transport

Entropic OT <d-cite key="peyre2019computational"></d-cite> is a powerful tool with many applications in machine learning, including generative modelling <d-cite key="genevay2018learning"></d-cite>, domain adaptation <d-cite key="courty2017joint"></d-cite> and dimensionality reduction <d-cite key="van2024snekhorn"></d-cite>.

We consider two discrete distributions that we wish to compare: $$\sum_i a_i \delta_{\mathbf{x}_i}$$ <d-footnote> $\delta_{\mathbf{x}}$ is a dirac distribution with a unit mass in position $\mathbf{x}$ and $0$ elsewhere. </d-footnote> and $$\sum_j b_j \delta_{\mathbf{y}_j}$$ where $$\mathbf{a}$$ $$= (a_1,...,a_p)$$ and $$\mathbf{b}$$ $$= (b_1,...,b_m)$$ are vectors with positive entries  in the probability simplex (*ie*  such that $$\sum_i a_i = \sum_j b_j =1$$).
We also consider a cost matrix $$\mathbf{C}$$ with entries $$C_{ij} = d(\mathbf{x}_i, \mathbf{y}_j)$$ where $$d$$ is a dissimilarity function.

**Primal problem.** The entropic OT problem reads <d-footnote> $\langle \mathbf{C}, \mathbf{P} \rangle = \sum_{ij} C_{ij} P_{ij}$ denotes the Euclidean inner product. </d-footnote>

$$
\DeclareMathOperator*{\argmin}{arg\,min}
\begin{align}\label{eq:eot}
\min_{\mathbf{P} \in \Pi(\mathbf{a}, \mathbf{b})} \: \: \langle \mathbf{C}, \mathbf{P} \rangle - \varepsilon \mathrm{H}(\mathbf{P})
\end{align}
$$

where $$\Pi(\mathbf{a}, \mathbf{b})=\left\{\mathbf{P}\mathbf{\geq0},\mathbf{P}\mathbf{1}=\mathbf{a},\mathbf{P}^{\top}\mathbf{1}=\mathbf{b}\right\}$$ is the set of couplings with marginals $$(\mathbf{a}, \mathbf{b})$$ and $$\mathrm{H}(\mathbf{P}) = - \langle \mathbf{P}, \log \mathbf{P} - \mathbf{1} \mathbf{1}^\top \rangle$$ <d-footnote> $\mathbf{1}$ is the vector of ones $(1,...,1)$. </d-footnote>.
$$\varepsilon > 0$$ is a regularizer that sets the entropy of the transport plan.

**Dual problem.** The above entropic OT problem \eqref{eq:eot} can be solved through the following dual

$$
\begin{align}\label{eq:dual_eot}
    \max_{\mathbf{f},\mathbf{g}} \: \: \langle \mathbf{f}, \mathbf{a} \rangle + \langle \mathbf{g}, \mathbf{b} \rangle - \varepsilon \left\langle \exp((\mathbf{f} \oplus \mathbf{g} - \mathbf{C}) / \varepsilon), \mathbf{1} \mathbf{1}^\top \right\rangle \:.
\end{align}
$$

The solution $$\mathbf{P}^\star$$ of the primal problem \eqref{eq:eot} can be expressed in terms of the optimal dual variables $$(\mathbf{f}^\star, \mathbf{g}^\star)$$ solving \eqref{eq:dual_eot} as
$$\mathbf{P}^{\star} = \exp((\mathbf{f}^\star \oplus \mathbf{g}^\star - \mathbf{C}) / \varepsilon)$$.

{% details proof %}
The Lagrangian of the above problem is as follows, with dual variables $$\mathbf{f}$$ and $$\mathbf{g}$$
$$
\begin{align}\label{eq:lagrangian_eot}
    \langle \mathbf{C}, \mathbf{P} \rangle - \varepsilon \mathrm{H}(\mathbf{P}) - \langle \mathbf{f}, \mathbf{P} \mathbf{1} - \mathbf{a} \rangle - \langle \mathbf{g}, \mathbf{P}^\top \mathbf{1} - \mathbf{b} \rangle \:.
\end{align}
$$
Strong duality holds for \eqref{eq:eot} and the first order KKT condition gives
$$
\begin{align}
    \mathbf{C} - \varepsilon \log(\mathbf{P}^\star) - \mathbf{f}^\star\mathbf{1}^\top - \mathbf{1}(\mathbf{g}^\star)^{\top} \mathbf{=0}
\end{align}
$$
for optimal primal $$\mathbf{P}^\star$$ and dual $$(\mathbf{f}^\star, \mathbf{g}^\star)$$ variables.

It gives the primal/dual relation $$\mathbf{P}^\star = \exp((\mathbf{f}^\star \oplus \mathbf{g}^\star - \mathbf{C}) / \varepsilon)$$.

Plugging it back into the Lagrangian we recover the dual objective of equation \eqref{eq:dual_eot}.
{% enddetails %}

Problem \eqref{eq:dual_eot} can be solved using block coordinate ascent, alternatively optimizing with respect to $$\mathbf{f}$$ and $$\mathbf{g}$$ with the following updates:
$$
\begin{align}
  f_i &\leftarrow \varepsilon \log a_i - \varepsilon \log \sum_j e^{(g_j-C_{ij}) / \varepsilon} \label{eq:sinkhorn-f} \\
  g_j &\leftarrow \varepsilon \log b_j - \varepsilon \log \sum_i e^{(f_i-C_{ij}) / \varepsilon} \label{eq:sinkhorn-g} \:.
\end{align}
$$
:bulb: The above updates are known as Sinkhorn iterations (in log domain) due to the seminal work of Sinkhorn and Knopp who proved their convergence <d-cite key="sinkhorn1967concerning"></d-cite>.

### Inverse Optimal Transport :arrow_right_hook:

In inverse OT <d-cite key="ma2020learning"></d-cite>, from an OT plan $$\widehat{\mathbf{P}} \in \Pi(\mathbf{a}, \mathbf{b})$$, one seeks to reconstruct a cost $$\mathbf{C}$$ likely to have generated $$\widehat{\mathbf{P}}$$ when solving OT on $$\mathbf{C}$$.
We will see some applications in what follows.

When using entropic OT, the inverse OT problem is usually formulated with a KL divergence
$$\mathrm{KL}(\mathbf{P} \| \mathbf{Q}) = \langle \mathbf{P}, \log (\mathbf{P} \oslash \mathbf{Q}) \rangle - \mathbf{P} + \mathbf{Q}$$.
The problem we consider is as follows

$$
\DeclareMathOperator*{\argmin}{arg\,min}
\begin{align}
\min_{\mathbf{C}} \quad &\mathrm{KL}(\widehat{\mathbf{P}} \| \mathbf{P}^{\mathbf{C}}) \label{eq:outer_invot}\\[1em]
\text{s.t.} \quad &\mathbf{P}^{\mathbf{C}} = \argmin_{\mathbf{P} \in \Pi(\mathbf{a}, \mathbf{b})} \: \: \langle \mathbf{C}, \mathbf{P} \rangle - \varepsilon \mathrm{H}(\mathbf{P}) \label{eq:inner_invot} \:.
\end{align}
$$

**Issue** : the above is a nested problem and we need to unroll the Sinkhorn iterations of the inner problem \eqref{eq:inner_invot} to solve the outer problem \eqref{eq:outer_invot}. Another approach would be to rely on the implicit function theorem but it requires a costly inversion.

Hopefully, a computationally simpler formulation can be derived from the above, as shown in the theorem 1 of <d-cite key="ma2020learning"></d-cite>.
Indeed, problem \eqref{eq:outer_invot} is equivalent to the following single-level problem

$$
\begin{align}
    \min_{\mathbf{C}, \mathbf{f}, \mathbf{g}} \: \: \left\langle \widehat{\mathbf{P}}, \mathbf{C} \right\rangle - \langle \mathbf{f}, \mathbf{a} \rangle - \langle \mathbf{g}, \mathbf{b} \rangle + \varepsilon \left\langle \exp(\left(\mathbf{f} \oplus \mathbf{g} - \mathbf{C}\right) / \varepsilon), \mathbf{1} \mathbf{1}^\top \right\rangle \:.
\end{align}
$$

We detail this derivation in what follows.

### Simplification of inverse OT :rocket:

A first step is to observe that the outer objective \eqref{eq:outer_invot} of inverse OT can be expressed in terms of the optimal dual variables $$(\mathbf{f}^\star,\mathbf{g}^\star)$$ of the entropic OT inner problem \eqref{eq:inner_invot}. Indeed, it holds

$$
\begin{align}\label{eq:first_step}
    \varepsilon \left( \mathrm{KL}(\widehat{\mathbf{P}} \| \mathbf{P}^{\mathbf{C}}) + \operatorname{H}(\widehat{\mathbf{P}}) \right)  &= \left\langle \widehat{\mathbf{P}}, \mathbf{C} \right\rangle - \langle \mathbf{f}^\star, \mathbf{a} \rangle - \langle \mathbf{g}^\star, \mathbf{b} \rangle \:.
\end{align}
$$

{% details proof %}
The KL can be decomposed as
$$
\begin{align}
\operatorname{KL}(\widehat{\mathbf{P}} | \mathbf{P}^{\mathbf{C}}) = - \langle \widehat{\mathbf{P}}, \log \mathbf{P}^{\mathbf{C}} \rangle - \operatorname{H}(\widehat{\mathbf{P}}) \:.
\end{align}
$$

For optimal dual variables $$(\mathbf{f}^\star, \mathbf{g}^\star)$$, the solution of the primal of entropic OT is given by
$$
\begin{align}
    \mathbf{P}^{\mathbf{C}} = \exp((\mathbf{f}^\star \oplus \mathbf{g}^\star - \mathbf{C}) / \varepsilon) \:.
\end{align}
$$

Therefore we have
$$
\begin{align}
    \varepsilon \left( \mathrm{KL}(\widehat{\mathbf{P}} \| \mathbf{P}^{\mathbf{C}}) + \operatorname{H}(\widehat{\mathbf{P}}) \right) &= - \left\langle \widehat{\mathbf{P}}, \mathbf{f}^\star \oplus \mathbf{g}^\star - \mathbf{C} \right\rangle \\
    &= \left\langle \widehat{\mathbf{P}}, \mathbf{C} \right\rangle - \left\langle \widehat{\mathbf{P}}, \mathbf{f}^\star \oplus \mathbf{g}^\star \right\rangle \:.
\end{align}
$$

Focusing on the last term, using that $$\widehat{\mathbf{P}} \in \Pi(\mathbf{a}, \mathbf{b})$$ it holds
$$
\begin{align}
    \left\langle \widehat{\mathbf{P}}, \mathbf{f}^\star \oplus \mathbf{g}^\star \right\rangle &= \sum_i f^\star_i \sum_j \widehat{P}_{ij} + \sum_j g^\star_j \sum_i \widehat{P}_{ij} \\
    &= \sum_i f^\star_i a_i + \sum_j g^\star_j b_j \\
    &= \langle \mathbf{f}^\star, \mathbf{a} \rangle + \langle \mathbf{g}^\star, \mathbf{b} \rangle \:.
\end{align}
$$
Therefore
$$
\begin{align}
    \varepsilon \left( \mathrm{KL}(\widehat{\mathbf{P}} \| \mathbf{P}^{\mathbf{C}}) + \operatorname{H}(\widehat{\mathbf{P}}) \right)  &= \left\langle \widehat{\mathbf{P}}, \mathbf{C} \right\rangle - \langle \mathbf{f}^\star, \mathbf{a} \rangle - \langle \mathbf{g}^\star, \mathbf{b} \rangle \:.
\end{align}
$$
{% enddetails %}

In equation \eqref{eq:first_step}, $$\mathbf{f}^\star$$ and $$\mathbf{g}^\star$$ implicitly depend on $$\mathbf{C}$$ through problem \eqref{eq:dual_eot}. Thus we are still stuck with the bilevel structure and have'nt made any real progress yet.

Recall that we would like to derive a joint single-level objective for both outer variable $\mathbf{C}$ and inner variables $(\mathbf{f}, \mathbf{g})$. To do so, one can notice that equation \eqref{eq:first_step} has terms in common with the dual problem of entropic OT \eqref{eq:dual_eot}. Indeed, in both \eqref{eq:dual_eot} and \eqref{eq:first_step} we find
$$
\begin{align}
\langle\mathbf{f},\mathbf{a}\rangle+\langle\mathbf{g},\mathbf{b}\rangle \:.
\end{align}
$$

The **trick** is to add the missing term of dual entropic OT \eqref{eq:dual_eot} in \eqref{eq:first_step}.
Doing so, we define the following joint objective
$$
\begin{align}
    \cal{G}(\widehat{\mathbf{P}}, \mathbf{C}, \mathbf{f}, \mathbf{g}) = &\left\langle \widehat{\mathbf{P}}, \mathbf{C} \right\rangle - \langle \mathbf{f}, \mathbf{a} \rangle - \langle \mathbf{g}, \mathbf{b} \rangle \\
    + &\varepsilon \left\langle \exp(\left(\mathbf{f} \oplus \mathbf{g} - \mathbf{C}\right) / \varepsilon), \mathbf{1} \mathbf{1}^\top \right\rangle \:.
\end{align}
$$

For any $$\mathbf{C}$$, minimizing $$\cal{G}$$ with respect to $$(\mathbf{f}, \mathbf{g})$$ exactly amounts to solving dual entropic OT \eqref{eq:dual_eot}, because $$\left\langle \widehat{\mathbf{P}}, \mathbf{C} \right\rangle$$ does not depend on $$(\mathbf{f}, \mathbf{g})$$. Hence we have:
$$
\begin{align}
    \min_{\mathbf{f},\mathbf{g}} \: \cal{G}(\widehat{\mathbf{P}}, \mathbf{C}, \mathbf{f}, \mathbf{g}) = \left\langle \widehat{\mathbf{P}}, \mathbf{C} \right\rangle - \langle \mathbf{f}^\star, \mathbf{a} \rangle - \langle \mathbf{g}^\star, \mathbf{b} \rangle + \varepsilon \left\langle \mathbf{P}^{\mathbf{C}}, \mathbf{1} \mathbf{1}^\top \right\rangle
\end{align}
$$

where $$\mathbf{P}^{\mathbf{C}} = \exp((\mathbf{f}^\star \oplus \mathbf{g}^\star - \mathbf{C}) / \varepsilon)$$ as we have seen in the first part.

Importantly, because we have $$\mathbf{P}^{\mathbf{C}} \in \Pi(\mathbf{a}, \mathbf{b})$$, we can notice that the term we added no longer depends on $$\mathbf{C}$$ when evaluted in $$(\mathbf{f}^\star,\mathbf{g}^\star)$$. Indeed
$$
\begin{align}
    \left\langle \mathbf{P}^{\mathbf{C}}, \mathbf{1} \mathbf{1}^\top \right\rangle = \sum_{ij} P^{\mathbf{C}}_{ij} = \sum_i a_i = 1 \:.
\end{align}
$$

Thus, when evaluated in $$(\mathbf{f}^\star,\mathbf{g}^\star)$$, thanks to equation \eqref{eq:first_step} the objective writes
$$
\begin{align}
    \min_{\mathbf{f},\mathbf{g}} \: \cal{G}(\widehat{\mathbf{P}}, \mathbf{C}, \mathbf{f}, \mathbf{g}) &= \left\langle \widehat{\mathbf{P}}, \mathbf{C} \right\rangle - \langle \mathbf{f}^\star, \mathbf{a} \rangle - \langle \mathbf{g}^\star, \mathbf{b} \rangle + \varepsilon \\
    &= \varepsilon \left( \mathrm{KL}(\widehat{\mathbf{P}} \| \mathbf{P}^{\mathbf{C}}) + \operatorname{H}(\widehat{\mathbf{P}}) + \textrm{1} \right) \label{eq:final_derivation} \:.
\end{align}
$$

Minimizing the above with respect to $$\mathbf{C}$$ then amounts to minimizing $$\mathrm{KL}(\widehat{\mathbf{P}}\|\mathbf{P}^{\mathbf{C}})$$ since it is the only term that depends on $$\mathbf{C}$$ in equation \eqref{eq:final_derivation}.

Therefore solving inverse OT is equivalent to the following jointly convex problem
$$
\begin{align}\label{eq:new_form_invot}
    \min_{\mathbf{C}, \mathbf{f}, \mathbf{g}} \: \: \cal{G}(\widehat{\mathbf{P}}, \mathbf{C}, \mathbf{f}, \mathbf{g}) \:.
\end{align}
$$

Concretely, this means that $$(\mathbf{C}^\star, \mathbf{f}^\star, \mathbf{g}^\star)$$ solves \eqref{eq:new_form_invot} if and only if $$\mathbf{C}^\star$$ solves inverse OT \eqref{eq:outer_invot} where $$\mathbf{P}^{\mathbf{C}} = \exp((\mathbf{f}^\star \oplus \mathbf{g}^\star - \mathbf{C}) / \varepsilon)$$ solves the inner problem \eqref{eq:inner_invot}.

### Parallel with Monge gap

Let's take a moment to decipher this new expression closely.

Since strong duality holds for entropic OT, one has the equality between the primal optimal objective and the dual optimal objective *ie*  \eqref{eq:eot} = \eqref{eq:dual_eot}.

Therefore we have
$$
\begin{align}\label{eq:min_formulation_invot}
    \min_{\mathbf{f},\mathbf{g}} \: \cal{G}(\widehat{\mathbf{P}}, \mathbf{C}, \mathbf{f}, \mathbf{g}) &= \left\langle \widehat{\mathbf{P}}, \mathbf{C} \right\rangle - \left(\min_{\mathbf{P} \in \Pi(\mathbf{a}, \mathbf{b})} \: \: \langle \mathbf{C}, \mathbf{P} \rangle - \varepsilon \mathrm{H}(\mathbf{P}) \right) \:.
\end{align}
$$

Hence $$\cal{G}$$ quantifies the difference in the transport cost when using $$\widehat{\mathbf{P}}$$ against the solution of the inner problem $$\mathbf{P}^{\mathbf{C}}$$. This quantity is known as the Monge gap <d-cite key="pmlr-v202-uscidda23a"></d-cite>.

:bulb: As discussed earlier, optimizing *w.r.t.* $$\mathbf{C}$$ an argmin like in \eqref{eq:inner_invot} requires computationally demanding tools such as unrolling or implicit function theorem. On the contrary, optimizing the min as in \eqref{eq:min_formulation_invot} is much simpler. It can be done using Danskin’s theorem (or envelope theorem).

In our case, this result simply states that, for each update of $$\mathbf{C}$$, we can optimize $$\cal{G}$$ in $$\mathbf{C}$$ by considering $$\mathbf{f}$$ and $$\mathbf{g}$$ as constants. Without further constraint on $$\mathbf{C}$$, the update reads
$$
\begin{align}\label{eq:update_C}
  \mathbf{C} &\leftarrow \mathbf{f} \oplus \mathbf{g} - \varepsilon \log \widehat{\mathbf{P}} \:.
\end{align}
$$

Overall, to efficiently solve inverse OT one can use block coordinate descent  alternating between updating $$\mathbf{f}$$ and $$\mathbf{g}$$ with Sinkhorn iterations \eqref{eq:sinkhorn-f}-\eqref{eq:sinkhorn-g} and updating $$\mathbf{C}$$ with \eqref{eq:update_C}.

### Applications to learn embeddings

In this last part, we are going to see how inverse OT and the presented trick can be used to learn data representations, as shown in <d-cite key="van2024snekhorn"></d-cite>
.
We are given a dataset $$(\mathbf{x}_1, .., \mathbf{x}_n)$$ and the goal is to compute embeddings $$(\mathbf{z}_1, .., \mathbf{z}_n)$$ such that each $$\mathbf{z}_i$$ is a low-dimensional representation of the input data point $$\mathbf{x}_i$$.

To do so, we are going to look for a cost of the form $$d(\mathbf{z}_i, \mathbf{z}_j)$$ which solves inverse OT with an input $$\widehat{\mathbf{P}}$$ computed from $$(\mathbf{x}_1, .., \mathbf{x}_n)$$. To compute $$\widehat{\mathbf{P}}$$, one can simply solve the symmetric variant of entropic OT wich is exactly problem \eqref{eq:eot} with symmetric $$\mathbf{C}$$ $$=(d(\mathbf{x}_i, \mathbf{x}_j))_{ij}$$ and $$\mathbf{a}=\mathbf{b}$$. We pick $$\mathbf{a}=\mathbf{b}=\mathbf{1}$$ to give the same mass to every data point.

In symmetric entropic OT, we only have one dual variable $$\mathbf{f}$$ as the primal solution is given by $$\widehat{\mathbf{P}} = \exp((\mathbf{f}^\star \oplus \mathbf{f}^\star - \mathbf{C}) / \varepsilon)$$. Moreover $$\mathbf{f}^\star$$ can be computed by simply iterating <d-footnote> In the code we use the following well-conditioned variant : $f_i \leftarrow \frac{1}{2} (f_i-\varepsilon \log \sum_j e^{(f_j-C_{ij}) / \varepsilon})$. </d-footnote>.

$$
\begin{align}
  f_i &\leftarrow - \varepsilon \log \sum_j e^{(f_j-C_{ij}) / \varepsilon} \label{eq:sinkhorn-sym} \:.
\end{align}
$$

:bulb: In symmetric entropic OT, each point spreads its mass to its closest neighbors thus capturing the geometry of the data. In this context, the regularizer $$\varepsilon$$ controls the scale of dependencies that is captured.

Once we have computed $$\widehat{\mathbf{P}}$$, the goal is to solve the inverse problem of finding the embeddings $$(\mathbf{z}_1, .., \mathbf{z}_n)$$ that would generate a similar entropic OT plan in low-dimension. In other words, we want the geometry in the low-dimensional space to be similar to the one in input space. This method has strong connections with the t-SNE algorithm as developped in <d-cite key="van2024snekhorn"></d-cite> <d-footnote> This work relies on a more elaborate version of symmetric entropic OT for computing $\widehat{\mathbf{P}}$ but the methodology to update the $(\mathbf{z}_1, .., \mathbf{z}_n)$ is the same as here. </d-footnote>.

To do so, we rely on the presented trick for inverse OT and therefore focus on solving
$$
\begin{align}
    \min_{(\mathbf{z}_1, .., \mathbf{z}_n), \mathbf{f}, \mathbf{g}} \: \: \cal{G}(\widehat{\mathbf{P}}, \mathbf{C}_{\mathbf{Z}}, \mathbf{f}, \mathbf{g}) \:.
\end{align}
$$

where $$\mathbf{C}_{\mathbf{Z}}$$ it the symmetric cost matrix with entries $$d(\mathbf{z}_i, \mathbf{z}_j)$$.

We consider the common task of embedding the swiss roll (depicted below) from 3d to 2d.
![](/assets/img/blog-invot/swiss_roll.svg){:style="display:block; margin-left:auto; margin-right:auto; width:50%;"}

In the experiments, we take the squared Euclidean distance for $$d$$, $$\varepsilon=10$$ for the entropic regularizer and independent $$\cal{N}(0,1)$$ variables to initialize the embedding coordinates. The code is provided in the box below.

{% details Python Code %}
{% highlight python %}

import torch
import matplotlib.pyplot as plt
from sklearn import datasets
from tqdm import tqdm
import time

def symmetric_sinkhorn(C, eps=1e0, f0=None, max_iter=1000, tol=1e-6, verbose=False):
    """
    Performs Sinkhorn iterations in log domain to solve the entropic symmetric
    OT problem with symmetric cost C and entropic regularization eps.
    """
    n = C.shape[0]

    if f0 is None:
        f = torch.zeros(n, dtype=C.dtype, device=C.device)
    else:
        f = f0

    for k in range(max_iter):
        # well-conditioned symmetric Sinkhorn update
        f = 0.5 * (f - eps * torch.logsumexp((f - C) / eps, -1))

        # check convergence every 10 iterations
        if k % 10 == 0:
            log_T = (f[:, None] + f[None, :] - C) / eps
            if (torch.abs(torch.exp(torch.logsumexp(log_T, -1))-1) < tol).all():
                if verbose:
                    print(f'---------- Breaking at iter {k} ----------')
                break

        if k == max_iter-1:
            if verbose:
                print('---------- Max iter attained for Sinkhorn algorithm ----------')

    return (f[:, None] + f[None, :] - C) / eps, f

def inverse_OT_unrolling(log_P_hat, Z0=None, lr=1e0, eps=1e0, verbose=True, max_iter=1000, tol=1e-6):
    """
    Solves the inverse OT problem for an input P_hat using autodiff.
    """
    n = log_P_hat.shape[0]

    if Z0 is None:
        Z = torch.randn((n,2))
    else:
        Z = Z0

    Z.requires_grad = True
    optimizer = torch.optim.Adam([Z], lr=lr)
    f = torch.zeros(n, dtype=C.dtype, device=C.device)

    pbar = tqdm(range(max_iter))
    for k in pbar:
        Z_prev = Z.clone().detach()
        optimizer.zero_grad()

        C_z = torch.cdist(Z, Z, p=2)**2

        # Run Sinkhorn (with autograd) to update dual variables
        log_Q, f = symmetric_sinkhorn(C_z, eps=eps, max_iter=100, f0=f.detach())

        # Compute KL loss to update Z
        loss = (torch.exp(log_P_hat) * (log_P_hat - log_Q - 1) + torch.exp(log_Q)).sum()
        loss.backward()
        optimizer.step()

        # Check convergence
        delta = torch.abs(Z - Z_prev) / torch.abs(Z_prev)
        if (delta < tol).all():
            if verbose:
                print(f'---------- Breaking at iter {k+1} ----------')
            break

        if verbose:
            pbar.set_description(f'Loss : {float(loss.item()): .3e}, '
                                 f'Delta : {float(delta.mean().item()): .3e} '
                                )

    return Z

def inverse_OT_gap(log_P_hat, Z0=None, lr=1e0, eps=1e0, verbose=True, max_iter=1000, tol=1e-6):
    """
    Solves the inverse OT problem for an input P_hat using the trick detailed in the blog.
    """
    n = log_P_hat.shape[0]

    if Z0 is None:
        Z = torch.randn((n,2))
    else:
        Z = Z0

    Z.requires_grad = True
    optimizer = torch.optim.Adam([Z], lr=lr)
    f = torch.zeros(n, dtype=C.dtype, device=C.device)

    pbar = tqdm(range(max_iter))
    for k in pbar:
        Z_prev = Z.clone().detach()
        optimizer.zero_grad()

        C_z = torch.cdist(Z, Z, p=2)**2

        # Run Sinkhorn (without autograd) to update dual variables
        with torch.no_grad():
            _, f = symmetric_sinkhorn(C_z, eps=eps, max_iter=100, f0=f.detach())
        log_Q = (f[:, None] + f[None, :] - C_z) / eps

        # Compute Monge gap loss to update Z
        loss = (torch.exp(log_P_hat)*C_z).sum() + eps * torch.exp(torch.logsumexp(log_Q, dim=(0,1)))
        loss.backward()
        optimizer.step()

        # Check convergence
        delta = torch.abs(Z - Z_prev) / torch.abs(Z_prev)
        if (delta < tol).all():
            if verbose:
                print(f'---------- Breaking at iter {k+1} ----------')
            break

        if verbose:
            pbar.set_description(f'Loss : {float(loss.item()): .3e}, '
                                 f'Delta : {float(delta.mean().item()): .3e} '
                                )

    return Z

### Run the experiments with Swiss Roll

# We fix a scale via the regularizer epsilon
eps = 1e1

N_list = [50, 100, 200, 300, 600, 1000]

list_Z_unrolling = []
list_Z_gap = []
list_color = []
timings_unrolling = []
timings_gap = []

for n in N_list:
    # Load n datapoints of the Swiss roll
    sr_points, sr_color = datasets.make_swiss_roll(n_samples=n, random_state=0)
    list_color.append(sr_color)
    sr_points_torch = torch.tensor(sr_points, dtype=torch.double)

    # Compute the corresponding input P_hat
    C = torch.cdist(sr_points_torch, sr_points_torch, p=2)**2
    log_P, _ = symmetric_sinkhorn(C, eps=eps)

    # We use the same initialisation for both algorithms
    Z0 = torch.randn((n,2))

    # Solve inverse OT via unrolling
    start = time.time()
    Z = inverse_OT_unrolling(log_P, Z0.clone(), eps=eps)
    end = time.time()
    timings_unrolling.append(end-start)
    list_Z_unrolling.append(Z.detach().numpy())

    # Solve inverse OT via Monge gap
    start = time.time()
    Z_ = inverse_OT_gap(log_P, Z0.clone(), eps=eps)
    end = time.time()
    timings_gap.append(end-start)
    list_Z_gap.append(Z_.detach().numpy())

### Plot the results

fig, axs = plt.subplots(2, 3, figsize=(10, 6), layout='constrained')

for e,i in enumerate([1,3,5]):
    axs[0,e].set_title(f'{N_list[i]} points via unrolling')
    axs[0,e].scatter(list_Z_unrolling[i][:,0], list_Z_unrolling[i][:,1], c=list_color[i])

    axs[1,e].set_title(f'{N_list[i]} points via Monge gap')
    axs[1,e].scatter(list_Z_gap[i][:,0], list_Z_gap[i][:,1], c=list_color[i])

plt.savefig("swiss_roll_inverse_OT.svg", bbox_inches='tight')
plt.show()

plt.plot(N_list, timings_unrolling, label='Unrolling')
plt.scatter(N_list, timings_unrolling, marker='X', s=100)
plt.scatter(N_list, timings_gap, marker='X', s=100)
plt.plot(N_list, timings_gap, label='Monge gap')
plt.xlabel('Number of points', fontsize=15)
plt.ylabel('Time (s)', fontsize=15)
plt.legend(fontsize=15)
plt.title('Computation time for inverse OT', fontsize=20)
plt.savefig("timings.svg", bbox_inches='tight')
plt.show()

{% endhighlight %}
{% enddetails %}

First, as shown in the figure below,  we can verify that we obtain exactly the same embeddings $$(\mathbf{z}_1, .., \mathbf{z}_n)$$ using unrolling and the Monge gap trick presented in this blog.
![](/assets/img/blog-invot/swiss_roll_inverse_OT.svg){:style="display:block; margin-left:auto; margin-right:auto; width:100%;"}

Regarding run-time, the Monge gap approach is faster than unrolling as we can see on the following plot. Hence the trick presented in this blog has a great practical interest, especially for large-scale applications.
![](/assets/img/blog-invot/timings.svg){:style="display:block; margin-left:auto; margin-right:auto; width:50%;"}

:bulb: Inverse OT is also useful for contrastive learning as shown in <d-cite key="pmlr-v202-shi23j"></d-cite>. In contrastive learning, one constructs augmented views $$(\mathbf{y}_1, .., \mathbf{y}_r)$$ of input data points $$(\mathbf{x}_1, .., \mathbf{x}_n)$$. The ground truth coupling $$\widehat{\mathbf{P}}$$ is taken such that $$\widehat{P}_{ij}=1$$ if $$\mathbf{y}_j$$ is an augmented view of $$\mathbf{x}_i$$ and $$0$$ otherwise. Then, inverse OT can be applied to compute latent representations
$$
\begin{align}
(\phi_{\theta}(\mathbf{x}_1), .., \phi_{\theta}(\mathbf{x}_n), \phi_{\theta}(\mathbf{y}_1), ..., \phi_{\theta}(\mathbf{y}_r))
\end{align}
$$
where $$\phi_{\theta}$$ is a neural network. Note that both directed and symmetric inverse OT can be considered <d-footnote> Indeed, directed inverse OT corresponds to treating the $(\mathbf{x}_1, .., \mathbf{x}_n)$ as source points and the $(\mathbf{y}_1, .., \mathbf{y}_r)$ as target points while symmetric inverse OT treats each point indifferently. Both approach use $\widehat{\mathbf{P}}$ as target coupling.</d-footnote>. Interestingly, the trick presented in this blog can be applied in this context thus alleviating the need to perform backpropagation through the Sinkhorn iterations.

:pencil2: Feel free to contact me for any question or remark on this blog !

### Citation

If you found this useful, you can cite this blog post using:

```{.bibtex}
@article{inverse_ot_unrolling,
  title   = {Inverse optimal transport does not require unrolling},
  author  = {Hugues Van Assel},
  year    = {2024},
  month   = {April},
  url     = {https://huguesva.github.io/blog/2024/inverseOT_mongegap/}
}
```