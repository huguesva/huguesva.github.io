<details markdown="1">
<summary>Derivation of the JEPA mode-selection score</summary>

**Assumptions.** Beyond the shared conditions in the VAE proof, each active encoder coordinate is restricted to a distinct common mode rather than an arbitrary mixture of modes, as made precise below.

Let $$\boldsymbol\phi_j$$ be a common mode from the setup. Its observation variance is $$p_j$$, the corresponding eigenvalue of the observation covariance:

$$
\boldsymbol\Sigma_{\mathrm X}\boldsymbol\phi_j
=p_j\boldsymbol\phi_j.
$$

The separate quantity $$a_j$$ is the fraction of that variance predictable from the current state and action. Both describe the data. Below, we optimize the encoder scale and show that $$p_j$$ cancels from the minimized JEPA loss, leaving a ranking determined only by $$a_j$$.

**Encoder class.** A general linear encoder has arbitrary rows $$\mathbf E_\ell\in\mathbb R^{1\times d}$$ and produces

$$
Z_{\ell,t}=\mathbf E_\ell\mathbf X_t.
$$

For this proof, we restrict each row to a multiple of one common mode, with distinct modes assigned to different coordinates:

$$
\mathbf E_\ell=s_\ell\boldsymbol\phi_{j_\ell}^\top,
\qquad \ell=1,\ldots,k.
$$

Within this class, both the selected modes $$j_\ell$$ and their scales $$s_\ell$$ remain to be optimized. The proof does not establish that an unrestricted optimal encoder must belong to this class.

**Parameterizing the encoder by latent variance.** Since the modes are unit-norm eigenvectors of the covariance matrix, the variance of coordinate $$\ell$$ is

$$
\begin{aligned}
v_\ell
:=\operatorname{Var}(Z_{\ell,t})
&=\mathbf E_\ell\boldsymbol\Sigma_{\mathrm X}\mathbf E_\ell^\top\\
&=s_\ell^2\boldsymbol\phi_{j_\ell}^\top
\boldsymbol\Sigma_{\mathrm X}\boldsymbol\phi_{j_\ell}\\
&=s_\ell^2p_{j_\ell}.
\end{aligned}
$$

An eigenvector's sign is arbitrary, so we may take $$s_\ell\geq0$$. Because $$p_{j_\ell}>0$$, choosing the scale $$s_\ell$$ is equivalent to choosing $$v_\ell\geq0$$, with

$$
s_\ell=\sqrt{\frac{v_\ell}{p_{j_\ell}}}.
$$

We can therefore rewrite the same encoder row as

$$
\mathbf E_\ell
=\sqrt{\frac{v_\ell}{p_{j_\ell}}}
\boldsymbol\phi_{j_\ell}^\top.
$$

This gives

$$
Z_{\ell,t}
=\sqrt{\frac{v_\ell}{p_{j_\ell}}}
\boldsymbol\phi_{j_\ell}^\top\mathbf X_t,
\qquad
\operatorname{Var}(Z_{\ell,t})=v_\ell.
$$

Distinct common modes are uncorrelated, so

$$
\operatorname{Cov}(\mathbf Z_t)
=\operatorname{diag}(v_1,\ldots,v_k),
$$

and the covariance penalty becomes

$$
\left\|\operatorname{Cov}(\mathbf Z_t)-\mathbf I_k\right\|_F^2
=\sum_{\ell=1}^k(v_\ell-1)^2.
$$

**Optimal latent predictor.** Consider a coordinate assigned to mode $$j$$ with latent variance $$v$$. The optimal squared-error predictor is

$$
\widehat Z_{t+1}^\star
=\rho_j Z_t
+\sqrt{\frac{v}{p_j}}
\boldsymbol\phi_j^\top\mathbf B_{\mathrm X}\mathbf U_t.
$$

The residual is the fresh innovation in mode $$j$$, scaled by $$\sqrt{v/p_j}$$. It is independent of the current latent state and action, so no predictor using those inputs can reduce its squared error. Its variance is

$$
\begin{aligned}
\mathbb E\!\left[
(\widehat Z_{t+1}^\star-Z_{t+1})^2
\right]
&=\frac{v}{p_j}\,p_j(1-a_j)\\
&=v(1-a_j).
\end{aligned}
$$

**Optimal coordinate variance.** After optimizing the predictor, a coordinate assigned to mode $$j$$ contributes

$$
\ell_j(v)
=(1-a_j)v+(v-1)^2,
\qquad v\geq0.
$$

This is a strictly convex quadratic, with

$$
\ell_j'(v)=2v-(1+a_j),
\qquad
\ell_j''(v)=2>0.
$$

The unique unconstrained minimizer is therefore

$$
v_j^\star=\frac{1+a_j}{2}.
$$

Since $$0\leq a_j\leq1$$, we have $$v_j^\star\in[1/2,1]$$, so the nonnegativity constraint is inactive. Substituting gives the optimized cost

$$
g_j
:=\ell_j(v_j^\star)
=1-\frac{(1+a_j)^2}{4}.
$$

This also shows that all $$k$$ coordinates are used. A zero coordinate has cost

$$
\ell_j(0)=1,
$$

whereas

$$
g_j\leq\frac34.
$$

Since $$k\leq d$$, any zero coordinate can therefore be assigned an unused mode and activated to strictly lower the loss.

For a selected set $$S$$ containing exactly $$k$$ distinct modes, the fully optimized JEPA loss is

$$
\mathcal L_{\mathrm{JEPA}}^\star(S)
=\sum_{j\in S}g_j.
$$

**Mode-selection score.** The loss is a sum of $$k$$ independent mode costs, so it is minimized by choosing the $$k$$ smallest $$g_j$$. Since

$$
g_j
=1-\frac{(1+a_j)^2}{4}
$$

is strictly decreasing in $$a_j$$ on $$[0,1]$$, these are exactly the $$k$$ modes with the largest predictable fractions. Thus we may use the score

$$
\boxed{
\operatorname{score}_j^{\mathrm{JEPA}}=a_j
}.
$$

Within the common-modal encoder class, the closed-form solution therefore selects exactly the $$k$$ modes with the largest $$a_j$$; ties may be resolved arbitrarily.

The observation variance $$p_j$$ only determines the encoder scale needed to achieve a given latent variance and disappears after optimizing that scale. Without an observation reconstruction term, it therefore does not affect the mode ranking.

</details>