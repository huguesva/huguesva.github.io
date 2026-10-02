<details markdown="1">
<summary>Derivation of the observation-prediction mode-selection score</summary>

Use the common eigenbasis from the setup. Thus

$$
\boldsymbol\Sigma_{\mathrm X}\boldsymbol\phi_j
=p_j\boldsymbol\phi_j,
$$

and mode $$j$$ has predictable fraction $$a_j$$.

**The predictable part of the next observation.** Under squared error, the
best unconstrained prediction of $$\mathbf X_{t+1}$$ from
$$(\mathbf X_t,\mathbf U_t)$$ is its conditional mean

$$
\mathbf M_t
:=\mathbb E[\mathbf X_{t+1}\mid\mathbf X_t,\mathbf U_t]
=\mathbf A_{\mathrm X}\mathbf X_t+\mathbf B_{\mathrm X}\mathbf U_t.
$$

Because the process noise is independent of $$(\mathbf X_t,\mathbf U_t)$$,

$$
\mathbf X_{t+1}=\mathbf M_t+\boldsymbol\varepsilon_t.
$$

The covariance of the predictable part is

$$
\begin{aligned}
\mathbf C_{\mathrm M}
:=\operatorname{Cov}(\mathbf M_t)
&=\sum_{j=1}^d
p_j(\rho_j^2+c_j)
\boldsymbol\phi_j\boldsymbol\phi_j^\top\\
&=\sum_{j=1}^d
p_ja_j\boldsymbol\phi_j\boldsymbol\phi_j^\top.
\end{aligned}
$$

Hence the eigenvectors of $$\mathbf C_{\mathrm M}$$ are the common modes
$$\boldsymbol\phi_j$$, and its eigenvalue in mode $$j$$ is the predictable
observation variance $$p_ja_j$$.

**From the network to a reduced-rank problem.** In the linear specialization,

$$
\begin{aligned}
\mathbf Z_t&=\mathbf E\mathbf X_t,\\
\widehat{\mathbf Z}_{t+1}
&=\mathbf K\mathbf Z_t+\mathbf B\mathbf U_t,\\
\widehat{\mathbf X}_{t+1}
&=\mathbf D\widehat{\mathbf Z}_{t+1},
\end{aligned}
$$

where the number of active coordinates is $$r\leq k$$ and
$$\mathbf E\boldsymbol\Sigma_{\mathrm X}\mathbf E^\top=\mathbf I_r$$.
The original optimization over
$$(\mathbf E,\mathbf K,\mathbf B,\mathbf D)$$ is not jointly convex: its
prediction contains the products $$\mathbf D\mathbf K\mathbf E$$ and
$$\mathbf D\mathbf B$$. We therefore do not use first-order conditions in the
raw network parameters.

Instead, observe that every decoded prediction belongs to the output subspace

$$
\mathcal U:=\operatorname{range}(\mathbf D),
\qquad
\dim(\mathcal U)\leq r\leq k.
$$

Let $$\boldsymbol\Pi_{\mathcal U}$$ be the orthogonal projector onto this
subspace. Among all $$\mathcal U$$-valued predictions, the one closest to
$$\mathbf M_t$$ is its orthogonal projection
$$\boldsymbol\Pi_{\mathcal U}\mathbf M_t$$. Therefore every network with output
subspace $$\mathcal U$$ satisfies

$$
\begin{aligned}
\mathbb E\|\mathbf X_{t+1}-\widehat{\mathbf X}_{t+1}\|_2^2
&\geq
\mathbb E\|\mathbf X_{t+1}
-\boldsymbol\Pi_{\mathcal U}\mathbf M_t\|_2^2\\
&=\operatorname{tr}(\boldsymbol\Sigma_{\mathrm X})
-\operatorname{tr}(\boldsymbol\Pi_{\mathcal U}\mathbf C_{\mathrm M}).
\end{aligned}
$$

The inequality uses the orthogonal decomposition into fresh noise and
conditional-mean prediction error. The equality uses stationarity,
$$\boldsymbol\Sigma_{\mathrm X}=\mathbf C_{\mathrm M}+\mathbf Q$$.

**Optimal output directions.** Minimizing the error bound amounts to
maximizing the captured predictable variance
$$\operatorname{tr}(\boldsymbol\Pi_{\mathcal U}\mathbf C_{\mathrm M})$$.
Expanding in the eigenbasis of $$\mathbf C_{\mathrm M}$$ gives

$$
\operatorname{tr}(\boldsymbol\Pi_{\mathcal U}\mathbf C_{\mathrm M})
=\sum_{j=1}^d p_ja_jq_j,
\qquad
q_j:=\|\boldsymbol\Pi_{\mathcal U}\boldsymbol\phi_j\|_2^2.
$$

Since $$\boldsymbol\Pi_{\mathcal U}$$ is an orthogonal projector onto a
subspace of dimension at most $$k$$,

$$
0\leq q_j\leq1,
\qquad
\sum_jq_j=\dim(\mathcal U)\leq k.
$$

The weighted sum cannot exceed the sum of the $$k$$ largest positive $$p_ja_j$$
(or all positive terms if there are fewer than $$k$$).
This bound is attained by taking $$\mathcal U$$ to be the span of those modes.
If $$S$$ indexes them, the resulting projector is

$$
\boldsymbol\Pi_S
=\sum_{j\in S}\boldsymbol\phi_j\boldsymbol\phi_j^\top.
$$

This identifies the best output subspace. It remains to show that our
encoder–predictor–decoder architecture can realize the projected conditional
mean $$\boldsymbol\Pi_S\mathbf M_t$$.

**A network that attains the bound.** Let $$S$$ be the selected set and
use one latent coordinate for each $$j\in S$$. Choose

$$
\mathbf E_j=\frac{\boldsymbol\phi_j^\top}{\sqrt{p_j}},
\qquad
K_{jj}=\rho_j,
\qquad
\mathbf B_j=\frac{\boldsymbol\phi_j^\top\mathbf B_{\mathrm X}}
{\sqrt{p_j}},
\qquad
\mathbf D_j=\sqrt{p_j}\boldsymbol\phi_j.
$$

Here $$\mathbf E_j$$ and $$\mathbf B_j$$ are rows, while $$\mathbf D_j$$ is a
decoder column; $$j$$ labels the selected mode assigned to that coordinate.
Set all off-diagonal entries of $$\mathbf K$$ to zero. These parameters satisfy
the encoder normalization and give

$$
\widehat{\mathbf X}_{t+1}
=\sum_{j\in S}
\boldsymbol\phi_j\boldsymbol\phi_j^\top\mathbf M_t.
$$

This is exactly the optimal projected conditional mean. The original network
therefore attains the error bound, proving global optimality.

**Mode-selection score.** For predictions constrained to the output span of
a mode set $$S$$, the bound is attained by the construction above. The minimized
loss is

$$
\mathcal L_{\mathrm{ObsPred}}^\star(S)
=\frac{1}{2\sigma_{\mathrm{dec}}^2}
\left[
\operatorname{tr}(\boldsymbol\Sigma_{\mathrm X})
-\sum_{j\in S}p_ja_j
\right].
$$

Adding mode $$j$$ decreases the loss by
$$p_ja_j/(2\sigma_{\mathrm{dec}}^2)$$. Dropping the common positive factor
$$1/2$$, which does not affect the ranking, gives

$$
\boxed{
\operatorname{score}_j^{\mathrm{ObsPred}}
=\frac{p_ja_j}{\sigma_{\mathrm{dec}}^2}
}.
$$

The closed-form solution retains up to $$k$$ modes with the largest positive
scores. Zero-score modes do not change the optimum, and ties may be resolved
arbitrarily.

</details>
