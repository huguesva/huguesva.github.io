<details markdown="1">
<summary>Derivation of the VAE mode-selection score</summary>

**Assumptions.** All five derivations concern the simplified linear objectives
above, with observation variances $$p_j>0$$, an integer capacity
$$1\leq k\leq d$$, and predictable fractions $$0\leq a_j\leq1$$ as implied by
stationarity. The later proofs add their own restrictions without repeating
these.

Let $$\boldsymbol\Sigma_{\mathrm X}\succ0$$ be the observation covariance, with
orthonormal eigenvectors $$\boldsymbol\phi_j$$ and eigenvalues $$p_j$$:

$$
\boldsymbol\Sigma_{\mathrm X}\boldsymbol\phi_j
=p_j\boldsymbol\phi_j.
$$

**Linear specialization.** Let $$\mathbf E\in\mathbb R^{r\times d}$$ and
$$\mathbf D\in\mathbb R^{d\times r}$$ be the encoder and decoder matrices:

$$
\mathbf Z_t=\mathbf E\mathbf X_t,
\qquad
\widehat{\mathbf X}_t=\mathbf D\mathbf Z_t,
\qquad
\mathbf E\boldsymbol\Sigma_{\mathrm X}\mathbf E^\top=\mathbf I_r,
\qquad
0\leq r\leq k.
$$

The normalization fixes the scale of every active coordinate. In particular,

$$
\mathbb E\|\mathbf E\mathbf X_t\|_2^2
=\operatorname{tr}(\mathbf E\boldsymbol\Sigma_{\mathrm X}\mathbf E^\top)
=r,
$$

so the mean-dependent KL contribution is $$\beta r/2$$.

**Optimal decoder.** For a fixed normalized encoder, completing the square
gives

$$
\begin{aligned}
\mathbb E\|\mathbf D\mathbf E\mathbf X_t-\mathbf X_t\|_2^2
={}&\operatorname{tr}(\boldsymbol\Sigma_{\mathrm X})
-\operatorname{tr}(\mathbf E\boldsymbol\Sigma_{\mathrm X}^2\mathbf E^\top)\\
&+\|\mathbf D-\boldsymbol\Sigma_{\mathrm X}\mathbf E^\top\|_F^2.
\end{aligned}
$$

The final term is minimized at

$$
\mathbf D^\star=\boldsymbol\Sigma_{\mathrm X}\mathbf E^\top.
$$

Define the whitened encoder

$$
\mathbf W:=\mathbf E\boldsymbol\Sigma_{\mathrm X}^{1/2}.
$$

Then $$\mathbf W\mathbf W^\top=\mathbf I_r$$. The matrix

$$
\boldsymbol\Pi:=\mathbf W^\top\mathbf W
$$

is therefore the orthogonal projector onto the row space of the whitened
encoder $$\mathbf W$$:

$$
\boldsymbol\Pi^2=\boldsymbol\Pi,
\qquad
\boldsymbol\Pi^\top=\boldsymbol\Pi,
\qquad
\operatorname{tr}(\boldsymbol\Pi)=r\leq k.
$$

Using cyclicity of the trace,

$$
\operatorname{tr}(\mathbf E\boldsymbol\Sigma_{\mathrm X}^2\mathbf E^\top)
=\operatorname{tr}(\boldsymbol\Pi\boldsymbol\Sigma_{\mathrm X}).
$$

Conversely, every rank-$$r$$ orthogonal projector can be written as
$$\mathbf W^\top\mathbf W$$ by taking the rows of $$\mathbf W$$ to be an
orthonormal basis of its range. Setting
$$\mathbf E=\mathbf W\boldsymbol\Sigma_{\mathrm X}^{-1/2}$$ then gives an
admissible normalized encoder. Thus this change of variables loses no
admissible encoder subspaces.

After optimizing the decoder, the VAE loss is consequently

$$
\mathcal L_{\mathrm{VAE}}^\star(\boldsymbol\Pi)
=\frac{\operatorname{tr}(\boldsymbol\Sigma_{\mathrm X})}
{2\sigma_{\mathrm{dec}}^2}
-\frac12\operatorname{tr}
\left[
\boldsymbol\Pi
\left(
\frac{\boldsymbol\Sigma_{\mathrm X}}{\sigma_{\mathrm{dec}}^2}
-\beta\mathbf I_d
\right)
\right].
$$

**Optimal encoder directions.** Define the symmetric score matrix

$$
\mathbf H_{\mathrm{VAE}}
:=\frac{\boldsymbol\Sigma_{\mathrm X}}{\sigma_{\mathrm{dec}}^2}
-\beta\mathbf I_d.
$$

Its eigenvectors are the observation covariance eigenvectors
$$\boldsymbol\phi_j$$, with eigenvalues

$$
h_j:=\frac{p_j}{\sigma_{\mathrm{dec}}^2}-\beta.
$$

Minimizing the loss is therefore equivalent to maximizing
$$\operatorname{tr}(\boldsymbol\Pi\mathbf H_{\mathrm{VAE}})$$ over encoder
subspaces of dimension at most $$k$$. Expand this trace in the eigenbasis:

$$
\operatorname{tr}(\boldsymbol\Pi\mathbf H_{\mathrm{VAE}})
=\sum_{j=1}^d h_jq_j,
\qquad
q_j:=\boldsymbol\phi_j^\top\boldsymbol\Pi\boldsymbol\phi_j
=\|\boldsymbol\Pi\boldsymbol\phi_j\|_2^2.
$$

Here $$q_j$$ is the squared length of mode $$j$$'s projection onto the encoder
subspace. Because $$\boldsymbol\Pi$$ is an orthogonal projector,

$$
0\leq q_j\leq1,
\qquad
\sum_jq_j=\operatorname{tr}(\boldsymbol\Pi)\leq k.
$$

The weighted sum cannot exceed the sum of the $$k$$ largest positive $$h_j$$
(or all positive $$h_j$$ if there are fewer than $$k$$).
This bound is attained by projecting onto exactly those modes:

$$
\boldsymbol\Pi^\star
=\sum_{j\in S}\boldsymbol\phi_j\boldsymbol\phi_j^\top,
$$

where $$S$$ indexes the selected modes. Thus a mode-aligned encoder is globally
optimal among all normalized linear encoders; its alignment has been derived,
rather than assumed.

**Recovering the encoder and decoder.** For every selected mode $$j\in S$$, take

$$
\mathbf E_j=\frac{\boldsymbol\phi_j^\top}{\sqrt{p_j}},
\qquad
\mathbf D_j^\star=\sqrt{p_j}\boldsymbol\phi_j.
$$

These rows and columns realize the projector $$\boldsymbol\Pi^\star$$ and the optimal
decoder derived above.

**Mode-selection score.** Retaining mode $$j$$ decreases the profiled loss by
$$h_j/2$$. Dropping the common positive factor $$1/2$$ gives

$$
\boxed{
\operatorname{score}_j^{\mathrm{VAE}}
=\frac{p_j}{\sigma_{\mathrm{dec}}^2}-\beta
}.
$$

The closed-form solution selects the largest positive scores, up to capacity
$$k$$. If every score is nonpositive, the zero representation is optimal;
zero-score modes leave the loss unchanged, and ties may be resolved
arbitrarily.

</details>
