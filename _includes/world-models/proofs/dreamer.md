<details markdown="1">
<summary>Derivation of the Dreamer mode-selection score</summary>

**Assumptions.** Beyond the shared conditions in the VAE proof, each active
encoder coordinate is restricted to a distinct common mode rather than an
arbitrary mixture of modes.

Use the common eigenbasis from the setup:

$$
\boldsymbol\Sigma_{\mathrm X}\boldsymbol\phi_j
=p_j\boldsymbol\phi_j,
$$

with predictable fraction $$a_j$$ for mode $$j$$.

**Linear specialization.** We work within the common-modal encoder class. For
a retained set $$S$$, with $$r=|S|\leq k$$, use the normalized encoder rows

$$
\mathbf E_j=\frac{\boldsymbol\phi_j^\top}{\sqrt{p_j}},
\qquad j\in S.
$$

Then

$$
Z_{j,t}
=\frac{\boldsymbol\phi_j^\top\mathbf X_t}{\sqrt{p_j}},
\qquad
\operatorname{Var}(Z_{j,t})=1.
$$

For each retained set, the decoder and latent predictor are optimized freely.
Unlike E2C, this simplified Dreamer objective does not decode the predicted
latent state into the next observation. The decoder and predictor therefore
occur in separate terms and can be optimized independently.

**Optimal decoder.** For a retained mode $$j$$, the decoder column that minimizes
current-observation reconstruction error is

$$
\mathbf D_j^\star=\sqrt{p_j}\boldsymbol\phi_j.
$$

This follows from the unrestricted decoder solution
$$\mathbf D^\star=\boldsymbol\Sigma_{\mathrm X}\mathbf E^\top$$ derived in
the VAE proof. Distinct current modes are uncorrelated, so a retained
coordinate cannot help reconstruct an omitted one.
The decoder reconstructs each retained mode exactly. Every omitted mode contributes its full
variance $$p_j$$, so

$$
\min_{\mathbf D}
\mathbb E\|\mathbf D\mathbf E\mathbf X_t-\mathbf X_t\|_2^2
=\operatorname{tr}(\boldsymbol\Sigma_{\mathrm X})
-\sum_{j\in S}p_j.
$$

**Optimal latent predictor.** The squared-error-optimal prediction of a
retained coordinate is its conditional mean:

$$
\widehat Z_{j,t+1}^\star
=\mathbb E[Z_{j,t+1}\mid\mathbf Z_t,\mathbf U_t]
=\rho_jZ_{j,t}
+\frac{\boldsymbol\phi_j^\top\mathbf B_{\mathrm X}\mathbf U_t}
{\sqrt{p_j}}.
$$

The shared dynamics eigenbasis makes this conditional mean depend only on the
retained coordinate and the action; it requires no omitted state coordinates.
For any other predictor, the squared error equals the innovation variance
plus the expected squared difference from this conditional mean, because the
innovation is independent of the predictor inputs.
The remaining error is the normalized fresh innovation. Its variance is
$$1-a_j$$, and therefore

$$
\min_{\mathbf K,\mathbf B}
\mathbb E\|\mathbf K\mathbf Z_t+\mathbf B\mathbf U_t
-\mathbf Z_{t+1}\|_2^2
=\sum_{j\in S}(1-a_j).
$$

**Profiled loss.** Define

$$
h_j^{\mathrm{Dreamer}}
:=\frac{p_j}{\sigma_{\mathrm{dec}}^2}
-\beta(1-a_j).
$$

Substituting the optimal decoder and predictor gives

$$
\mathcal L_{\mathrm{Dreamer}}^\star(S)
=\frac{\operatorname{tr}(\boldsymbol\Sigma_{\mathrm X})}
{2\sigma_{\mathrm{dec}}^2}
-\frac12\sum_{j\in S}h_j^{\mathrm{Dreamer}}.
$$

Thus all remaining optimization is the choice of retained modes.

**Mode-selection score.** Retaining mode $$j$$ decreases the loss by
$$h_j^{\mathrm{Dreamer}}/2$$. We therefore keep up to $$k$$ modes with the largest
positive values of $$h_j^{\mathrm{Dreamer}}$$. Dropping the common factor $$1/2$$
gives the score

$$
\boxed{
\operatorname{score}_j^{\mathrm{Dreamer}}
=\frac{p_j}{\sigma_{\mathrm{dec}}^2}
-\beta(1-a_j)
}.
$$

Within the common-modal encoder class, the closed-form solution selects the
largest positive scores, up to capacity $$k$$. If every score is nonpositive,
the zero representation is optimal; zero-score modes leave the loss
unchanged, and ties may be resolved arbitrarily.

</details>
