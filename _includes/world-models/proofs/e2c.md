<details markdown="1">
<summary>Derivation of the E2C mode-selection score</summary>

**Assumptions.** Beyond the shared conditions in the VAE proof, each active
encoder coordinate is restricted to a distinct common mode rather than an
arbitrary mixture of modes, and the transition predictor is fixed to the
conditional mean.

Use the common eigenbasis from the setup:

$$
\boldsymbol\Sigma_{\mathrm X}\boldsymbol\phi_j
=p_j\boldsymbol\phi_j,
$$

with predictable fraction $$a_j$$ for mode $$j$$.
We assume $$\sigma_{\mathrm{dec}}^2>0$$, $$\beta>0$$, and
$$\lambda_{\mathrm{con}}\geq0$$.

**Linear specialization.** We work within the common-modal encoder class. For
a retained set $$S$$, with $$r=|S|\leq k$$, use the normalized encoder rows

$$
\mathbf E_j=\frac{\boldsymbol\phi_j^\top}{\sqrt{p_j}},
\qquad j\in S.
$$

The corresponding coordinate has unit variance:

$$
Z_{j,t}
=\frac{\boldsymbol\phi_j^\top\mathbf X_t}{\sqrt{p_j}},
\qquad
\operatorname{Var}(Z_{j,t})=1.
$$

By the assumption above, the transition predictor is the conditional mean of
the next encoded state:

$$
\widehat Z_{j,t+1}
=\rho_jZ_{j,t}
+\frac{\boldsymbol\phi_j^\top\mathbf B_{\mathrm X}\mathbf U_t}
{\sqrt{p_j}}.
$$

Write

$$
Z_{j,t+1}=\widehat Z_{j,t+1}+\eta_{j,t}.
$$

The conditional-mean prediction and fresh innovation are uncorrelated, with

$$
\mathbb E[\widehat Z_{j,t+1}^2]=a_j,
\qquad
\mathbb E[\eta_{j,t}^2]=1-a_j.
$$

**Optimal shared decoder.** E2C uses the same decoder for the current encoding
and the predicted next encoding. First we check that an unrestricted decoder
has an optimum aligned with the retained modes. Define $$\mathbf D_0$$ to have
columns $$\sqrt{p_j}\boldsymbol\phi_j$$ for $$j\in S$$, and let
$$\mathbf A_S=\operatorname{diag}(a_j:j\in S)$$. The shared eigenbasis and
independence assumptions give

$$
\begin{aligned}
\mathbb E[\mathbf Z_t\mathbf Z_t^\top]&=\mathbf I_r,
&
\mathbb E[\widehat{\mathbf Z}_{t+1}\widehat{\mathbf Z}_{t+1}^\top]
&=\mathbf A_S,\\
\mathbb E[\mathbf X_t\mathbf Z_t^\top]&=\mathbf D_0,
&
\mathbb E[\mathbf X_{t+1}\widehat{\mathbf Z}_{t+1}^\top]
&=\mathbf D_0\mathbf A_S.
\end{aligned}
$$

Let $$R(\mathbf D)$$ be the sum of the current and future squared observation
errors. Differentiating this convex quadratic yields

$$
\nabla_{\mathbf D}R
=2(\mathbf D-\mathbf D_0)(\mathbf I_r+\mathbf A_S).
$$

Every diagonal entry of $$\mathbf I_r+\mathbf A_S$$ is positive. Thus the
unique optimal decoder is $$\mathbf D^\star=\mathbf D_0$$; components along
other modes cannot improve it.

To evaluate the resulting error, write column $$j$$ as
$$d_j\boldsymbol\phi_j$$. Its current-observation error along that mode is

$$
\mathbb E\left[
\left(d_jZ_{j,t}-\boldsymbol\phi_j^\top\mathbf X_t\right)^2
\right]
=(d_j-\sqrt{p_j})^2.
$$

For the next observation,

$$
\boldsymbol\phi_j^\top\mathbf X_{t+1}
=\sqrt{p_j}(\widehat Z_{j,t+1}+\eta_{j,t}),
$$

so the prediction error is

$$
\begin{aligned}
&\mathbb E\left[
\left(d_j\widehat Z_{j,t+1}
-\boldsymbol\phi_j^\top\mathbf X_{t+1}\right)^2
\right]\\
&\qquad
=a_j(d_j-\sqrt{p_j})^2+p_j(1-a_j).
\end{aligned}
$$

Adding the current and future observation errors gives the convex quadratic

$$
(1+a_j)(d_j-\sqrt{p_j})^2+p_j(1-a_j),
$$

whose unique minimizer is

$$
d_j^\star=\sqrt{p_j}.
$$

An omitted mode contributes total observation error $$2p_j$$. A retained mode
at the optimal decoder contributes only $$p_j(1-a_j)$$. Retaining it therefore
reduces the two observation errors by

$$
p_j(1+a_j).
$$

Normalization also gives a VAE KL cost $$\beta/2$$ for every retained mode. The
E2C consistency cost is

$$
\frac{\lambda_{\mathrm{con}}}{2}
\mathbb E[(\widehat Z_{j,t+1}-Z_{j,t+1})^2]
=\frac{\lambda_{\mathrm{con}}}{2}(1-a_j).
$$

**Profiled loss.** Define

$$
h_j^{\mathrm{E2C}}
:=\frac{p_j(1+a_j)}{\sigma_{\mathrm{dec}}^2}
-\beta-\lambda_{\mathrm{con}}(1-a_j).
$$

After optimizing the shared decoder, the loss for a retained set $$S$$ is

$$
\mathcal L_{\mathrm{E2C}}^\star(S)
=\frac{\operatorname{tr}(\boldsymbol\Sigma_{\mathrm X})}
{\sigma_{\mathrm{dec}}^2}
-\frac12\sum_{j\in S}h_j^{\mathrm{E2C}}.
$$

The constant is the loss when no modes are retained. Thus all remaining
optimization is the choice of $$S$$.

**Mode-selection score.** Retaining mode $$j$$ decreases the loss by
$$h_j^{\mathrm{E2C}}/2$$. We therefore keep up to $$k$$ modes with the largest
positive values of $$h_j^{\mathrm{E2C}}$$. Dropping the common factor $$1/2$$
gives the score

$$
\boxed{
\operatorname{score}_j^{\mathrm{E2C}}
=\frac{p_j(1+a_j)}{\sigma_{\mathrm{dec}}^2}
-\beta-\lambda_{\mathrm{con}}(1-a_j)
}.
$$

Within the common-modal encoder class and with the transition fixed to the
conditional mean, the closed-form solution selects the largest positive
scores, up to capacity $$k$$. If every score is nonpositive, the zero
representation is optimal; zero-score modes leave the loss unchanged, and
ties may be resolved arbitrarily.

</details>
