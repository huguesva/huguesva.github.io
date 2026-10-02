<details markdown="1">
<summary>The JEPA losses with VICReg and SIGReg, and our approximation</summary>

JEPA covers several training choices. Here are two ways to combine embedding prediction with an anti-collapse regularizer.

**VICReg regularization.** One way is to combine next-embedding prediction with VICReg's variance and covariance terms<d-cite key="bardes2022vicreg"></d-cite>:

$$
\begin{aligned}
\mathcal L_{\mathrm{JEPA\text{-}VICReg}}
&=\operatorname{MSE}(\widehat{\mathbf Z}_{t+1},\mathbf Z_{t+1})\\
&\quad+\frac{\lambda_{\mathrm{var}}}{2}
\sum_{s\in\{t,t+1\}}\mathcal V(\mathcal Z_s)
+\frac{\lambda_{\mathrm{cov}}}{2}
\sum_{s\in\{t,t+1\}}\mathcal C(\mathcal Z_s).
\end{aligned}
$$

For a batch $$\mathcal Z=\{\mathbf z_b\}_{b=1}^{B}$$, let $$\widehat{\boldsymbol\Sigma}_{\mathcal Z}$$ be its sample covariance, computed with denominator $$B-1$$. The regularizers are

$$
\begin{aligned}
\mathcal V(\mathcal Z)
&=\frac1k\sum_{j=1}^{k}
\max\!\left(0,\gamma-
\sqrt{(\widehat{\boldsymbol\Sigma}_{\mathcal Z})_{jj}+\epsilon}\right),\\
\mathcal C(\mathcal Z)
&=\frac1k\sum_{i\ne j}
(\widehat{\boldsymbol\Sigma}_{\mathcal Z})_{ij}^{2}.
\end{aligned}
$$

Together, the variance and covariance terms prevent collapse by keeping each coordinate variable while discouraging redundancy between coordinates.

**LeWorldModel with SIGReg.** Another option is to use SIGReg<d-cite key="maes2026leworldmodel"></d-cite>, which is closely related to a sliced maximum mean discrepancy (MMD) that matches the embedding distribution to a standard Gaussian.

**Our approximation.** We use one-step prediction and the simpler penalty $$\|\operatorname{Cov}(\mathbf Z)-\mathbf I_k\|_F^2$$. VICReg encourages a variance floor and small off-diagonal covariances. SIGReg encourages Gaussian projected distributions, including properties beyond second moments. Our penalty makes the spectral calculation tractable by only controlling second moments. The selection rule below is derived for this simplified objective.

</details>
