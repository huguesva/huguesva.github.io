<details markdown="1">
<summary>The variational VAE loss and our approximation</summary>

The VAE stage of World Models combines reconstruction with a KL penalty toward a standard Gaussian prior<d-cite key="ha2018worldmodels"></d-cite>. Writing $$q_{\mathbf E}(\mathbf z\mid\mathbf x)$$ for the encoder distribution and $$p_{\mathbf D}(\mathbf x\mid\mathbf z)$$ for the decoder likelihood, the $$\beta$$-weighted objective is

$$
\begin{aligned}
\mathcal L_{\mathrm{VAE}}^{\mathrm{var}}
=\mathbb E_{\mathbf X_t}\!\Big[
&\mathbb E_{\mathbf Z_t\sim q_{\mathbf E}(\cdot\mid\mathbf X_t)}
[-\log p_{\mathbf D}(\mathbf X_t\mid\mathbf Z_t)]\\
&+\beta\,\mathrm{KL}\!\left(
q_{\mathbf E}(\cdot\mid\mathbf X_t)\,\middle\|\,
\mathcal N(\mathbf0,\mathbf I_k)\right)
\Big].
\end{aligned}
$$

For a Gaussian encoder $$q_{\mathbf E}=\mathcal N(\boldsymbol\mu,\mathbf C)$$, where $$\boldsymbol\mu=f_{\mathbf E}^{\mathrm{enc}}(\mathbf X_t)$$, the KL is

$$
\frac12\left[
\|\boldsymbol\mu\|_2^2+\operatorname{tr}(\mathbf C)
-\log\det\mathbf C-k\right].
$$

A fixed-variance Gaussian decoder gives a squared reconstruction loss. Our approximation evaluates the decoder at $$\boldsymbol\mu$$ and keeps the quadratic mean term of the KL. It omits covariance learning and the effect of posterior sampling on reconstruction. For example, with a linear decoder, sampling contributes the additional term $$\operatorname{tr}(\mathbf D\mathbf C\mathbf D^\top)/(2\sigma_{\mathrm{dec}}^2)$$, which depends on the decoder even when $$\mathbf C$$ is fixed. The normalization used for our feature-selection result is a further assumption, stated in the proof below.

In the *World Models* paper, the authors then train a recurrent latent predictor while keeping the encoder fixed.

</details>
