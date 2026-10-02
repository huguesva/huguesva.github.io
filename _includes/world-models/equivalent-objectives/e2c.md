<details markdown="1">
<summary>The variational E2C loss and our approximation</summary>

For a transition $$(\mathbf X_t,\mathbf U_t,\mathbf X_{t+1})$$, let $$q_t=q_{\mathbf E}(\cdot\mid\mathbf X_t)$$ and $$q_{t+1}=q_{\mathbf E}(\cdot\mid\mathbf X_{t+1})$$. Let $$\widehat q_{t+1}$$ be the next-latent distribution predicted by E2C's locally linear transition, and let $$p_0=\mathcal N(\mathbf0,\mathbf I_k)$$. E2C's training objective combines four terms<d-cite key="watter2015e2c"></d-cite>:

$$
\begin{aligned}
\mathcal L_{\mathrm{E2C}}^{\mathrm{var}}
=\mathbb E\!\Big[
&-\log p_{\mathbf D}(\mathbf X_t\mid\mathbf Z_t)\\
&-\log p_{\mathbf D}(\mathbf X_{t+1}\mid\widehat{\mathbf Z}_{t+1})\\
&+\mathrm{KL}(q_t\|p_0)\\
&+\lambda_{\mathrm{con}}\,
\mathrm{KL}(\widehat q_{t+1}\|q_{t+1})
\Big].
\end{aligned}
$$

The expectation includes the data, encoder samples, and transition samples. The original paper uses Gaussian latent distributions and Bernoulli observation likelihoods. The consistency KL points from the predicted distribution toward the next encoding.

Our Gaussian observation model turns the two reconstruction terms into squared errors. We use encoder means and the mean-matching part of the consistency KL, absorbing its fixed covariance weight into $$\lambda_{\mathrm{con}}$$. We also allow a weight $$\beta$$ on the fixed-prior KL. The proof makes two further choices: global linear maps and a transition predictor fixed to the conditional mean. Learned covariances, posterior sampling, and state-dependent local transition matrices are outside this reduction.

</details>
