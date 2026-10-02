<details markdown="1">
<summary>Dreamer's variational world-model loss and our approximation</summary>

The original Dreamer uses a recurrent state-space model<d-cite key="hafner2020dreamer"></d-cite>. Its state contains a deterministic memory $$\mathbf h_t$$ and a stochastic latent $$\mathbf Z_t$$. In our notation,

$$
\begin{aligned}
\mathbf h_t&=F(\mathbf h_{t-1},\mathbf Z_{t-1},\mathbf U_{t-1}),\\
q_t&=q_{\mathbf E}(\cdot\mid\mathbf h_t,\mathbf X_t),\\
p_t&=p_{\mathbf P}(\cdot\mid\mathbf h_t).
\end{aligned}
$$

The reconstruction-based world-model objective in the original paper, written as a loss to minimize, is

$$
\begin{aligned}
\mathcal L_{\mathrm{Dreamer}}^{\mathrm{var}}
=\mathbb E\!\sum_t\Big[
&-\log p_{\mathbf D}(\mathbf X_t\mid\mathbf h_t,\mathbf Z_t)\\
&-\log p_{\mathbf R}(R_t\mid\mathbf h_t,\mathbf Z_t)\\
&+\beta\,\mathrm{KL}(q_t\|p_t)
\Big].
\end{aligned}
$$

Here $$R_t$$ is the reward, and the expectation includes data and posterior samples. The memory summarizes past observations and actions. The reward term trains the latent state to support reward prediction. For tasks with early termination, Dreamer also learns to predict discounts; actor and value learning have separate objectives. Later Dreamer versions change the latent distributions and KL training rules.

Our one-step comparison uses $$q_{\mathbf E}(\cdot\mid\mathbf X_{t+1})$$ and $$p_{\mathbf P}(\cdot\mid\mathbf Z_t,\mathbf U_t)$$, with deterministic encoder means. For Gaussians with equal fixed covariance $$\tau^2\mathbf I$$, their KL is

$$
\mathrm{KL}\!\left(
\mathcal N(\boldsymbol\mu_q,\tau^2\mathbf I)
\,\middle\|\,
\mathcal N(\boldsymbol\mu_p,\tau^2\mathbf I)\right)
=\frac{\|\boldsymbol\mu_q-\boldsymbol\mu_p\|_2^2}{2\tau^2}.
$$

Absorbing this covariance scale into $$\beta$$ gives the latent squared error above. We leave out posterior-sampling effects, covariance learning, recurrent memory, and reward supervision to isolate feature selection from observation dynamics. This supports the simplified variance–predictability comparison.

</details>
