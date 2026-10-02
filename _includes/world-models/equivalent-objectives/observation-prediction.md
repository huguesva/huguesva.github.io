<details markdown="1">
<summary>The observation-prediction loss and our approximation</summary>

Oh et al.<d-cite key="oh2015action"></d-cite> train action-conditioned video predictors using squared errors over several future frames. Written as an average over training sequences, the loss is

$$
\mathcal L_{\mathrm{video}}
=\mathbb E\!\left[
\frac{1}{2H}\sum_{h=1}^{H}
\|\widehat{\mathbf X}_{t+h}-\mathbf X_{t+h}\|_2^2
\right].
$$

Here $$H$$ is the training prediction horizon. The predictions use the observed frame history and the intervening actions; multi-step training feeds predicted frames back into the model. The paper includes both a fixed-history encoder and a recurrent encoder, and increases $$H$$ during training.

Our comparison uses $$H=1$$, the current observation and action, and a fixed Gaussian likelihood scale. The squared-error learning signal is already present in the original method. The simplification concerns the history, prediction horizon, and linear maps used in the proof.

</details>
