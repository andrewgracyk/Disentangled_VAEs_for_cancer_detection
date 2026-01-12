This repository contains a VAE method for detecting cancer by diverging (disentangling) the latent space into salient and cancerous components. This method is convolutional NN and transpose convolutional NN based.

$$
\begin{aligned}
\min_{\theta, \phi} \max_{\omega} \mathcal{L} &= \underbrace{\mathbb{E}_{q_\phi}\left[ \| X - D_\theta(z_e, z_c) \|^2 + \lambda_e \| X - D_\theta(z_e,0) \|^2 \right]}_{\text{image reconstruction}} \\
&+ \underbrace{\alpha \text{KL} \big( q_\phi(z_e | X) \parallel p(z_e) \big)}_{\text{essential KL regularization}} + \underbrace{\gamma \text{KL} \big( q_\phi(z_c | X, y) \parallel p(z_c | y) \big)}_{\text{conditional cancerous KL regularization}} \\
&- \underbrace{\beta \mathbb{E}_{q_\phi} \big[ \text{CE}(C_{\omega}(z_e), y) \big]}_{\text{adversarial disentanglement}} .
\end{aligned}
$$


The motivation for this work is that there is a higher level of identifiability to a Vision Transformer (ViT), which is SOTA.
