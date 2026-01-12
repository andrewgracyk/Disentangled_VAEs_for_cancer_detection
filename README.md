This repository contains a VAE method for detecting cancer by diverging (disentangling) the latent space into salient and cancerous components. This method is convolutional NN and transpose convolutional NN based.

$$
\begin{aligned}
\min_{\theta, \phi} \max_{\omega} \mathcal{L} &= \underbrace{\mathbb{E}_{q_\phi}\left[ \| X - D_\theta(z_e, z_c) \|^2 + \lambda_e \| X - D_\theta(z_e,0) \|^2 \right]}_{\text{image reconstruction}} \\
&+ \underbrace{\alpha \text{KL} \big( q_\phi(z_e | X) \parallel p(z_e) \big)}_{\text{essential KL regularization}} + \underbrace{\gamma \text{KL} \big( q_\phi(z_c | X, y) \parallel p(z_c | y) \big)}_{\text{conditional cancerous KL regularization}} \\
&- \underbrace{\beta \mathbb{E}_{q_\phi} \big[ \text{CE}(C_{\omega}(z_e), y) \big]}_{\text{adversarial disentanglement}}  ,
\end{aligned}
$$
updated discretely as
$$
\begin{aligned}
& \omega^{(j+1)} = \omega^{(j)} + \eta \nabla_\omega  \frac{1}{N} \sum_{i=1}^N \bigg[ -\beta \text{CE}(C_\omega(z_{e,i}), y_i)  \bigg]
\\
& \phi^{(j+1)} = \phi^{(j)} - \eta \nabla_\phi \frac{1}{N} \sum_{i=1}^N \bigg[ || X_i - D_\theta(z_{e,i}, z_{c,i}) ||^2 + \lambda_e || X_i - D_\theta(z_{e,i}, 0) ||^2 
\\
&  \ \ \ \ \ \ \ \ \ \  + \alpha \text{KL} \big( q_\phi(z_{e,i} | X_i) \parallel \mathcal{N}(0, I) \big) + \gamma \text{KL} \big( q_\phi(z_{c,i} | X_i, y_i) \parallel p(z_c | y_i) \big) - \beta \text{CE}(C_{\omega}(z_{e,i}), y_i) \bigg]
\\
& \theta^{(j+1)} = \theta^{(j)} - \eta \nabla_\theta \frac{1}{N} \sum_{i=1}^N \bigg[ || X_i - D_\theta(z_{e,i}, z_{c,i}) ||^2 + \lambda_e || X_i - D_\theta(z_{e,i}, 0) ||^2 \bigg] .
\end{aligned}
$$

The motivation for this work is that there is a higher level of identifiability to a Vision Transformer (ViT), which is SOTA.
