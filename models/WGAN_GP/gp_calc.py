import torch


def gradient_penalty(critic, real, fake, device="cpu"):
    batch_size = real.shape[0]
    epsilon = torch.rand((batch_size, 1, 1, 1)).to(device)
    interpolated = real * epsilon + fake * (1 - epsilon)

    critic_output = critic(interpolated)

    grad = torch.autograd.grad(
        inputs=interpolated,
        outputs=critic_output,
        grad_outputs=torch.ones_like(critic_output),
        create_graph=True,
        retain_graph=True,
    )[0]

    grad_norm = grad.view(grad.shape[0], -1).norm(2, dim=1)

    gp = torch.mean((grad_norm - 1) ** 2)

    return gp
