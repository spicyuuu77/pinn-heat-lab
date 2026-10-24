import torch
from problem import exact_solution

def r(t,x, alpha):
    # 计算 PDE 的残差 r(t, x) = u_t - alpha * u_xx
    u = exact_solution(t, x, alpha=alpha)

    u_t = torch.autograd.grad(
        outputs=u,
        inputs=t,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    u_x = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
    )[0]

    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=x, 
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]
    return u_t - alpha * u_xx


if __name__ == "__main__":

    t = torch.tensor([[0.1], [0.2], [0.5]], dtype=torch.float32, requires_grad=True)
    x = torch.tensor([[-0.5], [0.25], [0.5]], dtype=torch.float32, requires_grad=True)
    alpha = 1.0  # 热扩散系数

    residual = r(t, x, alpha=alpha)
    print("Residual r(t, x):", residual)

    # 计算残差的最大绝对值
    max_abs = torch.max(torch.abs(residual))
    assert max_abs.item() < 1e-5, "PDE residual is too large"
    print("Maximum absolute value of residual:", max_abs.item())
