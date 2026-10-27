import torch


def v(t, x):
    return t**2 + x**3

if __name__ == "__main__":
    t = torch.tensor([[0.1], [0.2], [0.3]], dtype=torch.float32, requires_grad=True)
    x = torch.tensor([[-1], [0], [1]], dtype=torch.float32, requires_grad=True)

    v_x = torch.autograd.grad(
        outputs=v(t, x), inputs=x, grad_outputs=torch.ones_like(v(t, x)), create_graph=True
    )[0]
    v_t = torch.autograd.grad(
        outputs=v(t, x), inputs=t, grad_outputs=torch.ones_like(v(t, x)), create_graph=True
    )[0]

    v_xx = torch.autograd.grad(
        outputs=v_x, inputs=x, grad_outputs=torch.ones_like(v_x), create_graph=True
    )[0]
    v_tt = torch.autograd.grad(
        outputs=v_t, inputs=t, grad_outputs=torch.ones_like(v_t), create_graph=True
    )[0]

    print(v_x)
    print(v_t)
    print(v_xx)
    print(v_tt)
    print(v_x.shape)  # 输出梯度的形状
