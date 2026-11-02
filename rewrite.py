# 1. 导入需要的库和解析解
import torch
from problem import exact_solution
import matplotlib.pyplot as plt

# 2. 编写内部点采样函数，训练用随机网格，验证画图用均匀网格
def sample_interior(n):
    t=torch.rand(n,1)
    x=torch.rand(n,1)*2-1
    return t, x

def sample_initial_grid(num_points=201):
    x = torch.linspace(-1, 1, num_points).view(-1, 1)
    t = torch.zeros_like(x)
    return t, x

# 3. 编写初始条件采样函数
def sample_initial(n):
    x=torch.rand(n,1)*2-1
    t=torch.zeros_like(x)
    return t, x

# 4. 编写边界条件采样函数

def sample_boundary(n):
    t=torch.rand(n,1)*2-1
    x_left=torch.full_like(t,-1.0)
    x_right=torch.full_like(t,1.0)
    return t, x_left, x_right

# 5. 定义 PDE、初始条件和边界条件损失
def L_pde(residual):
    return torch.mean(residual**2)

def L_IC(u_pred_ic, x_ic):
    u_exact_ic=torch.sin(torch.pi*x_ic)
    return torch.mean((u_pred_ic - u_exact_ic)**2)

def L_BC(u_pred_bl,u_pred_br):
    # u_exact_bl=torch.zeros_like(u_pred_bl)
    # u_exact_rl=torch.zeros_like(u_pred_rl)
    #都是0可以省略了
    return torch.mean(u_pred_bl**2) + torch.mean(u_pred_br**2)

# 6. 定义总损失
def L_total(L_pde, L_IC, L_BC):
    return L_pde + 0.5 * L_IC + 0.5 * L_BC

# 7. 创建神经网络和优化器
n_samples=100
n_steps=5000

if __name__ == "__main__":
    torch.manual_seed(42)
    net = torch.nn.Sequential(
        torch.nn.Linear(2, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 50),
        torch.nn.Tanh(),
        torch.nn.Linear(50, 1),
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# 8. 编写 PINN 训练循环
    for step in range(n_steps):
        optimizer.zero_grad()

        t,x=sample_interior(n=n_samples)
        t_ic,x_ic=sample_initial(n=n_samples)
        t_bc,x_left,x_right=sample_boundary(n=n_samples)

        t.requires_grad_(True)
        x.requires_grad_(True)

        tx=torch.cat([t,x],dim=1)
        tx_ic=torch.cat([t_ic,x_ic],dim=1)
        txl_bc=torch.cat([t_bc,x_left],dim=1)
        txr_bc=torch.cat([t_bc,x_right],dim=1)

        u_pred=net(tx)
        u_pred_ic=net(tx_ic)
        u_pred_bl=net(txl_bc)
        u_pred_br=net(txr_bc)

    # 9. 使用 autograd 计算一阶、二阶导数
        u_t=torch.autograd.grad(
        outputs=u_pred,
        inputs=t,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True
        )[0]

        u_x=torch.autograd.grad(
        outputs=u_pred,
        inputs=x,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True
        )[0]

        u_xx=torch.autograd.grad(
        outputs=u_x,
        inputs=x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
        )[0]
    # 10. 计算 PDE 残差和各项损失
        alpha = 1.0
        residual = u_t - alpha * u_xx

        loss_pde = L_pde(residual)
        loss_ic = L_IC(u_pred_ic, x_ic)
        loss_bc = L_BC(u_pred_bl, u_pred_br)
        loss_total = L_total(loss_pde, loss_ic, loss_bc)

    # 11. 反向传播并更新参数
        loss_total.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"Step {step}, Loss: {loss_total.item()}")

# 12. 在训练结束后评估模型
net.eval()
with torch.no_grad():
    t, x = sample_initial_grid()
    tx = torch.cat([t, x], dim=1)
    u_pred= net(tx)

    alpha = 1.0
    u_exact=exact_solution(t, x, alpha=alpha)

# 13. 将预测解与解析解进行比较并绘图
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(x.numpy(), u_pred.numpy(), label='PINN Prediction')
    plt.plot(x.numpy(), u_exact.numpy(), label='Exact Solution')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('PINN Prediction vs Exact Solution')
    plt.legend()
    max_abs=torch.max(torch.abs(u_pred-u_exact))
    print(f"Maximum absolute error: {max_abs.item()}")
    t_axis = torch.linspace(0, 1, 101)
    x_axis = torch.linspace(-1, 1, 201)
    t_grid, x_grid = torch.meshgrid(t_axis, x_axis, indexing="ij")
    t_eval = t_grid.reshape(-1, 1)
    x_eval = x_grid.reshape(-1, 1)
    tx_eval = torch.cat([t_eval, x_eval], dim=1)
    u_pred_eval = net(tx_eval).reshape(t_grid.shape)
    u_exact_eval = exact_solution(t_eval, x_eval, alpha=alpha).reshape(t_grid.shape)
    error_eval = torch.abs(u_pred_eval - u_exact_eval)
    MAE = torch.mean(error_eval)
    relative_L2 = torch.sqrt(torch.sum(error_eval**2) / torch.sum(u_exact_eval**2))
    print("Mean absolute error over the grid:", MAE.item())
    print("Relative L2 error over the grid:", relative_L2.item())
    u_left = u_pred_eval[:, 0]
    u_right = u_pred_eval[:, -1]
    max_abs_left = torch.max(torch.abs(u_left))
    max_abs_right = torch.max(torch.abs(u_right))
    print("Maximum absolute error at the left boundary:", max_abs_left.item())
    print("Maximum absolute error at the right boundary:", max_abs_right.item())
    plt.show()
