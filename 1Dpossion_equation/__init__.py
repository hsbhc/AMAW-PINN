if self.net.iter % gradient_epochs == 0:
    temp = torch.linspace(-1, 1, 10000).unsqueeze(-1)
    x = Variable(temp, requires_grad=True)
    u = self.train_U(x)
    dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0].squeeze()
    dxx = torch.autograd.grad(dx, x, grad_outputs=torch.ones_like(dx), create_graph=True)[0].squeeze()
    u = abs(u)
    u = (u - min(u)) / (max(u) - min(u))
    dx = abs(dx)
    # dx = (dx - min(dx)) / (max(dx) - min(dx))
    dxx = abs(dxx)
    dxx = (dxx - min(dxx)) / (max(dxx) - min(dxx))
    d = dx / torch.mean(dx) + 1
    # d = (d / torch.mean(d)).cpu().detach().numpy() + 1
    d_normalized = (d / sum(d)).cpu().detach().numpy()
    # d_normalized = torch.softmax(d / 2, dim=0).cpu().detach().numpy()
    X_ids = np.random.choice(a=len(temp), size=len(self.x_f), replace=False, p=d_normalized)
    self.x_f = temp[X_ids]
    draw_exact(self.x_f, self.x_test, self.x_test_exact)