import os
import time
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.autograd import Variable
from tqdm import trange

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_gpu = False  # torch.cuda.is_available()


class Net(nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.layers = layers
        self.iter = 0
        self.activation = nn.Tanh()
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = self.activation(self.linear[0](x))
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
        a = self.linear[-1](a)
        return a


class Model:
    def __init__(self, net, net_transform_fun, x_label, x_labels, x_label_w, x_f, x_f_loss_fun,
                 x_test, x_test_exact, x_test_estimate_epochs):

        self.net = net
        self.net_transform_fun = net_transform_fun

        self.x_label = x_label
        self.x_labels = x_labels
        self.x_label_w = x_label_w
        self.x_f = x_f
        self.x_f_loss_fun = x_f_loss_fun

        self.x_test = x_test
        self.x_test_exact = x_test_exact
        self.x_test_estimate_epochs = x_test_estimate_epochs

        self.x_label_loss_collect = []
        self.x_f_loss_collect = []
        self.x_test_estimate_collect = []

    def train_U(self, x):
        if self.net_transform_fun is None:
            return self.net(x)
        else:
            return self.net_transform_fun(x, self.net(x))

    def predict_U(self, x):
        return self.train_U(x)

    # computer backward loss
    def epoch_loss(self):
        loss_equation = torch.mean(self.x_f_loss_fun(self.x_f, self.train_U) ** 2)
        self.x_f_loss_collect.append([self.net.iter, loss_equation.item()])
        loss = loss_equation

        if self.x_label is not None:
            loss_label = torch.mean((self.train_U(self.x_label) - self.x_labels) ** 2)
            self.x_label_loss_collect.append([self.net.iter, loss_label.item()])
            loss += self.x_label_w * loss_label

        loss.backward()
        return loss

    def estimate(self):
        loss_equation = torch.mean(self.x_f_loss_fun(self.x_test, self.train_U) ** 2)
        loss = loss_equation
        # if self.x_label is not None:
        #     loss_label = torch.mean((self.train_U(self.x_label) - self.x_labels) ** 2)
        #     loss += self.x_label_w * loss_label
        test_loss = loss.cpu().detach().numpy()
        pred = self.train_U(x_test).cpu().detach().numpy()
        exact = self.x_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        self.x_test_estimate_collect.append([self.net.iter, test_loss, error])
        return error, test_loss

    def train(self, epochs, lr=0.001, is_random=False,
              random_epochs=10000,
              is_RAM=False,
              RAM_epochs=10000,
              is_WAM=False,
              WAM_epochs=10000
              ):

        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        start_time = time.time()
        if is_random:
            pbar = trange(epochs, ncols=100)
            for i in pbar:
                optimizer.zero_grad()
                loss = self.epoch_loss()
                optimizer.step()
                self.net.iter += 1
                pbar.set_postfix({'Iter': self.net.iter,
                                  'Loss': '{0:.2e}'.format(loss.item())})

                if self.net.iter % self.x_test_estimate_epochs == 0:
                    _ = self.estimate()

                if self.net.iter % random_epochs == 0:
                    point_N = int(len(self.x_f) / 2)
                    point_M = len(self.x_f) - point_N
                    temp1 = torch.linspace(-1, 1, point_N).unsqueeze(-1)
                    temp = torch.full([point_M, 1], -1) + torch.rand([point_M, 1]) * 2

                    self.x_f = torch.cat((temp1, temp), dim=0)

                if self.net.iter == epochs:
                    break
        elif is_RAM:
            pbar = trange(epochs, ncols=100)
            for i in pbar:
                optimizer.zero_grad()
                loss = self.epoch_loss()
                optimizer.step()
                self.net.iter += 1
                pbar.set_postfix({'Iter': self.net.iter,
                                  'Loss': '{0:.2e}'.format(loss.item())})

                if self.net.iter % self.x_test_estimate_epochs == 0:
                    _ = self.estimate()

                if self.net.iter % RAM_epochs == 0:
                    point_N = int(len(self.x_f) / 2)
                    point_M = len(self.x_f) - point_N
                    temp1 = torch.linspace(-1, 1, point_N).unsqueeze(-1)

                    x_init = torch.linspace(-1, 1, 1000).unsqueeze(-1)
                    x_init_residual = abs(self.x_f_loss_fun(x_init, self.train_U))
                    x_init_residual = x_init_residual.cpu().detach().numpy()
                    err_eq = np.power(x_init_residual, 1) / np.power(x_init_residual, 1).mean()
                    err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
                    X_ids = np.random.choice(a=len(x_init), size=point_M, replace=False, p=err_eq_normalized)
                    temp = x_init[X_ids]
                    self.x_f = torch.cat((temp1, temp), dim=0)
                    # draw_exact(self.x_f,self.x_test,self.x_test_exact)

                if self.net.iter == epochs:
                    break
        elif is_WAM:
            pbar = trange(epochs, ncols=100)
            for i in pbar:
                optimizer.zero_grad()
                loss = self.epoch_loss()
                optimizer.step()
                self.net.iter += 1
                pbar.set_postfix({'Iter': self.net.iter,
                                  'Loss': '{0:.2e}'.format(loss.item())})

                if self.net.iter % self.x_test_estimate_epochs == 0:
                    _ = self.estimate()

                if self.net.iter % WAM_epochs == 0:
                    point_N = int(len(self.x_f) / 2)
                    point_M = len(self.x_f) - point_N
                    temp1 = torch.linspace(-1, 1, point_N).unsqueeze(-1)

                    x_init = torch.linspace(-1, 1, 1000).unsqueeze(-1)
                    x = Variable(x_init, requires_grad=True)
                    u = self.train_U(x)
                    dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0].squeeze()
                    dx = torch.sqrt(1 + dx ** 2).cpu().detach().numpy()
                    err_eq = np.power(dx, 1) / np.power(dx, 1).mean()
                    p = (err_eq / sum(err_eq))
                    select_index = np.random.choice(len(x_init), point_M, replace=False,
                                                    p=p)
                    temp = x_init[select_index]
                    self.x_f = torch.cat((temp1, temp), dim=0)

                if self.net.iter == epochs:
                    break
        else:
            pbar = trange(epochs, ncols=100)
            for i in pbar:
                optimizer.zero_grad()
                loss = self.epoch_loss()
                optimizer.step()
                self.net.iter += 1
                pbar.set_postfix({'Iter': self.net.iter,
                                  'Loss': '{0:.2e}'.format(loss.item())})

                if self.net.iter % self.x_test_estimate_epochs == 0:
                    _ = self.estimate()

                if self.net.iter == epochs:
                    break

        elapsed = time.time() - start_time
        print('Training time: %.2f' % elapsed)


def x_f_loss_fun(x, train_U):
    if not x.requires_grad:
        x = Variable(x, requires_grad=True)
    u = train_U(x)
    dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    dxx = torch.autograd.grad(dx, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    f = dxx + 50 * (100 - 100 * torch.tanh(50 * x) ** 2) * torch.tanh(50 * x) + \
        1.6 * torch.pi ** 2 * torch.sin(4 * torch.pi * x)
    return f


def draw_exact(points, x_test, u_test, predict=False):
    x_test_np = x_test.cpu().detach().numpy()
    u_test_np = u_test.cpu().detach().numpy()
    points_np = points.cpu().detach().numpy()
    plt.plot(x_test_np, u_test_np, 'b-', label='Exact')
    if predict:
        predict_np = model.predict_U(x_test).cpu().detach().numpy()
        plt.plot(x_test_np, predict_np, 'r--', label='Prediction')
    plt.plot(points_np, exact_u(points).cpu().detach().numpy(), 'kx', markersize=4, clip_on=False)
    plt.xlabel('$x$')
    plt.ylabel('$u$')
    plt.legend()
    plt.show()


def draw_residual(x_test, u_test, points, add_points=None, show_exact=False, title=''):
    f = x_f_loss_fun(x_test, model.train_U)
    f = f.cpu().detach().numpy()

    x_test_np = x_test.cpu().detach().numpy()
    u_test_np = u_test.cpu().detach().numpy()
    points_np = points.cpu().detach().numpy()

    if show_exact:
        plt.subplot(2, 1, 1)
    plt.plot(x_test_np, f ** 2)
    plt.xlabel('$x$')
    plt.ylabel('$Residual$')
    plt.title(title)
    plt.plot(points_np, np.zeros_like(points_np), 'kx', markersize=4, clip_on=False)
    if add_points is not None:
        adds = add_points.cpu().detach().numpy()
        plt.plot(adds, np.zeros_like(adds), 'rx', markersize=4, clip_on=False)

    if show_exact:
        plt.subplot(2, 1, 2)
        plt.plot(x_test_np, u_test_np, 'b-', label='Exact')
        predict_np = model.predict_U(x_test).cpu().detach().numpy()
        plt.plot(x_test_np, predict_np, 'r--', label='Prediction')
        plt.xlabel('$x$')
        plt.ylabel('$u$')
        if add_points is not None:
            adds = add_points.cpu().detach().numpy()
            plt.plot(adds, np.zeros_like(adds), 'rx', markersize=4, clip_on=False)
        plt.legend()
    plt.show()


def draw_x_test_estimate():
    x_test_estimate_collect = np.array(model.x_test_estimate_collect)
    plt.subplot(2, 1, 1)
    plt.yscale('log')
    plt.plot(x_test_estimate_collect[:, 0], x_test_estimate_collect[:, 1], 'g-')
    plt.xlabel('$Epoch$')
    plt.ylabel('$Test-loss$')

    plt.subplot(2, 1, 2)
    plt.yscale('log')
    plt.plot(x_test_estimate_collect[:, 0], x_test_estimate_collect[:, 2], 'g-')
    plt.xlabel('$Epoch$')
    plt.ylabel('$Test-error$')

    plt.show()


def draw_epoch_loss():
    x_label_loss_collect = np.array(model.x_label_loss_collect)
    x_f_loss_collect = np.array(model.x_f_loss_collect)
    plt.subplot(2, 1, 1)
    plt.yscale('log')
    plt.plot(x_label_loss_collect[:, 0], x_label_loss_collect[:, 1], 'b-', label='Label_loss')
    plt.xlabel('$Epoch$')
    plt.ylabel('$Loss$')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.yscale('log')
    plt.plot(x_f_loss_collect[:, 0], x_f_loss_collect[:, 1], 'r-', label='PDE_loss')
    plt.xlabel('$Epoch$')
    plt.ylabel('$Loss$')
    plt.legend()
    plt.show()


def getNew_Model(Nf):
    Nf = Nf
    Nt = 10000

    x_label = torch.tensor([[-1], [1]]).float()
    x_labels = exact_u(x_label)
    x_f = torch.linspace(-1, 1, Nf).unsqueeze(-1)
    x_test = torch.linspace(-1, 1, Nt).unsqueeze(-1)
    x_test_exact = exact_u(x_test)

    seed = 1234
    torch.set_default_dtype(torch.float)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    layers = [1, 20, 20, 20, 20, 1]
    net = Net(layers)

    if use_gpu:
        net = net.cuda()
        x_label = x_label.cuda()
        x_labels = x_labels.cuda()
        x_f = x_f.cuda()
        x_test = x_test.cuda()
        x_test_exact = x_test_exact.cuda()

    model = Model(
        net=net,
        net_transform_fun=lambda x, u: u,
        x_label=x_label,
        x_labels=x_labels,
        x_label_w=10,
        x_f=x_f,
        x_f_loss_fun=x_f_loss_fun,
        x_test=x_test,
        x_test_exact=x_test_exact,
        x_test_estimate_epochs=2000)
    return model


if __name__ == '__main__':
    exact_u = lambda x: 0.1 * torch.sin(4 * np.pi * x) + torch.tanh(50 * x)

    x_test = torch.linspace(-1, 1, 10000).unsqueeze(-1)
    x_test_exact = exact_u(x_test)

    if use_gpu:
        x_test = x_test.cuda()
        x_test_exact = x_test_exact.cuda()

    all_error_residual_name = ['PINN', 'Random', 'RAM', 'WAM']

    all_Nf = []
    all_error = []
    all_residual = []

    epoch = 50000
    change_epoch = 10000
    for Nf in range(30, 120, 20):
        all_Nf.append(Nf)
        error = []
        residual = []

        model = getNew_Model(Nf=Nf)
        model.train(epochs=epoch)
        x_test_estimate_collect = np.array(model.x_test_estimate_collect)
        min_test_loss_index = np.argmin(x_test_estimate_collect[:, 1])
        min_test_loss = x_test_estimate_collect[min_test_loss_index, 1]
        min_test_loss_l2error = x_test_estimate_collect[min_test_loss_index, 2]
        error.append(min_test_loss_l2error)
        residual.append(min_test_loss)

        model = getNew_Model(Nf=Nf)
        model.train(epochs=epoch, is_random=True, random_epochs=change_epoch)
        x_test_estimate_collect = np.array(model.x_test_estimate_collect)
        min_test_loss_index = np.argmin(x_test_estimate_collect[:, 1])
        min_test_loss = x_test_estimate_collect[min_test_loss_index, 1]
        min_test_loss_l2error = x_test_estimate_collect[min_test_loss_index, 2]
        error.append(min_test_loss_l2error)
        residual.append(min_test_loss)
        # f = x_f_loss_fun(x_test, model.train_U)
        # f = f.cpu().detach().numpy()
        # residual.append(sum(abs(f)))

        model = getNew_Model(Nf=Nf)
        model.train(epochs=epoch, is_RAM=True, RAM_epochs=change_epoch)
        x_test_estimate_collect = np.array(model.x_test_estimate_collect)
        min_test_loss_index = np.argmin(x_test_estimate_collect[:, 1])
        min_test_loss = x_test_estimate_collect[min_test_loss_index, 1]
        min_test_loss_l2error = x_test_estimate_collect[min_test_loss_index, 2]
        error.append(min_test_loss_l2error)
        residual.append(min_test_loss)
        # f = x_f_loss_fun(x_test, model.train_U)
        # f = f.cpu().detach().numpy()
        # residual.append(sum(abs(f)))

        model = getNew_Model(Nf=Nf)
        model.train(epochs=epoch, is_WAM=True, WAM_epochs=change_epoch)
        x_test_estimate_collect = np.array(model.x_test_estimate_collect)
        min_test_loss_index = np.argmin(x_test_estimate_collect[:, 1])
        min_test_loss = x_test_estimate_collect[min_test_loss_index, 1]
        min_test_loss_l2error = x_test_estimate_collect[min_test_loss_index, 2]
        error.append(min_test_loss_l2error)
        residual.append(min_test_loss)
        # f = x_f_loss_fun(x_test, model.train_U)
        # f = f.cpu().detach().numpy()
        # residual.append(sum(abs(f)))

        all_error.append(error)
        all_residual.append(residual)

    all_error = np.array(all_error)
    all_residual = np.array(all_residual)

    plt.yscale('log')
    plt.plot(all_Nf, all_error[:, 0], 'k-', label=all_error_residual_name[0])
    plt.plot(all_Nf, all_error[:, 1], 'b-',
             label=all_error_residual_name[1])
    plt.plot(all_Nf, all_error[:, 2], 'y-', label=all_error_residual_name[2])
    plt.plot(all_Nf, all_error[:, 3], 'r-',
             label=all_error_residual_name[3])
    plt.xlabel('$point-num$')
    plt.ylabel('$Test-error$')
    plt.legend()
    plt.show()

    plt.yscale('log')
    plt.plot(all_Nf, all_residual[:, 0], 'k-', label=all_error_residual_name[0])
    plt.plot(all_Nf, all_residual[:, 1], 'b-',
             label=all_error_residual_name[1])
    plt.plot(all_Nf, all_residual[:, 2], 'y-', label=all_error_residual_name[2])
    plt.plot(all_Nf, all_residual[:, 3], 'r-',
             label=all_error_residual_name[3])

    plt.xlabel('$point-num$')
    plt.ylabel('$Test-residual$')
    plt.legend()
    plt.show()

    print(all_Nf)
    print(all_error)
    print(all_residual)
