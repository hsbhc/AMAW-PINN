import os
import time
import numpy as np
import torch
from matplotlib import pyplot as plt, gridspec
from torch import nn
from torch.autograd import Variable
from tqdm import tqdm, trange
from pyDOE import lhs
import PDE_Lid_driven_Gordon.Lid_driven_Gordon as lid_driven

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
seed = 1234
torch.set_default_dtype(torch.float)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

use_gpu = torch.cuda.is_available()
print('GPU:', use_gpu)


def random_fun(num):
    temp = torch.from_numpy(lb + (ub - lb) * lhs(2, num)).float()
    if use_gpu:
        temp = temp.cuda()
    return temp


def is_cuda(data):
    if use_gpu:
        data = data.cuda()
    return data


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


class Net_attention(nn.Module):
    def __init__(self, layers):
        super(Net_attention, self).__init__()
        self.layers = layers
        self.iter = 0
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.attention1 = nn.Linear(layers[0], layers[1])
        self.attention2 = nn.Linear(layers[0], layers[1])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linear[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linear[i].bias.data)
        nn.init.xavier_normal_(self.attention1.weight.data, gain=1.0)
        nn.init.zeros_(self.attention1.bias.data)
        nn.init.xavier_normal_(self.attention2.weight.data, gain=1.0)
        nn.init.zeros_(self.attention2.bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = self.activation(self.linear[0](x))
        encoder_1 = self.activation(self.attention1(x))
        encoder_2 = self.activation(self.attention2(x))
        a = a * encoder_1 + (1 - a) * encoder_2
        for i in range(1, len(self.layers) - 2):
            z = self.linear[i](a)
            a = self.activation(z)
            a = a * encoder_1 + (1 - a) * encoder_2
        a = self.linear[-1](a)
        return a


class Model:
    def __init__(self, net, x_label, x_labels, x_f_loss_fun,
                 x_test, x_test_exact
                 ):

        self.x_label_s = None
        self.x_f_s = None
        self.s_collect = []

        self.optimizer_LBGFS = None
        self.net = net

        self.x_label = x_label
        self.x_labels = x_labels

        self.x_f_N = None
        self.x_f_M = None

        self.x_f_loss_fun = x_f_loss_fun

        self.x_test = x_test
        self.x_test_exact = x_test_exact

        self.start_loss_collect = False
        self.x_label_loss_collect = []
        self.x_f_loss_collect = []
        self.x_test_estimate_collect = []

    def train_U(self, x):
        return self.net(x)

    def predict_U(self, x):
        return self.train_U(x)

    def likelihood_loss(self, loss_e, loss_l):
        loss = torch.exp(-self.x_f_s) * loss_e.detach() + self.x_f_s \
               + torch.exp(-self.x_label_s) * loss_l.detach() + self.x_label_s
        return loss

    def true_loss(self, loss_e, loss_l):
        return torch.exp(-self.x_f_s.detach()) * loss_e + torch.exp(-self.x_label_s.detach()) * loss_l

    # computer backward loss
    def epoch_loss(self):
        x_f = torch.cat((self.x_f_N, self.x_f_M), dim=0)
        f_u, f_v, f_f = self.x_f_loss_fun(x_f, self.train_U)

        loss_equation = torch.mean(f_u ** 2)
        loss_equation += torch.mean(f_v ** 2)
        loss_equation += torch.mean(f_f ** 2)

        out = self.train_U(self.x_label)
        u, v, p = out[:, [0]], out[:, [1]], out[:, [2]]
        loss_label = torch.mean((torch.cat((u, v), dim=-1) - self.x_labels) ** 2)

        return loss_equation, loss_label

    # computer backward loss
    def LBGFS_epoch_loss(self):
        self.optimizer_LBGFS.zero_grad()
        x_f = torch.cat((self.x_f_N, self.x_f_M), dim=0)
        f_u, f_v, f_f = self.x_f_loss_fun(x_f, self.train_U)

        loss_equation = torch.mean(f_u ** 2)
        loss_equation += torch.mean(f_v ** 2)
        loss_equation += torch.mean(f_f ** 2)

        out = self.train_U(self.x_label)
        u, v, p = out[:, [0]], out[:, [1]], out[:, [2]]
        loss_label = torch.mean((torch.cat((u, v), dim=-1) - self.x_labels) ** 2)

        loss = self.true_loss(loss_equation, loss_label)
        loss.backward()
        self.net.iter += 1
        print('Iter:', self.net.iter, 'Loss:', loss.item())
        return loss

    def evaluate(self):
        pred = self.train_U(x_test).cpu().detach().numpy()
        u_pred, v_pred, p_pred = pred[:, [0]], pred[:, [1]], pred[:, [2]]
        pred = np.sqrt(u_pred ** 2 + v_pred ** 2)
        exact = self.x_test_exact.cpu().detach().numpy()
        error = np.linalg.norm(pred - exact, 2) / np.linalg.norm(exact, 2)
        return error

    def run_baseline(self):
        optimizer_adam = torch.optim.Adam(self.net.parameters(), lr=adam_lr)
        self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(), lr=lbgfs_lr,
                                                 max_iter=lbgfs_iter)
        pbar = trange(adam_iter, ncols=100)
        for i in pbar:
            optimizer_adam.zero_grad()
            loss_e, loss_label = self.epoch_loss()
            loss = self.true_loss(loss_e, loss_label)
            loss.backward()
            optimizer_adam.step()
            self.net.iter += 1
            pbar.set_postfix({'Iter': self.net.iter,
                              'Loss': '{0:.2e}'.format(loss.item())
                              })

        print('Adam done!')
        self.optimizer_LBGFS.step(self.LBGFS_epoch_loss)
        print('LBGFS done!')

        error = self.evaluate()
        print('Test_L2error:', '{0:.2e}'.format(error))

    def run_AM(self):
        for move_count in range(AM_count):
            self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(), lr=lbgfs_lr,
                                                     max_iter=lbgfs_iter)
            optimizer_adam = torch.optim.Adam(self.net.parameters(), lr=adam_lr)
            pbar = trange(adam_iter, ncols=100)

            for i in pbar:
                optimizer_adam.zero_grad()
                loss_equation, loss_label = self.epoch_loss()
                loss = self.true_loss(loss_equation, loss_label)
                loss.backward()
                self.net.iter += 1
                optimizer_adam.step()
                pbar.set_postfix({'Iter': self.net.iter,
                                  'Loss': '{0:.2e}'.format(loss.item())
                                  })

            print('Adam done!')
            self.optimizer_LBGFS.step(self.LBGFS_epoch_loss)
            print('LBGFS done!')

            error = self.evaluate()
            print('change_counts', move_count, 'Test_L2error:', '{0:.2e}'.format(error))
            self.x_test_estimate_collect.append([move_count, '{0:.2e}'.format(error)])

            if AM_type == 0:
                x_init = random_fun(100000)
                x_init_residual = abs(self.x_f_loss_fun(x_init, self.train_U))
                x_init_residual = x_init_residual.cpu().detach().numpy()
                err_eq = np.power(x_init_residual, AM_K) / np.power(x_init_residual, AM_K).mean()
                err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
                X_ids = np.random.choice(a=len(x_init), size=M, replace=False, p=err_eq_normalized)
                self.x_f_M = x_init[X_ids]

            elif AM_type == 1:
                x_init = random_fun(100000)
                x = Variable(x_init, requires_grad=True)
                u = self.train_U(x)
                dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
                grad_x1 = dx[:, [0]].squeeze()
                grad_x2 = dx[:, [1]].squeeze()
                dx = torch.sqrt(1 + grad_x1 ** 2 + grad_x2 ** 2).cpu().detach().numpy()
                err_dx = np.power(dx, AM_K) / np.power(dx, AM_K).mean()
                p = (err_dx / sum(err_dx))
                X_ids = np.random.choice(a=len(x_init), size=M, replace=False, p=p)
                self.x_f_M = x_init[X_ids]

    def run_AM_AW1(self):
        self.x_f_s = nn.Parameter(self.x_f_s, requires_grad=True)
        self.x_label_s = nn.Parameter(self.x_label_s, requires_grad=True)

        for move_count in range(AM_count):
            self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(), lr=lbgfs_lr,
                                                     max_iter=lbgfs_iter)
            optimizer_adam = torch.optim.Adam(self.net.parameters(), lr=adam_lr)
            optimizer_adam_weight = torch.optim.Adam([self.x_f_s] + [self.x_label_s],
                                                     lr=AW_lr)

            pbar = trange(adam_iter, ncols=100)
            for i in pbar:
                self.s_collect.append([self.net.iter, self.x_f_s.item(), self.x_label_s.item()])

                loss_e, loss_label = self.epoch_loss()

                optimizer_adam.zero_grad()
                loss = self.true_loss(loss_e, loss_label)
                loss.backward()
                optimizer_adam.step()
                self.net.iter += 1
                pbar.set_postfix({'Iter': self.net.iter,
                                  'Loss': '{0:.2e}'.format(loss.item())
                                  })

                optimizer_adam_weight.zero_grad()
                loss = self.likelihood_loss(loss_e, loss_label)
                loss.backward()
                optimizer_adam_weight.step()

            print('Adam done!')
            self.optimizer_LBGFS.step(self.LBGFS_epoch_loss)
            print('LBGFS done!')

            error = self.evaluate()
            print('change_counts', move_count, 'Test_L2error:', '{0:.2e}'.format(error))
            self.x_test_estimate_collect.append([move_count, '{0:.2e}'.format(error)])

            if AM_type == 0:
                x_init = random_fun(100000)
                f_u, f_v, f_f = self.x_f_loss_fun(x_init, self.train_U)
                x_init_residual = abs(f_u) + abs(f_v) + abs(f_f)
                x_init_residual = x_init_residual.cpu().detach().numpy()
                err_eq = np.power(x_init_residual, AM_K) / np.power(x_init_residual, AM_K).mean()
                err_eq_normalized = (err_eq / sum(err_eq))[:, 0]
                X_ids = np.random.choice(a=len(x_init), size=M, replace=False, p=err_eq_normalized)
                self.x_f_M = x_init[X_ids]

            elif AM_type == 1:
                x_init = random_fun(100000)
                x = Variable(x_init, requires_grad=True)
                out = self.train_U(x)
                u_pred, v_pred, p_pred = out[:, [0]], out[:, [1]], out[:, [2]]
                predict = torch.sqrt(u_pred ** 2 + v_pred ** 2)
                udx = torch.autograd.grad(predict, x, grad_outputs=torch.ones_like(u_pred), create_graph=True)[0]
                u_grad_x1 = udx[:, [0]].squeeze()
                u_grad_x2 = udx[:, [1]].squeeze()
                dx = torch.sqrt(1 + u_grad_x1 ** 2 + u_grad_x2 ** 2).cpu().detach().numpy()
                err_dx = np.power(dx, AM_K) / np.power(dx, AM_K).mean()
                p = (err_dx / sum(err_dx))
                X_ids = np.random.choice(a=len(x_init), size=M, replace=False, p=p)
                self.x_f_M = x_init[X_ids]

    def run_AM_AW(self):
        self.run_AM()
        self.net.iter = 0
        self.start_loss_collect = True
        print('AW start!')
        self.x_f_s = nn.Parameter(self.x_f_s, requires_grad=True)
        self.x_label_s = nn.Parameter(self.x_label_s, requires_grad=True)

        optimizer_adam_weight = torch.optim.Adam([self.x_f_s] + [self.x_label_s],
                                                 lr=AW_lr)

        self.optimizer_LBGFS = torch.optim.LBFGS(self.net.parameters(), lr=lbgfs_lr,
                                                 max_iter=lbgfs_iter)
        optimizer_adam = torch.optim.Adam(self.net.parameters(), lr=adam_lr)

        pbar = trange(adam_iter, ncols=100)
        for i in pbar:
            self.s_collect.append([self.net.iter, self.x_f_s.item(), self.x_label_s.item()])

            loss_e, loss_label = self.epoch_loss()

            optimizer_adam.zero_grad()
            loss = self.true_loss(loss_e, loss_label)
            loss.backward()
            optimizer_adam.step()
            self.net.iter += 1
            pbar.set_postfix({'Iter': self.net.iter,
                              'Loss': '{0:.2e}'.format(loss.item())
                              })

            optimizer_adam_weight.zero_grad()
            loss = self.likelihood_loss(loss_e, loss_label)
            loss.backward()
            optimizer_adam_weight.step()

        print('Adam done!')
        self.optimizer_LBGFS.step(self.LBGFS_epoch_loss)
        print('LBGFS done!')

        error = self.evaluate()
        print('change_counts', -1, 'Test_L2error:', '{0:.2e}'.format(error))
        self.x_test_estimate_collect.append([-1, '{0:.2e}'.format(error)])

    def train(self):

        self.x_f_N = random_fun(N)
        self.x_f_M = random_fun(M)

        # self.x_f_s = is_cuda(-torch.log(torch.tensor(1.).float()))
        # self.x_label_s = is_cuda(
        #     -torch.log(torch.tensor(100.).float()))  # 0.5*torch.exp(-self.x_label_s.detach()) = 100

        self.x_f_s = is_cuda(torch.tensor(0.).float())
        self.x_label_s = is_cuda(torch.tensor(0.).float())

        start_time = time.time()
        if model_type == 0:
            self.run_baseline()
        elif model_type == 1:
            self.run_AM()
        elif model_type == 2:
            self.run_AM_AW1()
        elapsed = time.time() - start_time
        print('Training time: %.2f' % elapsed)


def x_f_loss_fun(x, train_U):
    if not x.requires_grad:
        x = Variable(x, requires_grad=True)
    out = train_U(x)
    u, v, p = out[:, [0]], out[:, [1]], out[:, [2]]
    ud = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)
    udx = ud[0][:, [0]]
    udy = ud[0][:, [1]]
    udxx = torch.autograd.grad(udx, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, [0]]
    udyy = torch.autograd.grad(udy, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, [1]]

    vd = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)
    vdx = vd[0][:, [0]]
    vdy = vd[0][:, [1]]
    vdxx = torch.autograd.grad(vdx, x, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, [0]]
    vdyy = torch.autograd.grad(vdy, x, grad_outputs=torch.ones_like(v), create_graph=True)[0][:, [1]]

    pd = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)
    pdx = pd[0][:, [0]]
    pdy = pd[0][:, [1]]

    f_u = u * udx + v * udy + pdx - (udxx + udyy) / 100.
    f_v = u * vdx + v * vdy + pdy - (vdxx + vdyy) / 100.
    f_f = udx + vdy

    return f_u, f_v, f_f


def draw_exact():
    predict = model.predict_U(x_test).cpu().detach().numpy()
    u_pred, v_pred, p_pred = predict[:, [0]], predict[:, [1]], predict[:, [2]]
    predict_np = np.sqrt(u_pred ** 2 + v_pred ** 2)
    u_test_np = x_test_exact.cpu().detach().numpy()
    XX1, XX2 = np.meshgrid(x, y)

    e = np.reshape(u_test_np, (XX1.shape[0], XX1.shape[1]))
    plt.pcolor(XX1, XX2, e, shading='auto', cmap='jet')
    plt.colorbar()
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$y$', fontsize=20)
    plt.title(r'Exact $Velocity$', fontsize=20)
    plt.tight_layout()
    plt.savefig('lid_driven_exact.pdf')
    plt.show()

    e = np.reshape(predict_np, (XX1.shape[0], XX1.shape[1]))
    plt.pcolormesh(XX1, XX2, e, shading='auto', cmap='jet')
    plt.colorbar()
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$y$', fontsize=20)
    plt.title(r'WAM-AW $Velocity$', fontsize=20)
    plt.tight_layout()
    plt.savefig('lid_driven_pred_WAM-AW.pdf')
    plt.show()


def draw_exact_points(points, N_points=None, show_exact=True):
    if show_exact:
        u_test_np = x_test_exact.cpu().detach().numpy()
        XX1, XX2 = np.meshgrid(x, y)
        e = np.reshape(u_test_np, (XX1.shape[0], XX1.shape[1]))
        plt.pcolor(XX1, XX2, e, shading='auto', cmap='jet')
        plt.colorbar()
        plt.title(r'Exact $Velocity$')
    if N_points is not None:
        adds = N_points.cpu().detach().numpy()
        plt.plot(adds[:, [0]], adds[:, [1]], 'kx', markersize=4, clip_on=False)

    points = points.cpu().detach().numpy()
    plt.plot(points[:, [0]], points[:, [1]], 'rx', markersize=4)
    plt.xlim(-0.1, 1.1)
    plt.ylim(-0.1, 1.1)
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$y$', fontsize=20)
    plt.savefig('lid_driven_xnm-WAM-AW.pdf')
    plt.show()


def draw_residual():
    f1, f2, f3 = x_f_loss_fun(x_test, model.train_U)
    f = (abs(f1) + abs(f2) + abs(f3)).cpu().detach().numpy()
    XX1, XX2 = np.meshgrid(x, y)
    e = np.reshape(abs(f), (XX1.shape[0], XX1.shape[1]))
    plt.pcolor(XX1, XX2, e, shading='auto', cmap='jet')
    plt.colorbar()
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$y$', fontsize=20)
    plt.title('$Residual$', fontsize=20)
    plt.tight_layout()
    plt.savefig('lid_driven_residual-WAM-AW.pdf')
    plt.show()


def draw_error():
    predict = model.predict_U(x_test).cpu().detach().numpy()
    u_pred, v_pred, p_pred = predict[:, [0]], predict[:, [1]], predict[:, [2]]
    predict_np = np.sqrt(u_pred ** 2 + v_pred ** 2)
    u_test = x_test_exact.cpu().detach().numpy()
    XX1, XX2 = np.meshgrid(x, y)
    e = np.reshape(abs(predict_np - u_test), (XX1.shape[0], XX1.shape[1]))
    plt.pcolor(XX1, XX2, e, shading='auto', cmap='jet')
    plt.colorbar()
    plt.xlabel('$x$', fontsize=20)
    plt.ylabel('$y$', fontsize=20)
    plt.title('$Error$', fontsize=20)
    plt.tight_layout()
    plt.savefig('lid_driven_error-WAM-AW.pdf')
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


def draw_epoch_w():
    s_collect = np.array(model.s_collect)
    np.savetxt('s_WAM-AW.npy', s_collect)
    plt.rc('legend', fontsize=16)
    plt.yscale('log')
    plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 1]), 'b-', label='$e^{-s_{r}}$')
    plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 2]), 'r-', label='$e^{-s_{b}}$')
    plt.xlabel('$Iters$', fontsize=20)
    plt.ylabel('$\lambda$', fontsize=20)
    plt.legend()
    plt.savefig('lid_driven_S_WAM-AW.pdf', fontsize=20)
    plt.show()


def draw_some_t():
    predict = model.predict_U(x_test).cpu().detach().numpy()
    u_pred, v_pred, p_pred = predict[:, [0]], predict[:, [1]], predict[:, [2]]
    predict_np = np.sqrt(u_pred ** 2 + v_pred ** 2)

    u_test_np = x_test_exact.cpu().detach().numpy()
    XX1, XX2 = np.meshgrid(x, y)
    u_pred = np.reshape(predict_np, (XX1.shape[0], XX1.shape[1]))
    u_test = np.reshape(u_test_np, (XX1.shape[0], XX1.shape[1]))
    gs1 = gridspec.GridSpec(2, 2)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, u_test.T[int((rc1) / (1 / 100)), :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, u_pred.T[int((rc1) / (1 / 100)), :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$y$', fontsize=20)
    ax.set_ylabel('$Velocity$', fontsize=20)
    ax.set_title('$x = ' + str(rc1) + '$', fontsize=20)
    ax.axis('square')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, u_test.T[:, int((rc2) / (1 / 100))], 'b-', linewidth=2, label='Exact')
    ax.plot(x, u_pred.T[:, int((rc2) / (1 / 100))], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$', fontsize=20)
    ax.set_ylabel('$Velocity$', fontsize=20)
    ax.set_title('$y = ' + str(rc2) + '$', fontsize=20)
    ax.axis('square')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])

    ax = plt.subplot(gs1[1, 0])
    ax.plot(x, u_test.T[int((rc3) / (1 / 100)), :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, u_pred.T[int((rc3) / (1 / 100)), :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$y$', fontsize=20)
    ax.set_ylabel('$Velocity$', fontsize=20)
    ax.set_title('$x = ' + str(rc3) + '$', fontsize=20)
    ax.axis('square')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])

    ax = plt.subplot(gs1[1, 1])
    ax.plot(x, u_test.T[:, int((rc4) / (1 / 100))], 'b-', linewidth=2, label='Exact')
    ax.plot(x, u_pred.T[:, int((rc4) / (1 / 100))], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$', fontsize=20)
    ax.set_ylabel('$Velocity$', fontsize=20)
    ax.set_title('$y = ' + str(rc4) + '$', fontsize=20)
    ax.axis('square')
    ax.set_xlim([-0.1, 1.1])
    ax.set_ylim([-0.1, 1.1])

    plt.tight_layout()
    plt.savefig('lid_driven_qie-WAM-AW.pdf')
    plt.show()


if __name__ == '__main__':
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])

    layers = [2, 20, 20, 20, 20, 20, 3]
    net = is_cuda(Net(layers))

    N = 500
    M = 500
    Nbc = 100

    adam_iter, lbgfs_iter = 5000, 50000
    adam_lr, lbgfs_lr = 0.001, 0.5

    model_type = 2  # 0:baseline  1:AM  2:AM_AW

    AM_type = 1  # 0:RAM  1:WAM
    AM_K = 1
    AM_count = 4

    AW_lr = 0.001

    # test data
    rc1 = 0.5
    rc2 = 0.3
    rc3 = -0.5
    rc4 = -0.3

    data = lid_driven.PDE_get_data()
    x = data['x']
    y = data['y']
    X_T, Y_T, u_ref, v_ref, velocity_ref = data['solution']
    X = X_T.T
    Y = Y_T.T

    solution = velocity_ref.T
    x_test = is_cuda(torch.from_numpy(np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))).float())
    x_test_exact = is_cuda(torch.from_numpy(solution.flatten()[:, None]).float())

    # bc data
    x1_boundary_left = torch.cat((torch.zeros([Nbc, 1]), torch.rand([Nbc, 1])), dim=1)
    x1_boundary_right = torch.cat((torch.ones([Nbc, 1]), torch.rand([Nbc, 1])), dim=1)
    x2_boundary_left = torch.cat((torch.rand([Nbc, 1]), torch.zeros([Nbc, 1])), dim=1)
    x2_boundary_right = torch.cat((torch.rand([Nbc, 1]), torch.ones([Nbc, 1])), dim=1)

    x1_boundary_left_label = torch.cat((torch.zeros([Nbc, 1]), torch.zeros([Nbc, 1])), dim=1)
    x1_boundary_right_label = torch.cat((torch.zeros([Nbc, 1]), torch.zeros([Nbc, 1])), dim=1)
    x2_boundary_left_label = torch.cat((torch.zeros([Nbc, 1]), torch.zeros([Nbc, 1])), dim=1)
    x2_boundary_right_label = torch.cat((torch.ones([Nbc, 1]), torch.zeros([Nbc, 1])), dim=1)

    x_bc = is_cuda(torch.cat((x1_boundary_left, x1_boundary_right, x2_boundary_left, x2_boundary_right), dim=0))
    u_bc = is_cuda(torch.cat(
        (x1_boundary_left_label, x1_boundary_right_label, x2_boundary_left_label, x2_boundary_right_label), dim=0))

    model = Model(
        net=net,
        x_label=x_bc,
        x_labels=u_bc,
        x_f_loss_fun=x_f_loss_fun,
        x_test=x_test,
        x_test_exact=x_test_exact,
    )

    model.train()
    print(model.x_test_estimate_collect)

    draw_exact()
    draw_exact_points(model.x_f_M, show_exact=False)
    # draw_exact_points(model.x_f_M, N_points=model.x_f_N)
    # draw_exact_points(model.x_f_M, N_points=model.x_f_N, show_exact=False)
    draw_residual()
    draw_error()
    draw_some_t()
    # draw_epoch_loss()
    draw_epoch_w()
