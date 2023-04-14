import numpy as np
from matplotlib import pyplot as plt
from pyDOE import lhs

exact_u = lambda x: 100*np.exp(-10 * (x[:, [0]] ** 2 + x[:, [1]] ** 2))

x1 = np.expand_dims(np.linspace(-1, 1, 256), axis=1)
x2 = np.expand_dims(np.linspace(-1, 1, 256), axis=1)
X1, X2 = np.meshgrid(x1, x2)

x_test_np = np.concatenate((np.vstack(np.expand_dims(X1, axis=2)), np.vstack(np.expand_dims(X2, axis=2))), axis=-1)
solution = exact_u(x_test_np)

e = np.reshape(solution, (X1.shape[0], X2.shape[1]))

plt.pcolor(X1, X2, e, shading='auto', cmap='jet')
plt.colorbar()
plt.xlabel('$x$',fontsize = 20)
plt.ylabel('$y$',fontsize = 20)
plt.title(r'$u(x,y)$',fontsize = 20)
plt.tight_layout()
plt.savefig('fig_2_1.pdf')
plt.show()

M = 1000
lb = np.array([-1.0, -1.0])
ub = np.array([1.0, 1.0])
temp = lb + (ub - lb) * lhs(2, 10000)

u = exact_u(temp)

dx = np.sqrt(1 + u[:,0] ** 2)

err_dx = np.power(dx, 0.5) / np.power(dx, 0.5).mean()
p = (err_dx / sum(err_dx))
X_ids = np.random.choice(a=len(temp), size=M, replace=False, p=p)
x_M = temp[X_ids]

plt.plot(x_M[:, [0]], x_M[:, [1]], 'rx', markersize=4, clip_on=False)
plt.xlabel('$x$',fontsize = 20)
plt.ylabel('$y$',fontsize = 20)
plt.axis('square')
plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
plt.title('$k=0.5$',fontsize = 20)
plt.tight_layout()
plt.savefig('fig_2_2.eps')
plt.show()


err_dx = np.power(dx, 1) / np.power(dx, 1).mean()
p = (err_dx / sum(err_dx))
X_ids = np.random.choice(a=len(temp), size=M, replace=False, p=p)
x_M = temp[X_ids]

plt.plot(x_M[:, [0]], x_M[:, [1]], 'rx', markersize=4, clip_on=False)
plt.xlabel('$x$',fontsize = 20)
plt.ylabel('$y$',fontsize = 20)
plt.axis('square')
plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
plt.title('$k=1$',fontsize = 20)
plt.tight_layout()
plt.savefig('fig_2_3.eps')
plt.show()


err_dx = np.power(dx, 2) / np.power(dx, 2).mean()
p = (err_dx / sum(err_dx))
X_ids = np.random.choice(a=len(temp), size=M, replace=False, p=p)
x_M = temp[X_ids]

plt.plot(x_M[:, [0]], x_M[:, [1]], 'rx', markersize=4, clip_on=False)
plt.xlabel('$x$',fontsize = 20)
plt.ylabel('$y$',fontsize = 20)
plt.axis('square')
plt.xlim([-1.1, 1.1])
plt.ylim([-1.1, 1.1])
plt.title('$k=2$',fontsize = 20)
plt.tight_layout()
plt.savefig('fig_2_4.eps')
plt.show()




