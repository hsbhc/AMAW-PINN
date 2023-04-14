import numpy as np
from matplotlib import pyplot as plt
#
# s_collect = np.loadtxt('s_WAM-AW.npy')
# plt.rc('legend', fontsize=16)
# plt.yscale('log')
# plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 1]), 'b-', label='$e^{-s_{r}}$')
# plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 2]), 'r-', label='$e^{-s_{i}}$')
# plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 3]), 'g-', label='$e^{-s_{b}}$')
# plt.xlabel('$Iters$', fontsize=20)
# plt.ylabel('$\lambda$', fontsize=20)
# plt.legend()
# plt.savefig('burgers_S_WAM-AW.pdf', fontsize=20)
# plt.show()
#
# s_collect = np.loadtxt('s_RAM-AW.npy')
# plt.rc('legend', fontsize=16)
# plt.yscale('log')
# plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 1]), 'b-', label='$e^{-s_{r}}$')
# plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 2]), 'r-', label='$e^{-s_{i}}$')
# plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 3]), 'g-', label='$e^{-s_{b}}$')
# plt.xlabel('$Iters$', fontsize=20)
# plt.ylabel('$\lambda$', fontsize=20)
# plt.legend()
# plt.savefig('burgers_S_RAM-AW.pdf', fontsize=20)
# plt.show()


AM_count = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# weight 1:1
PINN = [2.25E-02, 2.25E-02, 2.25E-02, 2.25E-02, 2.25E-02, 2.25E-02, 2.25E-02, 2.25E-02, 2.25E-02, 2.25E-02]
RAM = [3.06E-02	,1.31E-03,	1.26E-03,	5.77E-04,	5.95E-04,	3.32E-04,	4.55E-04,	3.09E-04,	3.40E-04,	5.34E-04]
WAM = [3.06E-02,	7.80E-04,	1.17E-03,	1.21E-03,	7.63E-04,	5.68E-04,	5.64E-04,	1.22E-03,	8.12E-04,	5.15E-04]

plt.rc('legend', fontsize=16)
plt.yscale('log')
plt.xlabel('$AM-count$', fontsize=20)
plt.ylabel('$L2error$', fontsize=20)
# plt.title('$Error$', fontsize=20)
plt.plot(AM_count, PINN, 'b*-', label='PINN')
plt.plot(AM_count, RAM, 'r*-', label='RAM-AW')
plt.plot(AM_count, WAM, 'y*-', label='WAM-AW')
plt.legend()
plt.savefig('burgers_AM_count1.pdf')
plt.show()
