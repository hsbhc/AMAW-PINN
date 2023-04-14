import numpy as np
from matplotlib import pyplot as plt

AM_count = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# weight 1:100
PINN = [6.18E-01, 6.18E-01, 6.18E-01, 6.18E-01, 6.18E-01, 6.18E-01, 6.18E-01, 6.18E-01, 6.18E-01, 6.18E-01]
RAM = [6.18E-01, 1.31E-01, 6.91E-02, 9.61E-02, 3.35E-02, 6.18E-02, 1.96E-02, 4.54E-02, 2.35E-02, 2.31E-02]
WAM = [6.18E-01, 1.73E-01, 8.92E-02, 8.68E-02, 6.29E-02, 4.71E-02, 6.27E-02, 8.27E-02, 3.32E-02, 4.53E-02]

# plt.yscale('log')
plt.rc('legend', fontsize=16)
plt.xlabel('$AM-count$', fontsize=20)
plt.ylabel('$L2error$', fontsize=20)
# plt.title('$Error$', fontsize=20)
plt.plot(AM_count, PINN, 'b*-', label='PINN')
plt.plot(AM_count, RAM, 'r*-', label='RAM')
plt.plot(AM_count, WAM, 'y*-', label='WAM')
plt.legend()
plt.savefig('2dpossion_AM_count1.pdf')
plt.show()

# weight 1:1
PINN = [1.30E+00, 1.30E+00, 1.30E+00, 1.30E+00, 1.30E+00, 1.30E+00, 1.30E+00, 1.30E+00, 1.30E+00, 1.30E+00]
WAM1000 = [1.30E+00, 3.01E-01, 1.43E-01, 2.65E-01, 2.05E-01, 1.96E-01, 1.43E-01, 1.90E-01, 2.12E-01, 2.06E-01]
WAM1000_AW = [1.00E+00, 5.91E-02, 8.46E-02, 6.41E-02, 6.26E-02, 5.51E-02, 4.84E-02, 5.91E-02, 4.46E-02, 3.59E-02]

# plt.yscale('log')
plt.rc('legend', fontsize=16)
plt.xlabel('$AM-count$', fontsize=20)
plt.ylabel('$L2error$', fontsize=20)
# plt.title('$Error$', fontsize=20)
plt.plot(AM_count, PINN, 'b*-', label='PINN')
plt.plot(AM_count, WAM1000, 'r*-', label='WAM')
plt.plot(AM_count, WAM1000_AW, 'y*-', label='WAM-AW')
plt.legend()
plt.savefig('2dpossion_AMW_count1.pdf')
plt.show()

# weight 1:1
PINN = [8.52E-01, 8.52E-01, 8.52E-01, 8.52E-01, 8.52E-01, 8.52E-01, 8.52E-01, 8.52E-01, 8.52E-01, 8.52E-01]
RAM500 = [8.52E-01, 5.93E-01, 1.60E-01, 4.16E-01, 1.15E-01, 1.29E-01, 7.18E-02, 1.87E-01, 9.23E-02, 1.37E-01]
RAM500_AW = [9.06E-01, 1.12E-01, 1.07E-01, 6.12E-02, 3.73E-02, 3.86E-02, 3.30E-02, 3.22E-02, 1.88E-02, 1.69E-02]

# plt.yscale('log')
plt.rc('legend', fontsize=16)
plt.xlabel('$AM-count$', fontsize=20)
plt.ylabel('$L2error$', fontsize=20)
# plt.title('$Error$', fontsize=20)
plt.plot(AM_count, PINN, 'b*-', label='PINN')
plt.plot(AM_count, RAM500, 'r*-', label='RAM')
plt.plot(AM_count, RAM500_AW, 'y*-', label='RAM-AW')
plt.legend()
plt.savefig('2dpossion_AMW_count2.pdf')
plt.show()


s_collect = np.loadtxt('s_WAM-AW.npy')
plt.rc('legend', fontsize=16)

plt.yscale('log')
plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 1]), 'b-', label='$e^{-s_{r}}$')
plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 2]), 'r-', label='$e^{-s_{b}}$')
plt.xlabel('$Iters$', fontsize=20)
plt.ylabel('$\lambda$', fontsize=20)
plt.legend()
plt.savefig('2dpossion_S_WAM-AW.pdf')
plt.show()


s_collect = np.loadtxt('s_RAM-AW.npy')
plt.rc('legend', fontsize=16)
plt.yscale('log')
plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 1]), 'b-', label='$e^{-s_{r}}$')
plt.plot(s_collect[:, 0], np.exp(-s_collect[:, 2]), 'r-', label='$e^{-s_{b}}$')
plt.xlabel('$Iters$', fontsize=20)
plt.ylabel('$\lambda$', fontsize=20)
plt.legend()
plt.savefig('2dpossion_S_RAM-AW.pdf')
plt.show()
