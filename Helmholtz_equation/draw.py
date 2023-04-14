from matplotlib import pyplot as plt

AM_count = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# weight 1:1
PINN = [3.02E-02,3.02E-02,3.02E-02,3.02E-02,3.02E-02,3.02E-02,3.02E-02,3.02E-02,3.02E-02,3.02E-02]
RAM = [1.97E-02,	2.42E-03,	1.37E-03,	1.26E-03,	3.37E-03,	2.74E-03,	2.76E-03,	1.86E-03,	1.38E-03,	1.24E-03]
WAM = [1.97E-02	,3.24E-03,	3.86E-03,	1.65E-03,	1.33E-03,	1.60E-03,	1.29E-03,	1.21E-03,	1.23E-03,	1.16E-03]
plt.rc('legend', fontsize=16)
plt.yscale('log')
plt.xlabel('$AM-count$', fontsize=20)
plt.ylabel('$L2error$', fontsize=20)
# plt.title('$Error$', fontsize=20)
plt.plot(AM_count, PINN, 'b*-', label='PINN')
plt.plot(AM_count, RAM, 'r*-', label='RAM-AW')
plt.plot(AM_count, WAM, 'y*-', label='WAM-AW')
plt.legend()
plt.savefig('Helmholtz_AM_count1.pdf')
plt.show()