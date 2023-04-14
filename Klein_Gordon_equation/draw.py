from matplotlib import pyplot as plt

AM_count = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# weight 1:1
PINN = [1.76E-01,1.76E-01,1.76E-01,1.76E-01,1.76E-01,1.76E-01,1.76E-01,1.76E-01,1.76E-01,1.76E-01]
RAM = [2.22E-02	,2.75E-03,	2.57E-03,	2.02E-03,	1.96E-03,	2.15E-03,	2.02E-03,	1.63E-03,	1.69E-03,	1.94E-03]
WAM = [2.22E-02,	3.96E-03,	2.68E-03,	2.57E-03,	2.53E-03,	2.01E-03,	2.35E-03,	2.74E-03,	1.99E-03,	1.91E-03]
plt.rc('legend', fontsize=16)
plt.yscale('log')
plt.xlabel('$AM-count$', fontsize=20)
plt.ylabel('$L2error$', fontsize=20)
# plt.title('$Error$', fontsize=20)
plt.plot(AM_count, PINN, 'b*-', label='PINN')
plt.plot(AM_count, RAM, 'r*-', label='RAM-AW')
plt.plot(AM_count, WAM, 'y*-', label='WAM-AW')
plt.legend()
plt.savefig('KG_AM_count1.pdf')
plt.show()