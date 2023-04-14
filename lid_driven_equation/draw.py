from matplotlib import pyplot as plt

AM_count = [1, 2, 3, 4]

# weight 1:1
PINN = [3.32E-01,3.32E-01,3.32E-01,3.32E-01]
WAM = [7.68e-02	,9.33e-02,	8.67e-02,6.18e-02]
plt.rc('legend', fontsize=16)
plt.yscale('log')
plt.xlabel('$AM-count$', fontsize=20)
plt.ylabel('$L2error$', fontsize=20)
# plt.title('$Error$', fontsize=20)
plt.plot(AM_count, PINN, 'b*-', label='PINN')
plt.plot(AM_count, WAM, 'y*-', label='WAM-AW')
plt.legend()
plt.tight_layout()
plt.savefig('lid_driven_AM_count1.pdf')
plt.show()