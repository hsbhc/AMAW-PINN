import numpy as np
from matplotlib import pyplot as plt

all_error_residual_name = ['PINN', 'Random', 'RAM', 'WAM']
all_Nf = [30, 50, 70, 90, 110]
all_error = [
    [0.2213068, 0.2971648, 0.16465099, 0.05506437],

    [0.21227087, 0.12158904, 0.01065355, 0.00188192],

    [0.00694599, 0.02824025, 0.00446645, 0.00556956],

    [0.09327888, 0.00277377, 0.00244714, 0.00348987],

    [0.00613681, 0.00579698, 0.00439695, 0.00403545]]
all_residual = [
    [3.05417559e+04, 2.37773203e+04, 7.58422394e+01, 1.81504917e+01],

    [1.87683118e+03, 7.71087885e+00, 3.99608314e-01, 2.26613849e-01],

    [3.35746841e+01, 3.95387955e+01, 1.02786086e-01, 1.14581034e-01],

    [1.09376669e+01, 3.17952323e+00, 7.47850090e-02, 1.58884600e-01],

    [3.92104411e+00, 4.94067287e+00, 1.36464939e-01, 7.15938359e-02]]

all_error = np.array(all_error)
all_residual = np.array(all_residual)

plt.rc('legend', fontsize=16)
plt.yscale('log')
plt.plot(all_Nf, all_error[:, 0], 'k-', label=all_error_residual_name[0])
plt.plot(all_Nf, all_error[:, 1], 'b-',
         label=all_error_residual_name[1])
plt.plot(all_Nf, all_error[:, 2], 'y-', label=all_error_residual_name[2])
plt.plot(all_Nf, all_error[:, 3], 'r-',
         label=all_error_residual_name[3])
plt.xlabel('$N_{r}$',fontsize = 20)
plt.ylabel('$Test-L2error$',fontsize = 20)
# plt.legend()
plt.savefig('1dpossion_L2error.pdf')
plt.show()

plt.rc('legend', fontsize=16)
plt.yscale('log')
plt.plot(all_Nf, all_residual[:, 0], 'k-', label=all_error_residual_name[0])
plt.plot(all_Nf, all_residual[:, 1], 'b-',
         label=all_error_residual_name[1])
plt.plot(all_Nf, all_residual[:, 2], 'y-', label=all_error_residual_name[2])
plt.plot(all_Nf, all_residual[:, 3], 'r-',
         label=all_error_residual_name[3])

plt.xlabel('$N_{r}$',fontsize = 20)
plt.ylabel('$Test-residual$',fontsize = 20)
plt.legend()
plt.savefig('1dpossion_residual.pdf')
plt.show()

print(all_Nf)
print(all_error)
print(all_residual)
