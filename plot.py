import numpy as np
import matplotlib.pyplot as plt

# 0.4884  4.2772
# 3.2637 10.841
# 12.65  27.81
# 24.677 39.83
# 82.24  104.97
x = [0.4884, 3.2637, 12.65, 24.677, 82.24]
# x = [4.2772, 10.841, 27.81, 39.83]

# 0.3148 0.3161
# 0.1443 0.1536
# 0.0891 0.0985
# 0.0732 0.0763
# 0.0586 0.0622
# y = [0.3148, 0.1443, 0.0891, 0.0732, 0.0586]
y = [0.3161, 0.1536, 0.0985, 0.0763, 0.0622]

log_x = np.log(x)
log_y = np.log(y)

R2 = np.corrcoef(log_x, log_y)[0, 1]
print(R2)


plt.scatter(x, y)
# set scale to log
plt.xscale('log')
plt.yscale('log')
plt.savefig('plot.png')