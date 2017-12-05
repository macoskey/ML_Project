# Make random walk simulator for procedurally generating simulated stocks
#
#

import numpy as np
import matplotlib.pyplot as plt
from random import *

n = 1000
mu = 0.1
val = np.zeros(n)
val[0] = 50

for i in range(1,n):
   # r = uniform(1,10)
   # if(r > 5):
   #     val[i] = val[i-1] + uniform(1,10)
   # else:
   #     val[i] = val[i-1] - 1
   val[i] = val[i-1] + gauss(mu,15)
   if(val[i] < 0):
       val[i] = 0



f, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(val)
axarr[0].set_title('Price')
axarr[1].plot(np.diff(val))
axarr[1].set_title('diff(price)')
plt.show()
