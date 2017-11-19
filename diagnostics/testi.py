import matplotlib.pyplot as plt
import numpy as np

from data_provider.data_provider import DataProvider

d = DataProvider()

print "uniques for 50:"
print np.unique(d.train_stc.data[:, 50:])

print "\n\n\n"

print "uniques for 100:"
print np.unique(d.train_stc.data[:, 100:])

print "\n\n\n"

print "uniques for 200:"
print np.unique(d.train_stc.data[:, 200:])

print "\n\n\n"

print "uniques for 300:"
print np.unique(d.train_stc.data[:, 300:])

plt.plot(d.train_stc.times, d.train_stc.data.T)
plt.show()