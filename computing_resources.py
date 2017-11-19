# In this file you can allocate desired computing resources to be used by data simulation and neural network

from __future__ import unicode_literals
import platform

if platform.system() == 'Darwin':
    CPU_THREADS = 1 # NOTE: There may be some issues with multithreading when simulating data in MacOS
else:
    CPU_THREADS = 8

QUEUE_MAX_SIZE = 4

