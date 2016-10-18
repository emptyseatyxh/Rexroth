# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 21:14:54 2016

@author: Xinghao
"""

import numpy as np
import os
from shutil import copyfile
from tables import *
import h5py

# copy and write file in directory
a = np.random.randint(1, 100, (199, 4))
try:
    os.makedirs(r"C:\Users\Xinghao\Desktop\dsa")
except FileExistsError:
    pass
copyfile(r"C:\Users\Xinghao\Desktop\lohr-stu.pdf", r"C:\Users\Xinghao\Desktop\dsa\test.pdf")

with open(r"C:\Users\Xinghao\Desktop\dsa\fw.dat", "wb") as fw:
    np.savetxt(fw, a)
    
    
# save large numpy array into h5py
large = np.random.rand(4000, 400)
f = h5py.File(r"C:\Users\Xinghao\Desktop\mytestfile.hdf5", "w")
f.create_dataset('large_array', data=large, dtype=large.dtype)
f.close()


# work with PyTable for table like array or data
class Particle(IsDescription):
    name      = StringCol(16)   # 16-character String
    idnumber  = Int64Col()      # Signed 64-bit integer
    ADCcount  = UInt16Col()     # Unsigned short integer
    TDCcount  = UInt8Col()      # unsigned byte
    grid_i    = Int32Col()      # 32-bit integer
    grid_j    = Int32Col()      # 32-bit integer
    pressure  = Float32Col()    # float  (single-precision)
    energy    = Float64Col()    # double (double-precision)

h5file = open_file(r"C:\Users\Xinghao\Desktop\tutorial1.h5", mode = "w", title = "Test file")
group = h5file.create_group("/", 'detector', 'Detector information')
table = h5file.create_table(group, 'readout', Particle, "Readout example")

particle = table.row
for i in range(10):
    particle['name']  = 'Particle: %6d' % (i)
    particle['TDCcount'] = i % 256
    particle['ADCcount'] = (i * 256) % (1 << 16)
    particle['grid_i'] = i
    particle['grid_j'] = 10 - i
    particle['pressure'] = float(i*i)
    particle['energy'] = float(particle['pressure'] ** 4)
    particle['idnumber'] = i * (2 ** 34)
    # Insert a new particle record
    particle.append()

table.flush() #  write all this data to disk

table2 = h5file.root.detector.readout
pressure = [x['pressure'] for x in table2.iterrows() \
            if x['TDCcount'] > 3 and 20 <= x['pressure'] < 50]
names = [ x['name'] for x in table.where(
            """(TDCcount > 3) & (20 <= pressure) & (pressure < 50)""") ] # much better
            
gcolumns = h5file.create_group(h5file.root, "column", "save b ")
h5file.create_array(gcolumns, 'pressure', pressure, "Pressure column selection")
h5file.create_array(gcolumns, 'name', names, "Name column selection")

print(h5file)
h5file.close()


