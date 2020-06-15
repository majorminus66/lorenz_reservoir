import numpy as np
from matplotlib import pyplot as plt
#from lorenz import get_u

a = 10
b = 28
c = 8/3
x0 = 1
y0 = 1
z0 = 1
t = 0.01

def dxdt (x,y):
  return (-a*x + a*y)

def dydt (x,y,z):
  return (b*x - y - x*z)

def dzdt (x,y,z):
  return (-c*z + x*y)

def get_u (x0, y0, z0, a, b, c, t):
    xarr = np.array([x0])
    yarr = np.array([y0])
    zarr = np.array([z0])
    
    for i in range(0, 10000):
      xarr = np.append(xarr, (t*dxdt(xarr[i], yarr[i]) + xarr[i]))
      yarr = np.append(yarr, (t*dydt(xarr[i], yarr[i], zarr[i]) + yarr[i]))
      zarr = np.append(zarr, (t*dzdt(xarr[i], yarr[i], zarr[i]) + zarr[i]))

    u = np.stack([xarr, yarr, zarr], axis = 0)
    
    return u

xarr = np.array([x0])
yarr = np.array([y0])
zarr = np.array([z0])

for i in range(0, 1000):
  xarr = np.append(xarr, (t*dxdt(xarr[i], yarr[i]) + xarr[i]))
  yarr = np.append(yarr, (t*dydt(xarr[i], yarr[i], zarr[i]) + yarr[i]))
  zarr = np.append(zarr, (t*dzdt(xarr[i], yarr[i], zarr[i]) + zarr[i]))

times = np.array(range(0,1001))

plt.plot(times, xarr)
plt.plot(times, yarr)
plt.plot(times, zarr)
#plt.plot(xarr, zarr)

u = np.stack([xarr, yarr, zarr], axis = 0)
print(u.shape)
#print(u[:,0].shape)


    