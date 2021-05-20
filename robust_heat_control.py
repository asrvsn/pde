'''
Sliding-mode control of heat equation with unknown heat conductivity.
'''

from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pdb

def unit_bump(x):
	y = np.zeros_like(x)
	ix = np.abs(x) < 1
	y[ix] = np.exp(-1/(1-(2*x[ix]-1)**2))
	return y

def saturation(x):
	return np.tanh(20*x)

def rmse(x1, x2):
	return np.sqrt(((x1-x2)**2).mean())

def laplacian(u, dx):
	''' Centered-difference approximation with zero Neumann condition '''
	Lu = np.zeros_like(u)
	Lu -= 2*u
	Lu[:-1] += u[1:]
	Lu[1:] += u[:-1]
	Lu[0] += u[0]
	Lu[-1] += u[-1]
	Lu /= (dx ** 2)
	return Lu

def uncontrolled_system(
		resolution: int,
		real_diffusivity: Callable[[float], float],
		initial: Callable[[float], float],
		tf: float,
		nsamples: int,
	):
	'''
	Diffusion eq. on [0, 1] with zero-flux boundary conditions.
	'''
	assert resolution > 0
	X = np.linspace(0, 1, resolution)
	dx = 1 / resolution

	def f(t, y):
		return real_diffusivity(X) * laplacian(y, dx)

	t0 = 0
	y0 = initial(X)
	t_eval = np.linspace(t0, tf, nsamples)
	sol = solve_ivp(f, (t0, tf), y0, t_eval=t_eval, method='LSODA')
	return X, y0, sol.t, sol.y.T

def controlled_system(
		resolution: int,
		real_diffusivity: Callable[[float], float],
		nominal_diffusivity: float,
		initial: Callable[[float], float],
		desired: Callable[[float], float],
		tf: float,
		nsamples: int,
		tolerance: float=1e-2,
	):
	'''
	Diffusion eq. on [0, 1] with zero-flux boundary conditions.
	'''
	assert resolution > 0
	X = np.linspace(0, 1, resolution)
	dx = 1 / resolution
	yd = desired(X)
	uncertainty = max(np.abs(nominal_diffusivity - real_diffusivity(X)).max(), tolerance)
	# uncertainty = 1
	# pdb.set_trace()

	def f(t, y):
		print(f't: {t}')
		Ly = real_diffusivity(X) * laplacian(y, dx)
		s = y - yd
		print(f'abs error: {np.abs(s).max()}')
		control = -laplacian(y, dx) * nominal_diffusivity - np.abs(laplacian(y, dx)).max()*2*uncertainty*saturation(s)
		dydt = Ly + control
		return dydt

	t0 = 0
	y0 = initial(X)
	t_eval = np.linspace(t0, tf, nsamples)
	sol = solve_ivp(f, (t0, tf), y0, t_eval=t_eval, method='LSODA')
	return X, y0, yd, sol.t, sol.y.T

if __name__ == '__main__':
	# diffusivity = lambda x: 0.1
	diffusivity = lambda b: (lambda x: 0.1 + b*x)
	initial = unit_bump
	desired = lambda x: 0.15+0.03*np.sin(12*x)
	nominal = lambda b: 0.1 + b/2
	disturbances = [0, 0.01, 0.05, 0.075]
	tf = 0.225

	fig, axs = plt.subplots(nrows=len(disturbances)+1, ncols=5, figsize=(12,12))

	X, y0, ts, ys = uncontrolled_system(1000, diffusivity(0.05), initial, tf, 5)
	ymin, ymax = y0.min(), y0.max() + 1e-2
	for i in range(ys.shape[0]):
		if i == 0:
			axs[0][i].set_ylabel(f'uncontrolled\n $B=0.05$')
		axs[0][i].plot(X, ys[i], color='red', label='actual')
		# axs[0][i].plot(X, yd, color='blue', label='desired')
		axs[0][i].set_title(f't = {"{:.2e}".format(ts[i])}')
		# axs[0][i].set_xlabel(f't = {ts[i]}\n area = {ys[i].sum()}')
		axs[0][i].set_ylim(ymin, ymax)


	for j, disturbance in enumerate(disturbances):
		X, y0, yd, ts, ys = controlled_system(1000, diffusivity(disturbance), nominal(disturbance), initial, desired, tf, 5)
		ymin, ymax = y0.min(), y0.max() + 1e-2
		for i in range(ys.shape[0]):
			if i == 0:
				axs[j+1][i].set_ylabel(f'controlled\n $B={disturbance}$')
			axs[j+1][i].plot(X, yd, color='blue', label='reference')
			axs[j+1][i].plot(X, ys[i], color='red', label='actual')
			axs[j+1][i].set_xlabel(f'rmse:{"{:.3e}".format(rmse(ys[i], yd))}')
			# axs[i].set_xlabel(f't = {ts[i]}\n area = {ys[i].sum()}')
			axs[j+1][i].set_ylim(ymin, ymax)

	handles, labels = axs[-1][-1].get_legend_handles_labels()
	fig.legend(handles, labels, loc='lower right', fontsize='x-large', framealpha=1)
	plt.tight_layout()
	plt.savefig('robust_heat.pdf')
	plt.show()





