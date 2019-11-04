import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def p(k):
	if k == 0:
		return (3/(np.power(np.pi, 2) + 3))
	else:
		return (3/(np.power(np.pi, 2) + 3))*np.power(k, -2.0)

def metropolis(N):
	# Choose a random integer starting at zero
	k = 0
	sample = []
	for _ in range(int(N)):
		# Choose neighbour with probability 1/2
		if np.random.uniform(0, 1) < 0.5:
			neighbour = k + 1
		else:
			neighbour = k - 1
		# Go to neighbour deterministically if p(neighbour) > p(k)
		if p(neighbour) >= p(k):
			k = neighbour
		else:
			# Go to neighbour with probability p(neighbour)/p(k)
			if np.random.uniform(0, 1) < p(neighbour)/p(k):
				k = neighbour
		sample.append(k)
	return np.array(sample)

if __name__ == '__main__':
	N, Ns = 1e3, 1e3

	sample = np.concatenate(Parallel(n_jobs=-1)(delayed(metropolis)(N=N) for _ in range(int(Ns))))

	max_abs = max(np.abs(np.min(sample)), np.abs(np.max(sample)))
	test = np.arange(-max_abs, max_abs, 1.0)
	p_test = []
	for t in test:
		p_test.append(p(t))
	p_test = np.array(p_test)


	plt.figure()
	plt.scatter(test, p_test,
		c='#D1495B')
	plt.hist(sample, 
		bins=np.arange(np.min(sample), np.max(sample), 1.0),
		density=True, 
		alpha=0.8,
		histtype='step',
		align='left',
		label=r'$N = 10^4, N_s = 10^2$')
	plt.legend(fontsize=12)
	plt.xlabel(r'$k$')
	plt.ylabel(r'$p(k)$')
	plt.xlim(-max_abs, max_abs)
	plt.yscale('log')
	plt.savefig('betterN.png')
