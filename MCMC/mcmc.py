import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
import seaborn as sns
from scipy.stats import gaussian_kde

class DoubleGaussian:
	def __init__(self, mu=0.0, sigma=1.0):
		self.mu = mu
		self.sigma = sigma
		def normal_dist(x):
			return 1/(2*np.sqrt(2*np.pi)*sigma)*np.exp(-np.power(x - mu, 2)/(2*np.power(sigma, 2))) \
				 + 1/(2*np.sqrt(2*np.pi)*sigma)*np.exp(-np.power(x + mu, 2)/(2*np.power(sigma, 2)))
		self.evaluate = normal_dist

def metropolis(dist, N):
	x = np.random.uniform(-1, 1)
	p = dist(x)
	pts = []
	for i in range(N):
		xn = x + np.random.uniform(-1, 1)
		pn = dist(xn)
		if pn >= p:
			p = pn
			x = xn
		else:
			u = np.random.rand()
			if u < pn/p:
				p = pn
				x = xn
		pts.append(x)
	pts = np.array(pts)
	return pts


if __name__ == '__main__':

    plt.figure()

    N = 500

    mu = 0.0
    sigma = 1.0

    pdf = DoubleGaussian(mu=mu, sigma=sigma)
    print('Mean:\t', pdf.mu)
    print('Sigma:\t', pdf.sigma)
    print('p(0):\t', '{:.3f}'.format(pdf.evaluate(x=0.0)))

    pts1 = np.concatenate(Parallel(n_jobs=-1)(delayed(metropolis)(dist=pdf.evaluate, N=N) \
    										  for _ in range(500)))

    # kernel = gaussian_kde(pts1)
    test = np.linspace(-5, 5, 1000)
    # plt.plot(test, kernel.evaluate(test), 
    # 	ls='--',
    # 	c='k',
    # 	lw=1.0)
    plt.plot(test, pdf.evaluate(test),
    	ls='--',
    	c='#D1495B',
    	lw=1.0,
        label=r'$\mathrm{True}\,\,\mathrm{Distr.}$')

    plt.hist(pts1, 
    	bins=20, 
    	density=True, 
    	alpha=1.0,
    	label=r'$\mu = {:.1f}, \sigma = {:.1f}$'.format(mu, sigma),
    	histtype='step'
    	)

    print('\n')

    mu = 2.0
    sigma = 0.5

    pdf = DoubleGaussian(mu=mu, sigma=sigma)
    print('Mean:\t', pdf.mu)
    print('Sigma:\t', pdf.sigma)
    print('p(0):\t', '{:.3f}'.format(pdf.evaluate(x=0.0)))

    pts2 = np.concatenate(Parallel(n_jobs=-1)(delayed(metropolis)(dist=pdf.evaluate, N=N) \
    										  for _ in range(500)))

    # kernel = gaussian_kde(pts2)
    test = np.linspace(-5, 5, 1000)
    # plt.plot(test, kernel.evaluate(test), 
    # 	ls='--',
    # 	c='k',
    # 	lw=1.0)
    plt.plot(test, pdf.evaluate(test),
    	ls='--',
    	c='#D1495B',
    	lw=1.0)

    plt.hist(pts2, 
    	bins=20, 
    	density=True, 
    	alpha=1.0, 
    	label=r'$\mu = {:.1f}, \sigma = {:.1f}$'.format(mu, sigma),
    	histtype='step'
    	)

    plt.xlabel(r'$x$')
    plt.ylabel(r'$p(x)$')
    plt.legend(fontsize=10, loc='best', title='$N = {}$'.format(N), title_fontsize=12)
    plt.xlim(-5, 5)
    plt.ylim(0, 0.6)
    #plt.title(r'MCMC: Double Gaussian')
    plt.savefig('MCMC_good.png')

    sns.jointplot(pts1, pts2,
    	kind='hex',
    	xlim=(-4,4),
    	ylim=(-4,4),
    	color='#D1495B'

    	).set_axis_labels(r'$\mathcal{N}_2(0, 1)$', r'$\mathcal{N}_2(2, 0.5)$')
    plt.savefig('MCMCjoint_good.png')