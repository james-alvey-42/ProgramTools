import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

class BayesModel:
	def __init__(self, data, prior_range):
		self.data = data
		def pdata(data, mu):
			r"""
			Probability of :math:`x` given :math:`theta`, assuming :math:`\sigma = 1`.
			"""
			return norm.pdf(data, loc=mu).prod()
		self.pdata = pdata
		def priorprob(mu):
			r"""
			Flat prior on the mean
			"""
			return 1/(prior_range[1] - prior_range[0])
		self.priorprob = priorprob

		def samplemu(size=1):
			r"""
			Sample the parameter from the flat prior on the mean
			"""
			return np.random.uniform(prior_range[0], prior_range[1], size=size)
		self.samplemu = samplemu

def metropolis(model, N):
	mu = model.samplemu()
	posterior = []

	for _ in range(int(N)):
		new_mu = model.samplemu()
		
		old_posterior = model.pdata(data, mu)*model.priorprob(mu)
		new_posterior = model.pdata(data, new_mu)*model.priorprob(new_mu)

		paccept = new_posterior/old_posterior

		if (np.random.uniform(0, 1) < paccept):
			mu = new_mu
		posterior.append(mu)

	return np.array(posterior)

if __name__ == '__main__':
	Ndata = 100
	N = 10000

	data = np.random.normal(0.0, 1.0, Ndata)
	prior_range = [-2.0, 2.0]

	bayes = BayesModel(data, prior_range)

	posterior = metropolis(bayes, N=N)

	plt.figure(figsize=(10, 5))
	ax = plt.subplot(121)
	ax.hist(data,
		bins=max(10, int(np.sqrt(len(data)))),
		color='k',
		density=True,
		histtype='stepfilled',
		label='Data',
		alpha=0.4)
	ax.legend(fontsize=10,
		loc='upper left',
		title_fontsize=12,
		title=r'$N_{\mathrm{data}} =$' + r'${}$'.format(len(data)))
	ax.set_title('Data')
	ax.set_xlabel(r'$x$')
	ax.set_ylabel(r'$p(\mathrm{data} \sim \mathcal{N}(0, 1)$')


	ax = plt.subplot(122)
	ax.hist(bayes.samplemu(size=1000),
		bins=100,
		density=True,
		histtype='stepfilled',
		label='Prior',
		alpha=0.3)
	ax.hist(posterior, 
		bins=int(np.sqrt(len(posterior))),
		density=True,
		histtype='stepfilled',
		label='Posterior',
		alpha=0.3)
	ax.legend(fontsize=10,
		loc='upper left',
		title=r'$\hat{\mu} =$' + r'${:.2f} \pm {:.2f},$'.format(np.mean(posterior), np.std(posterior)) + '\n' + r'$N = {}$'.format(len(posterior)),
		title_fontsize=12)
	ax.set_title('Posterior Distribution')
	ax.set_xlabel(r'$\mu$')
	ax.set_ylabel(r'$p(\mu | \mathrm{data} \sim \mathcal{N}(0, 1)$')
	plt.savefig('bayes.png')
