import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import seaborn as sns
import pandas as pd

class BayesModel:
	def __init__(self, data, mu_prior_range, sigma_prior_range):
		self.data = data
		def pdata(data, mu, sigma):
			r"""
			Probability of :math:`x` given :math:`theta`, assuming :math:`\sigma = 1`.
			"""
			return norm.pdf(data, loc=mu, scale=sigma).prod()
		self.pdata = pdata
		def logpdata(data, mu, sigma):
			return np.log(norm.pdf(data, loc=mu, scale=sigma)).sum()
		self.logpdata = logpdata
		def mupriorprob(mu):
			r"""
			Flat prior on the mean
			"""
			return 1/(mu_prior_range[1] - mu_prior_range[0])
		self.mupriorprob = mupriorprob
		def sigmapriorprob(mu):
			r"""
			Flat prior on the mean
			"""
			return 1/(sigma_prior_range[1] - sigma_prior_range[0])
		self.mupriorprob = mupriorprob
		def logpriorprob(mu, sigma):
			return np.log(mupriorprob(mu)) + np.log(sigmapriorprob(sigma))
		self.logpriorprob = logpriorprob
		def samplemu(size=1):
			r"""
			Sample the parameter from the flat prior on the mean
			"""
			return np.random.uniform(mu_prior_range[0], mu_prior_range[1], size=size)
		self.samplemu = samplemu
		def samplesigma(size=1):
			r"""
			Sample the parameter from the flat prior on the std deviation
			"""
			return np.random.uniform(sigma_prior_range[0], sigma_prior_range[1], size=size)
		self.samplesigma = samplesigma

def metropolis(model, N):
	mu, sigma = model.samplemu(), model.samplesigma()
	mu_posterior = []
	sigma_posterior = []

	for _ in range(int(N)):
		new_mu, new_sigma = model.samplemu(), model.samplesigma()
		
		old_posterior = model.pdata(model.data, mu, sigma)*model.mupriorprob(mu)*model.sigmapriorprob(sigma)
		new_posterior = model.pdata(model.data, new_mu, new_sigma)*model.mupriorprob(new_mu)*model.sigmapriorprob(new_sigma)

		paccept = new_posterior/old_posterior

		if (np.random.uniform(0, 1) < paccept):
			mu = new_mu
			sigma = new_sigma
		mu_posterior.append(mu)
		sigma_posterior.append(sigma)

	return np.array(mu_posterior), np.array(sigma_posterior)

def logmetropolis(model, N):
	mu, sigma       = model.samplemu(), model.samplesigma()
	mu_posterior    = []
	sigma_posterior = []

	for _ in range(int(N)):
		new_mu, new_sigma = model.samplemu(), model.samplesigma()

		logpaccept = np.log(np.random.uniform(0, 1))
		logpcrit   = model.logpdata(model.data, new_mu, new_sigma) + model.logpriorprob(new_mu, new_sigma) \
					 - model.logpdata(model.data, mu, sigma) - model.logpriorprob(mu, sigma)

		if logpaccept < logpcrit:
			mu    = new_mu
			sigma = new_sigma
		mu_posterior.append(mu)
		sigma_posterior.append(sigma)
	return np.array(mu_posterior), np.array(sigma_posterior)



if __name__ == '__main__':
	Ndata = 100
	N = 100

	data = np.random.normal(0.0, 1.0, Ndata)
	mu_prior_range = [-2.0, 2.0]
	sigma_prior_range = [0.0, 3.0]

	bayes = BayesModel(data, mu_prior_range, sigma_prior_range)

	mu_posterior, sigma_posterior = logmetropolis(bayes, N)

	plt.figure()
	df = pd.DataFrame({r'$\mu$': mu_posterior.reshape(1, -1)[0], r'$\sigma$': sigma_posterior.reshape(1, -1)[0]})
	
	sns.set_style("ticks", {"xtick.major.size": 0, "ytick.major.size": 0})

	pp = sns.pairplot(df,
		height=2,
		diag_kind="kde",
		plot_kws=dict(s=10, linewidth=1),
		diag_kws=dict(shade=True))
	pp.axes[0][0].set_xlim(-0.5, 0.5)
	plt.savefig('pairplot.png')

	mu_kde = gaussian_kde(mu_posterior.reshape(1, -1))
	sigma_kde = gaussian_kde(sigma_posterior.reshape(1, -1))

	plt.figure(figsize=(15, 5))
	ax = plt.subplot(131)
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
	ax.set_ylabel(r'$p(\mathrm{data}) \sim \mathcal{N}(0, 1)$')


	ax = plt.subplot(132)
	ax.hist(bayes.samplemu(size=1000),
		bins=20,
		density=True,
		histtype='stepfilled',
		label='Prior',
		alpha=0.3)
	test = np.linspace(-1, 1, 1000)
	ax.plot(test, mu_kde.evaluate(test),
		c='#D1495B',
		ls='-',
		)
	ax.hist(mu_posterior, 
		bins=int(np.sqrt(len(mu_posterior))),
		density=True,
		histtype='stepfilled',
		label='Posterior',
		alpha=0.3)
	ax.legend(fontsize=10,
		loc='upper left',
		title=r'$\hat{\mu} =$' + r'${:.2f} \pm {:.2f},$'.format(np.mean(mu_posterior), np.std(mu_posterior)) + '\n' + r'$N = {}$'.format(len(mu_posterior)),
		title_fontsize=12)
	ax.set_title('Posterior on the Mean')
	ax.set_xlabel(r'$\mu$')
	ax.set_ylabel(r'$p(\mu | \mathrm{data}) \sim \mathcal{N}(0, 1)$')
	ax.set_xlim(-1, 1)

	ax = plt.subplot(133)
	ax.hist(bayes.samplesigma(size=1000),
		bins=100,
		density=True,
		histtype='stepfilled',
		label='Prior',
		alpha=0.3)
	test = np.linspace(0, 2, 1000)
	ax.plot(test, sigma_kde.evaluate(test),
		c='#D1495B',
		ls='-',
		)
	ax.hist(sigma_posterior, 
		bins=int(np.sqrt(len(sigma_posterior))),
		density=True,
		histtype='stepfilled',
		label='Posterior',
		alpha=0.3)
	ax.legend(fontsize=10,
		loc='upper left',
		title=r'$\hat{\sigma} =$' + r'${:.2f} \pm {:.2f},$'.format(np.mean(sigma_posterior), np.std(sigma_posterior)) + '\n' + r'$N = {}$'.format(len(sigma_posterior)),
		title_fontsize=12)
	ax.set_title('Posterior on the Variance')
	ax.set_xlabel(r'$\sigma$')
	ax.set_ylabel(r'$p(\sigma | \mathrm{data}) \sim \mathcal{N}(0, 1)$')
	ax.set_xlim(0.0, 2.0)
	plt.savefig('2dbayes.png')
