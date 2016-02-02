

# rating the aspects



def initialize():
    # mu,Sigma,delta,beta
    return

def e_step(mu,Sigma,delta,beta):
    return alpha

def m_step(alpha):
    return mu,Sigma,delta,beta

def compute_loglikelihood():
    likelihood = 0
    return likelihood

def run_em(maxIter):
    # mu,Sigma,delta,beta = initialize()
    iter = 0
    converge = False
    while iter < maxIter and not converge:
        # old_mu = mu.copy()
        # old_pi = pi.copy()
        # gamma = e_step(mu,Sigma,delta,beta)
        # mu,pi = m_step(alpha)
        # if np.sum(abs(old_mu-mu))/np.sum(abs(old_mu))<0.001:
        #     converge=True
        #     print("EM algorithm converges in "+str(iter+1)+" iterations")
        iter = iter + 1
    if iter == maxIter:
        print("EM algorithm fails to converge in "+str(iter)+" iterations")