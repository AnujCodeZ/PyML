import numpy as np 


def fit_params(x):
    
    mu = np.mean(x, axis=0)
    sigma = np.mean(np.dot((x - mu), (x - mu).T))
    
    return mu, sigma

def gaussian_dist(x, mu, sigma):
    
    n = x.shape[1]
    
    det = np.linalg.det(sigma)
    inv = np.linalg.inv(sigma)
    
    exp = np.exp((-1/2) * np.sum((x - mu).T @ inv * (x - mu), axis=1))
    prior = np.power(2*np.pi, -(n/2)) * np.power(det, -(1/2))
    
    return prior * exp

def select_threshHold(yval, pval):
    
    F1 = 0
    bestF1 = 0
    best_epsilon = 0
    
    stepsize = (np.max(pval) - np.min(pval))/1000
        
    eps_vec = np.arange(np.min(pval), np.max(pval), stepsize)
    noe = len(eps_vec)
    
    for eps in range(noe):
        epsilon = eps_vec[eps]
        pred = (pval < epsilon)
        prec, rec = 0,0
        tp,fp,fn = 0,0,0
        
        try:
            for i in range(np.size(pval,0)):
                if pred[i] == 1 and yval[i] == 1:
                    tp+=1
                elif pred[i] == 1 and yval[i] == 0:
                    fp+=1
                elif pred[i] == 0 and yval[i] == 1:
                    fn+=1
            prec = tp/(tp + fp)
            rec = tp/(tp + fn)
            F1 = 2*prec*rec/(prec + rec)
            if F1 > bestF1:
                bestF1 = F1
                best_epsilon = epsilon
        except ZeroDivisionError:
            print('Warning dividing by zero!!')          
       
    return bestF1, best_epsilon

# If p(new example) < best_epsilon then it is outlier