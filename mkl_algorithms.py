#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from numpy import linalg as LA



class OMKR:
    def __init__(self, eta, gamma):
        self.eta = eta
        self.gamma = np.array(gamma)
        
    def predict(self, X, w, theta):
        M, N = X.shape
        b = self.gamma.shape[0]
        f_RF_p = np.zeros((b,1))
        for j in range(0,self.gamma.shape[0]):
            if M > 1:
                for k in range(0,M-1):
                    f_RF_p[j,0] = f_RF_p[j,0] + theta[j,k+1]*np.exp(-(LA.norm(X[M-1,:]-X[k,:])**2)/self.gamma[j])
        w_bar = w/np.sum(w)
        f_RF = w_bar.dot(f_RF_p)
        return f_RF, f_RF_p
    
    def update(self, f_RF_p, Y, theta, w):
        l = np.zeros((1,self.gamma.shape[0]))
        for j in range(0,self.gamma.shape[0]):
            l[0,j] = (f_RF_p[j,0]-Y)**2
            w[0,j] = w[0,j]*(.5**(l[0,j]))
        theta = np.concatenate((theta,-self.eta*(f_RF_p-Y*np.ones((self.gamma.shape[0],1)))),axis=1)
        return w, theta


class OMKRFA:
    def __init__(self, rf_feature, eta):
        self.eta = eta
        self.rf_feature = np.array(rf_feature)
        
    def predict(self, X, theta, w):
        M, N = X.shape
        a, n_components, b = self.rf_feature.shape
        f_RF_p = np.zeros((b,1))
        X_f = np.zeros((b,n_components))
        X_features = np.zeros((b,2*n_components))
        for j in range(0,b):
            X_f[j,:] = X.dot(self.rf_feature[:,:,j])
        X_features = (1/np.sqrt(n_components))*np.concatenate((np.sin(X_f),np.cos(X_f)),axis=1)
        for j in range(0,b):
            f_RF_p[j,0] = X_features[j,:].dot(theta[:,j])
        f_RF = w.dot(f_RF_p)
        return f_RF, f_RF_p, X_features
    
    def update(self, f_RF_p, Y, theta, w, X_features):
        b, n_components = X_features.shape
        l = np.zeros((1,b))
        for j in range(0,b):
            theta[:,j] = theta[:,j] - self.eta*(2*(f_RF_p[j,0] - Y)*np.transpose(X_features[j,:]))
            l[0,j] = (f_RF_p[j,0]-Y)**2
            w[0,j] = w[0,j]*(.5**(l[0,j]))
        return w, theta

    

class Raker:
    def __init__(self, lam, rf_feature, eta):
        self.lam = lam
        self.eta = eta
        self.rf_feature = np.array(rf_feature)
        
    def predict(self, X, theta, w):
        M, N = X.shape
        a, n_components, b = self.rf_feature.shape
        f_RF_p = np.zeros((b,1))
        X_f = np.zeros((b,n_components))
        X_features = np.zeros((b,2*n_components))
        for j in range(0,b):
            X_f[j,:] = X.dot(self.rf_feature[:,:,j])
        X_features = (1/np.sqrt(n_components))*np.concatenate((np.sin(X_f),np.cos(X_f)),axis=1)
        for j in range(0,b):
            f_RF_p[j,0] = X_features[j,:].dot(theta[:,j])
        w_bar = w/np.sum(w)
        f_RF = w_bar.dot(f_RF_p)
        return f_RF, f_RF_p, X_features
    
    def update(self, f_RF_p, Y, theta, w, X_features):
        b, n_components = X_features.shape
        l = np.zeros((1,b))
        for j in range(0,b):
            theta[:,j] = theta[:,j] - self.eta*( (2*(f_RF_p[j,0] - Y)*np.transpose(X_features[j,:]))                                                     +2*self.lam*theta[:,j] )
            l[0,j] = (f_RF_p[j,0]-Y)**2+self.lam*(LA.norm(theta[:,j])**2)
            w[0,j] = w[0,j]*np.exp(-self.eta*l[0,j])
        return w, theta
    
    
    
class OMKLGF:
    def __init__(self, lam, rf_feature, gamma, eta, eta_e, M, J):
        self.lam = lam
        self.eta = eta
        self.eta_e = eta_e
        self.rf_feature = np.array(rf_feature)
        self.gamma = np.array(gamma)
        self.M = M
        self.J = J
        
    def graph_gen(self, w):
        p_k = np.zeros((self.gamma.shape[0],self.J))
        p_kk = np.zeros((self.gamma.shape[0],self.J))
        p_c = np.zeros((1,self.J))
        w_bar = w/np.sum(w)
        a, n_components, c = self.rf_feature.shape
        A_t = np.zeros((c,self.J))
        for j in range(0,self.J):
            p_k[:,j:j+1] = (1-self.eta_e**(j+1))*np.transpose(w_bar)+(1/c)*(self.eta_e**(j+1))*np.ones((c,1))
            for k in range(0,c):
                p_kk[k:k+1,j:j+1] = 1-((1-p_k[k:k+1,j:j+1])**self.M)
            for k in range(0,self.M):
                n = 0
                rr = np.random.rand()
                while rr>np.sum(p_k[0:n,j]) and n<c-1:
                    n = n+1
                A_t[n,j] = 1
        u = w.dot(A_t)
        u_bar = u/np.sum(u)
        p_c = (1-self.eta_e)*u_bar+(1/self.J)*self.eta_e*np.ones((1,self.J))
        q = p_kk.dot(np.transpose(p_c))
        return A_t, u, p_c, q
    
    def predict(self, X, theta, w, p_c, A_t):
        m, N = X.shape
        gamma_n = []
        n_n = []
        c = 0
        rr = np.random.rand()
        while rr>np.sum(p_c[0,0:c]) and c<self.J-1:
            c = c+1
        for n in range(0,self.gamma.shape[0]):
            if A_t[n,c]==1:
                gamma_n.append(self.gamma[n])
                n_n.append(n)
        gamma_n = np.array(gamma_n)
        n_n = np.array(n_n)
        a, n_components, c = self.rf_feature.shape
        f_RF_p = np.zeros((self.gamma.shape[0],1))
        X_f = np.zeros((self.gamma.shape[0],n_components))
        X_features = np.zeros((self.gamma.shape[0],2*n_components))
        f_RF_p = np.zeros((self.gamma.shape[0],1))
        for j in n_n:
            X_f[j,:] = X.dot(self.rf_feature[:,:,j])
        X_features = (1/np.sqrt(n_components))*np.concatenate((np.sin(X_f),np.cos(X_f)),axis=1)
        for j in n_n:
            f_RF_p[j,0] = X_features[j,:].dot(theta[:,j])
        w_n = np.zeros((1,self.gamma.shape[0]))
        f_RF_p_n = np.zeros((self.gamma.shape[0],1))
        for j in range(0,gamma_n.shape[0]):
            w_n[0,j] = w[0,n_n[j]]
            f_RF_p_n[j,0] = f_RF_p[n_n[j],0]
        w_bar = w_n/np.sum(w_n)
        f_RF = w_bar.dot(f_RF_p_n)
        return f_RF, f_RF_p, X_features, n_n
    
    def update(self, f_RF_p, Y, theta, w, X_features, n_n, q):
        b = np.floor(np.log2(self.J))
        c, n_components = X_features.shape
        l = np.zeros((1,c))
        for j in n_n:
            theta[:,j] -=  self.eta*( (2*(f_RF_p[j,0] - Y)*np.transpose(X_features[j,:]))\
                                                          +2*self.lam*theta[:,j] )
            l[0,j] = ( (f_RF_p[j,0]-Y)**2+self.lam*(LA.norm(theta[:,j])**2) )/np.sum(q[j,:])
            w[0,j] *= np.exp(-self.eta*(.5**b)*l[0,j])
        return w, theta