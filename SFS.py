# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 10:07:53 2022

@author: jason
"""
import numpy as np
import matplotlib.pyplot as plt
import itertools
from tqdm import tqdm
import scipy.optimize

#Simulation functions
#======================
def pmap(rho):
    return rho/(1+rho)

def rhomap(p):
    return p/(1-p)

def p_prime_sel_opt(p,delt_opt,a,sign,V_s):
    S=1/V_s
    p=pmap(rhomap(p)*np.exp(S*a*sign*(delt_opt+0.5*a*sign*(2*p-1))))    
    return p

#####################
#%%%
#Single core simulation

Vg_sims={}

L=100
rep=1

sigma_e2s=np.array([1e-2])

Ns=np.array([10000])
Vs=np.array([5])
mus=np.array([6.6e-6])
a=np.sqrt(0.1)

maxiter=int(10*Ns[0])
phist={str(_):np.zeros([maxiter,L,rep]) for _ in sigma_e2s}

for sigma_e2,N,V_s,mu in itertools.product(sigma_e2s,Ns,Vs,mus):
    #print(sigma_e2,N,V_s,mu,alpha**2/(2*V_s))
    sign=2*np.random.randint(0,2,[L,rep])-1
    
    opt=np.zeros(rep)
    
    p=np.zeros([L,rep]) 
    
    opt_hist=np.zeros([maxiter,rep])
    zbar_hist=np.zeros([maxiter,rep])
    numfix_hist=np.zeros([maxiter,rep]) 
    
    for t in tqdm(range(maxiter)):
        
        phist[str(sigma_e2)][t]=np.array(p)
        
        fixed_loci_1=(p==1)
        
        #Reset fixed loci and remove them from optimum
        p[fixed_loci_1]=0
        opt=opt-2*a*np.sum(sign*fixed_loci_1,0)
        
        zbar=np.sum(2*a*sign*p**2+a*sign*p*(1-p),0)
        
        
        fixed_loci_0=(p==0)
        mutation_mask=((np.random.rand(L,rep)<N*mu) & fixed_loci_0)
        
        zbar_hist[t]=zbar
        opt_hist[t]=opt
        numfix_hist[t]=np.sum(fixed_loci_1,0)
        
        np.place(p,mutation_mask,1/N)
        np.place(sign,mutation_mask,2*np.random.randint(0,2,np.sum(mutation_mask))-1)
        
        #new mutants excluded 
        poly_loci=np.logical_not(fixed_loci_0) & (p<1-1/N)        

        #p[poly_loci]=p[poly_loci]+0*mu*(1-2*p[poly_loci])
        p[poly_loci]=p[poly_loci]+\
            (np.random.rand(np.sum(poly_loci))<N*mu*(1-p[poly_loci]))/N-\
                (np.random.rand(np.sum(poly_loci))<N*mu*p[poly_loci])/N
        
        p=np.random.binomial(N,p_prime_sel_opt(p,opt-zbar,a,sign,V_s))/N
        
        opt=np.random.normal(opt,np.sqrt(sigma_e2),rep)
    
    Vg_sims[str([sigma_e2,N,V_s,mu])]=2*a**2*np.sum(p*(1-p),0)

#np.savetxt('hist_000_'+str(L)+'.txt',phist['0.0'][:,:,0])
#np.savetxt('hist_001_'+str(L)+'.txt',phist['0.01'][:,:,0])
#np.savetxt('delta_hist_001_'+str(L)+'.txt',opt_hist-zbar_hist)


#%%
#Genetic variance versus sigma_e2

params=np.array(list(itertools.product(sigma_e2s,Ns,Vs,mus)))

for N in Ns:
    for V_s in Vs:
        for mu in mus:
            plt.figure()

            offset=1e-7
            #Vg_mean=np.array([np.mean(Vg_sims[str([_,N,V_s,mu])]) for _ in sigma_e2s])

            plt.violinplot([Vg_sims[str([_,N,V_s,mu])]/(Vg_sims[str([_,N,V_s,mu])]+1) for _ in sigma_e2s],positions=np.log10(sigma_e2s+offset),widths=2e-1,showmeans=True)

            print((2.*V_s)**-1*Vm/(2*L*mu))

            #Environmental variance set to 1
            #h_2=Vg_mean/(Vg_mean+1)
            #plt.plot(sigma_e2s,h_2,label=str(N)+','+str(V_s)+','+str(mu))

            # gam=np.sqrt(Vm/(2*L*mu))
            # #Vg_theory=0.5*2*L*mu*V_s*(1+np.sqrt(1+sigma_e2s/(V_s*L**2*mu**2)))
            # Vg_theory=np.array([scipy.optimize.fsolve(lambda x: Vg_theory_opt(x,N,gam,_,L,mu,V_s),0.025)[0] for _ in sigma_e2s])
            # plt.plot(sigma_e2s, Vg_theory/(Vg_theory+1))


            # Vg_numerical=np.array([Vg_pred_consistent(2e-1,N,mu,gam,L,sigma_e2,V_s) for sigma_e2 in sigma_e2s])
            # plt.plot(sigma_e2s, Vg_numerical/(Vg_numerical+1),'k',label='numerical')

            plt.ylim([0,.5])
            #plt.xlim([-1e-3,1.2e-2])
            #plt.legend()



#%%
#Evolution 2024 

fig, axs=plt.subplots(1,2,figsize=[5,2.5])

#axs[0].plot(phist[str(0.0)][-11000:-10000,:,0],linewidth=0.5)
#axs[0].set_ylim([0,1])
#axs[0].set_ylabel(r'Frequency',fontsize=14)
#axs[0].set_title(r'Stationary')
#axs[0].set_xticklabels([])
#axs[0].set_xlim([0,1000])

#axs[1].plot(phist[str(1e-2)][-11000:-10000,:,0],linewidth=0.5)
axs[1].plot(phist[str(1e-2)][:,:,1],linewidth=0.5)
#axs[1].set_xlim([0,1000])
axs[1].set_ylim([0,1])
axs[1].set_yticklabels([])
axs[1].set_title(r'Fluctuating')
#axs[1].set_xticklabels([])

#plt.annotate(s=r'Time',xy=(0.52,0.04), xytext=(0.52,0.04),xycoords='figure fraction',fontsize=14)

plt.tight_layout()
plt.show()
#plt.savefig("stat_vs_fluc.pdf")

#%%
#SFS
#

V_g_theory=np.mean(V_g)
#V_g_theory=0.8

sigma_c2=alpha**2*sigma_e2/V_g_theory**2
#sigma_c2=0.008
s=alpha**2/(2*V_s)

x=np.linspace(1/N,1-1/N,1000)

#u=log(x)
u=np.linspace(np.log(1/N),np.log(1-1/N),31)

plt.figure()
poly_loci=((p<1) & (p>0))
plt.hist(np.log(phist[-1][poly_loci]),u,density=True)
#plt.hist(phist[-1][poly_loci],x,density=True)

#neutral
#plt.semilogy(x,0.9/(np.log(N)*x),'k--')
#ignoring frequency dependence of average selection pressure
#plt.semilogy(x,0.2*np.exp(-2*N*s*x)/(x*(1-x)),'k--')
#complete
u=np.linspace(np.log(1/N),np.log(1-1/N),101)
if sigma_e2>0:
    plt.semilogy(u,np.array([phi_not_normed(np.exp(_),N,sigma_c2,s)*np.exp(_) for _ in u])*phi_norm_const(N,sigma_c2,s,mu))
    #plt.semilogy(x,np.array([phi_not_normed(_,N,sigma_c2,s) for _ in x])*phi_norm_const(N,sigma_c2,s,mu))
    
#V->0
#plt.semilogy(x,phi_nofluc(x,N,s),'r')

#plt.ylim([10**-2,10**3])

#print(np.sum((phist[-1]>0.1) & (phist[-1]<0.9))/L)

# plt.figure()
# p_ordered=np.sort(phist[-1])[::-1]/np.sum(p_ordered)
# plt.plot(np.array(range(len(p_ordered))),np.cumsum(p_ordered))
#plt.ylim([0,1])

#%%%

from scipy.integrate import quad
from scipy.optimize import minimize

trunc=1/N

# neutral_norm=quad(lambda x: 1/x,trunc,1-trunc)[0]

# print(quad(lambda x: x*(1-x)*1/x,trunc,1-trunc)[0]/neutral_norm)

# MS_norm=quad(lambda x: np.exp(-2*N*s*x)/x,trunc,1-trunc)[0]
# print(quad(lambda x: x*(1-x)*np.exp(-2*N*s*x)/x,trunc,1-trunc)[0]/MS_norm)

# noV_norm=quad(lambda x: phi_noV(x,N,s),trunc,1-trunc)[0]
# print(quad(lambda x: x*(1-x)*phi_noV(x,N,s),trunc,1-trunc)[0]/noV_norm)

def Vg_pred_MS_drift(N,alpha,V_s,trunc):
    s=alpha**2/(2*V_s)
    
    norm=quad(lambda x: np.exp(-2*N*s*x)/(x*(1-x)),trunc,1-trunc)[0]
    return alpha*(quad(lambda x: x*(1-x)*np.exp(-2*N*s*x)/(x*(1-x)),trunc,1-trunc)[0])/norm

def Vg_pred_nofluc(N,alpha,V_s,trunc):
    s=alpha**2/(2*V_s)
    
    norm=quad(lambda x: phi_nofluc(x,N,s),trunc,1-trunc)[0]
    return alpha*(quad(lambda x: x*(1-x)*phi_nofluc(x,N,s),trunc,1-trunc)[0])/norm



#print(Vg_pred(Vg_sim,N,alpha,sigma_e2,V_s,trunc))

#Vg_theory=np.array([Vg_pred_consistent(alpha*0.5/np.log(N),N,alpha,sigma_e2,_,trunc) for _ in Vs_range])
Vg_theory_nofluc=np.array([Vg_pred_nofluc(N,alpha,_,trunc) for _ in Vs_range])
Vg_theory_MS_drift=np.array([Vg_pred_MS_drift(N,alpha,_,trunc) for _ in Vs_range])

plt.figure()
#plt.plot(np.sqrt(Vs_range),Vg_theory)
plt.plot(np.sqrt(Vs_range),Vg_theory_nofluc,'k--')
#plt.plot(np.sqrt(Vs_range),Vg_theory_MS_drift,'k')
plt.plot(np.sqrt(Vs_range),len(Vs_range)*[0.5*alpha/np.log(N)])
gamma=N*alpha**2/Vs_range
#plt.plot(np.sqrt(Vs_range),0.075*alpha/(gamma/2 + gamma/(np.exp(gamma)-1)))
plt.plot(np.sqrt(Vs_range),0.06*alpha*(2/gamma)*(1-np.exp(-gamma))/(np.exp(-gamma)+1),'k')
plt.plot(np.sqrt(Vs_range),Vg_sims,'.')

mu=5*10**-6
p_MS=mu/(1/(2*Vs_range))
plt.plot(np.sqrt(Vs_range),alpha*p_MS*(1-p_MS))


# plt.figure()
# plt.plot(np.sqrt(Vs_range),Vg_theory/(Vg_theory+Vs_range/10))
# plt.plot(np.sqrt(Vs_range),Vg_theory_nofluc/(Vg_theory_nofluc+Vs_range/10))
# plt.plot(np.sqrt(Vs_range),Vg_theory_MS_drift/(Vg_theory_MS_drift+Vs_range/10))
# plt.plot(np.sqrt(Vs_range),alpha*p_MS*(1-p_MS)/(alpha*p_MS*(1-p_MS)+Vs_range/10))

# plt.plot(V_g_range,Vg_pred)
# plt.plot(V_g_range,V_g_range)



#%%Approximation
N=10000
ss=gam**2/(2*Vs[1])
sc2=gam**2*1e-3/Vg_mean[1]**2; 
#sc2=0.00013613; 
a=N*sc2; b=2*ss/sc2
C=np.sqrt(1+4/a)
r1=0.5*(1-C)
r2=0.5*(1+C)

d=np.abs(r1)

x=np.linspace(1/N,1-1/N,1000)

plt.figure()

norm=((x*(1-x))**(2*N*mu-1)*(a*(x-r1)*(-x+r2))**-b)[0]
plt.plot(x,(x*(1-x))**(2*N*mu-1)*(a*(x-r1)*(-x+r2))**-b/norm,'.')

norm=((x*(1-x))**(2*N*mu-1))[0]
plt.plot(x,(x*(1-x))**(2*N*mu-1)/norm,'-')




def phi_exact(x):
    
    
    return (x*(1-x))**(2*N*mu-1)*(a*(x-r1)*(-x+r2))**-b

# norm=((x)**(2*N*mu-1)*(1-x*b/d))[0]
# plt.plot(x,(x)**(2*N*mu-1)*(1-x*b/d)/norm,'-')


x=np.linspace(0,1,1000)
plt.figure()
plt.plot(x,(a*(x-r1)*(-x+r2))**-b,'.')
#plt.plot(x,(C*a)**-b*((x-r1)**-b+(-x+r2)**-b),'-')

b=b*(1/(1+np.abs(r1)))

norm_const=1/((C*a)**-b*((0-r1)**-b+0*(-0+r2)**-b))
plt.plot(x,norm_const*(C*a)**-b*((x-r1)**-b+0*(-x+r2)**-b),'--')


#%%
# plt.legend()
x=np.linspace(0.05,1)
plt.figure()
for _ in [100,1000,1e4]:
    
    #plt.plot(x,Vg_theory_opt(x,_,gam,sigma_e2,L,mu,5),'--')
    plt.plot(x,[Vg_pred(x0,1e4,mu,a,_,0.01,5) for x0 in x])
    
    plt.plot(x,x)
    plt.ylim([0,10])

