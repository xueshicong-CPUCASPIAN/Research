# -*- coding: utf-8 -*-
"""
Created on Thu May 30 14:31:48 2024

@author: jason
"""

import numpy as np
import itertools
import time
from mpi4py import MPI

#%%
#Simulation functions
#======================
def pmap(rho):
    return rho/(1+rho) # inverse function of rhomap.

def rhomap(p):
    return p/(1-p) # odds of success

def p_prime_sel_opt(p,delt_opt,gam,sign,V_s):
    S=1/(2*V_s) #V_s is the strength of stabilizing selection. Larger V_s, weaker selection.
    p=pmap(rhomap(p)*np.exp(2*S*gam*sign*(delt_opt+0.5*gam*sign*(2*p-1))))  
    # delt_opt = opt - z_bar
    # gam = gamma
    # p = allele frequency  
    return p

def simulate(param):
    L,sigma_e2,N,V_s,mu,a2,theta,rep=param
    a=np.sqrt(a2)

    sign=2*np.random.randint(0,2,[L,rep])-1
    
    opt=np.zeros(rep)
    p=np.zeros([L,rep])
    maxiter=int(10*N)
    
    for t in range(maxiter): # generation
        
        fixed_loci_1=(p==1)
        if t<10:
            print("now time is ", t)
        #Reset fixed loci and remove them from optimum
        p[fixed_loci_1]=0
        opt=opt-2*a*np.sum(sign*fixed_loci_1,0) #re-centered to save computation
        
        zbar=np.sum(2*a*sign*p**2+a*sign*p*(1-p),0) #mean trait
        
        fixed_loci_0=(p==0)
        mutation_mask=((np.random.rand(L,rep)<N*mu) & fixed_loci_0)
        # array/matrix L rows and rep columns. entries from 0 to 1.
        # N = population size 
        # p==0 this mutation is absent in this population now.
        np.place(p,mutation_mask,1/N) # replace p = 0 with 1/N
        np.place(
                sign,mutation_mask,
                2*np.random.randint(0,2,np.sum(mutation_mask))-1
                )
        # generate #np.sum(mutation_mask) random integers whose values are 0 or 1
        # np.sum(mutation_mask) is the number of new mutations.
        #0->0->-1,  1->2->1
        
        poly_loci=np.logical_not(fixed_loci_0) & (p<1-1/N) 
        #new mutants excluded also??? p =1/N
        #exclude alleles that are absent or almost fixed.
    
        #p[poly_loci]=p[poly_loci]+0*mu*(1-2*p[poly_loci]) ??? nothing
        p[poly_loci]=p[poly_loci]+\
            (np.random.rand(np.sum(poly_loci))<N*mu*(1-p[poly_loci]))/N-\
                (np.random.rand(np.sum(poly_loci))<N*mu*p[poly_loci])/N
        #vector

        p=np.random.binomial(N,p_prime_sel_opt(p,opt-zbar,a,sign,V_s))/N
        #p=np.random.binomial(N,p,size=sim_L)/N
        # perform n independent Bernoulli trials with success prob p, return the number of success
        # the number of mutant individuals/N = frequency.
        opt=(1-theta)*opt + np.random.normal(0,np.sqrt(sigma_e2),rep)
        # random normal is the noise
    return 2*a**2*np.sum(p*(1-p),0)
    # genetic variance (no idea about the coefficient 2)


#%%

####################################
#Parallel handling of replicates with MPI.

#sigma_e2s=np.array([0,1e-4,5e-4,1e-3,5e-3,1e-2])
sigma_e2s=np.array([1e-3])
Ls=np.array([100]) # number of loci
#Ls=np.array([10,20,50,100,200,500,1000])
Ns=np.array([10000])
#Ns=np.array([100,200,500,1000,2000,5000,10000,20000,50000,100000])
Vs=np.array([5])
#Vs=np.linspace(1,20,9)
mus=np.array([6.6e-6])
#mus=np.array([1e-7,2e-7,5e-7,1e-6,2e-6,5e-6,1e-5,2e-5,5e-5])
thetas=np.array([0e-1])
#a2s=np.array([0.001,0.002,0.005,0.01,0.02,0.05,0.1])
a2s=np.array([0.1])
all_reps=100 # rep = number of replicate populations simulated in parallel. For MPI splitting.

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#number of replicates handled by each core
rep_local=int(all_reps/size)
params=[_ for _ in 
        itertools.product(Ls,sigma_e2s,Ns,Vs,mus,a2s,thetas,[rep_local])
        ]

output=[]
for param in params:
    start=time.time()
    Vg_local=simulate(param)

    recvbuf=None
    if rank==0:
        recvbuf=np.empty([size,rep_local],dtype='d')
    comm.Gather(Vg_local,recvbuf,root=0)

    if rank==0:
        print((time.time()-start)/60)
        output.append(recvbuf.flatten())

params=[_ for _ in
        itertools.product(Ls,sigma_e2s,Ns,Vs,mus,a2s,thetas,np.array([all_reps]))
        ]

if rank==0: print(output)

np.savetxt("Vg_sims",np.array(output),header=str(params))

# %%
# for Jason's original code,
# # 1 parameter combination
# output is a list with one array inside it.
# That array contains the simulated Vg values for 100 replicate populations.