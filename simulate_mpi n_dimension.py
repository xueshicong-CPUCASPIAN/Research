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

def p_prime_sel_opt(p, delt_opt, effects, V_s):
    """
    p:        (L, rep)
    delt_opt: (T, rep)
    effects:  (L, rep, T)
    """
    S = 1 / (2 * V_s)

    # dot_term[l, r] = sum_t effects[l, r, t] * delt_opt[t, r]
    dot_term = np.einsum('lrt,tr->lr', effects, delt_opt)

    # norm2[l, r] = sum_t effects[l, r, t]^2
    norm2 = np.sum(effects**2, axis=2)

    expo = 2 * S * (dot_term + 0.5 * norm2 * (2 * p - 1))
    p_new = pmap(rhomap(p) * np.exp(expo))
    return p_new

def simulate(param):
    L,sigma_e2,N,V_s,mu,a2,theta,n_traits,rep=param # add n_traits and rep to param tuple
    a = np.sqrt(a2) # still use a2 as the variance of mutational effects, but now mutational effects are n-dimensional vectors.
    effects = np.random.normal(0, a, size=(L, rep, n_traits)) # mutational effects for L loci and n traits
    opt = np.zeros((n_traits, rep)) # optimum for n traits and rep replicate populations
    p=np.zeros([L,rep]) #frequency of mutant allele at each locus and replicate population
    maxiter=int(10*N)
    
    for t in range(maxiter): # generation
        
        fixed_loci_1=(p==1)
        if t%1000==0:
            print("now time is ", t)
        #Reset fixed loci and remove them from optimum
        p[fixed_loci_1]=0
        opt=opt-2*np.einsum('lr,lrt->tr', fixed_loci_1, effects) #re-centered to save computation
        
        allele_expected = (2*p**2 + 2*p*(1-p))   # AA Aa aa, expected number with A frequency p.  shape L by rep.
        # 2*p^2 for AA, 2p(1-p) for Aa, 0 for aa.
        zbar = np.einsum('lr,lrt->tr', allele_expected, effects) # mean phenotype for n traits and rep populations.
        # shape of allele_expected is L by rep, shape of effects is L by n_traits, so the output shape is n_traits by rep.
        
        fixed_loci_0=(p==0)
        mutation_mask=((np.random.rand(L,rep)<N*mu) & fixed_loci_0)
        # array/matrix L rows and rep columns. entries from 0 to 1.
        # N = population size 
        # p==0 this mutation is absent in this population now.
        np.place(p,mutation_mask,1/N) # replace p = 0 with 1/N
        new_idx = np.where(mutation_mask)  # new_idx[0]=locus, new_idx[1]=replicate
        effects[new_idx[0], new_idx[1], :] = np.random.normal(0, a, size=(len(new_idx[0]), n_traits)
)
        # for each new mutation, assign it a new mutational effect drawn from the normal distribution.

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

        p=np.random.binomial(N,p_prime_sel_opt(p,opt-zbar,effects,V_s))/N # set parameter gamma as value a.
        #p=np.random.binomial(N,p,size=sim_L)/N
        # perform n independent Bernoulli trials with success prob p, return the number of success
        # the number of mutant individuals/N = frequency.
        opt = (1-theta)*opt + np.random.normal(0, np.sqrt(sigma_e2), size=(n_traits, rep))#modify
        # random normal is the noise
    return 2*np.sum(np.sum(effects**2, axis=2)*p*(1-p), axis=0) #simulation output: genetic variance V_g for each replicate population. shape (rep,).
    # effects has shape (L, rep, n_traits), so effects**2 has shape (L, rep, n_traits). sum over axis=2 gives shape (L, rep). 
    # multiply by p*(1-p) and sum over axis=0 gives shape (rep,). multiply by 2 gives the final output shape (rep,).


#%%

####################################
#Parallel handling of replicates with MPI.

#sigma_e2s=np.array([0,1e-4,5e-4,1e-3,5e-3,1e-2])
sigma_e2s=np.array([0,1e-4,5e-4,1e-3,5e-3,1e-2])
Ls=np.array([100]) # number of loci
Ns=np.array([10000])
Vs=np.array([5,20])
mus=np.array([6.6e-6])
thetas=np.array([0e-1])
a2s=np.array([0.1])
all_reps=100 # rep = number of replicate populations simulated in parallel. For MPI splitting.
n_traits = np.array([1,2,3,5]) # vary dimensionality to show n-D effect

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

#number of replicates handled by each core
rep_local=int(all_reps/size)
params=[_ for _ in 
        itertools.product(Ls,sigma_e2s,Ns,Vs,mus,a2s,thetas, n_traits,[rep_local])
        ] #add n_traits

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
        itertools.product(Ls,sigma_e2s,Ns,Vs,mus,a2s,thetas, n_traits,np.array([all_reps]))
        ] # add n_traits

if rank==0: print(output)

np.savetxt("Vg_sims_n_dimension",np.array(output),header=str(params)) # modify filename

# %%
# 