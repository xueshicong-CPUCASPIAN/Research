# -*- coding: utf-8 -*-
"""
Created on Tue May 30 15:27:44 2023

@author: jason
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.optimize
import ast

#
#Theory functions
#======================                  

def Vg_LB(mu,L,Vs):
    return 4*mu*L*Vs

def phi_not_normed(x,N,mu,sigma_c2,s):
    if sigma_c2==0:
        return np.exp(-2*N*s*x*(1-x))*(x*(1-x))**(2*N*mu-1)
    else:
        return ((1+N*sigma_c2*x*(1-x))**(-2*s/sigma_c2-2*N*mu)
                /(x*(1-x))**(1-2*N*mu))

def phi_norm_const(N,mu,sigma_c2,s):
    return (0.5/(integrate.quad(lambda x: 
            phi_not_normed(x,N,mu,sigma_c2,s),0/N,1/N)[0]
            +integrate.quad(lambda x: 
            phi_not_normed(x,N,mu,sigma_c2,s),1/N,0.5)[0]))

def E_H(N,mu,sigma_c2,s):
    return (2*(integrate.quad(lambda x: x*(1-x)
            *phi_not_normed(x,N,mu,sigma_c2,s),0/N,1/N)[0]
            +integrate.quad(lambda x: x*(1-x)
            *phi_not_normed(x,N,mu,sigma_c2,s),1/N,0.5)[0])
            *phi_norm_const(N,mu,sigma_c2,s))

def Vg_pred(Vg,N,mu,a,L,sigma2,V_s):
    s=a**2/(2*V_s)
    sigma_c2=a**2*sigma2/(Vg)**2
    return 2*a**2*L*E_H(N,mu,sigma_c2,s)

def Vg_pred_consistent(init,N,mu,a,L,sigma2,V_s):
    res = (scipy.optimize.minimize(lambda x: 
            (Vg_pred(x,N,mu,a,L,sigma2,V_s)-x)**2, init))
    return res.x[0]

def Vg_theory_opt(x,N,a,sigma_e2,L,mu,Vs):
    if sigma_e2==0:
        return x-4*L*mu*Vs
    else: 
        sc2=a**2*sigma_e2/x**2
        d=np.abs(0.5*(1-np.sqrt(1+4/(N*sc2))))
        b=x**2/(Vs*sigma_e2)-2*N*mu
        b_t=b/(d+1)
        return ((2*N*mu)*(2*L*a**2)*d**b_t*(d**(1-b_t)
                -(0.5+d)**(1-b_t))/(b_t-1)-x)


#split AF trajectories so that jumps to zero after fixation are avoided in plot
def split_traj(hist):
    temp=[]
    hist=np.array(hist.transpose())
    for loc,_ in enumerate(hist):
        _=np.array(_)
        ones_pos=np.arange(len(_))[_==1]
        ones_pos=np.concatenate( ([0],ones_pos,[len(_)]) ) 
        if len(ones_pos)>2:
            for i in range(len(ones_pos)-1):
                traj=np.concatenate(
                    (np.zeros(ones_pos[i]+1),
                    _[ones_pos[i]+1:ones_pos[i+1]],
                    np.ones(len(_)-ones_pos[i+1]))
                    )
                if i==0: 
                    hist[loc]=traj
                else:
                    temp.append(traj)
            
    return np.concatenate( (hist,np.array(temp)) ).transpose()

#%%
#Load data

fname="Vg_sims_th0"
with open(fname,'r') as fin:
    params=eval(fin.readline()[2:-1])
    #parameter format: L,sigma_e2,N,V_s,mu,a2,theta,rep

Vg_sims=np.loadtxt(fname)

hist0=np.loadtxt('hist_000.txt')
hist1=split_traj(np.loadtxt('hist_001.txt'))
delt=np.loadtxt('delta_hist_001.txt')


######################################
#%%
#timeseries figure

fig, axs=plt.subplots(3,1,figsize=[3,6])

t1=95000
t2=100000

axs[0].plot(hist1[t1:t2,:],'gray', linewidth=0.5)
axs[0].plot(hist0[t1:t2,:],'k', linewidth=0.5)
axs[0].set_ylim([0,1.])
axs[0].set_xlim([0,t2-t1])
axs[0].set_xticklabels([])
axs[0].set_ylabel(r'Frequency',fontsize=10)
axs[0].annotate(r'$A$',[0.92,0.87],xycoords='axes fraction',fontsize=14)

Vg=2*np.sum(0.1*hist0[t1:t2,:]*(1-hist0[t1:t2,:]),1)
axs[1].plot(Vg/(1+Vg),'k',label=r'$\sigma^2=0$ (LB)')
Vg=2*np.sum(0.1*hist1[t1:t2,:]*(1-hist1[t1:t2,:]),1)
axs[1].plot(Vg/(1+Vg),'gray',label=r'$\sigma^2=10^{-2}$')
axs[1].set_ylim([0,1])
axs[1].set_xlim([0,t2-t1])
axs[1].set_xticklabels([])
axs[1].set_ylabel(r'Heritability',fontsize=10)
axs[1].annotate(r'$B$',[0.92,0.87],xycoords='axes fraction',fontsize=14)

axs[2].plot(delt[t1:t2],'gray')
axs[2].axhline(y=0,color='k')
axs[2].set_xlim([0,t2-t1])
axs[2].set_ylabel(r'$\delta_t$',fontsize=10)
axs[2].set_xlabel(r'Generations')
axs[2].annotate(r'$C$',[0.92,0.87],xycoords='axes fraction',fontsize=14)

inset = axs[1].inset_axes([0.42,0.69,0.3,0.30])

Vg=2*np.sum(0.1*hist1[:,:]*(1-hist1[:,:]),1)

inset.hist(Vg/(1+Vg),bins=20,color='gray')
Vg=2*np.sum(0.1*hist0[:,:]*(1-hist0[:,:]),1)
inset.hist(Vg/(1+Vg),color='k')
inset.set_yticklabels([])
inset.set_yticks([])
inset.set_xticks([0,0.6])
inset.set_xticklabels([0,0.6],fontsize=8)
inset.set_xlabel(r'$h^2$',labelpad=-10)
#inset.set_title(r'$h^2$')

axs[1].legend(loc='upper left',fontsize=5)
#axs[1].legend(loc=[0.01,1.8])

#plt.savefig('timeseries.pdf',bbox_inches='tight')

#%%
########################################
#sigma2 dependency
offset=1e-5

#index of variable on x axis
xvar=1

N=10000
a2=0.1
fig, axs=plt.subplots(2,2,figsize=[7,7])
axs=axs.flat

unique_params=set([_[:1]+_[2:6] for _ in params if _[-3]==a2 and _[2]==N])
indices=[_ for _ in range(len(params)) if params[_][-3]==a2 and params[_][2]==N]
fig_dict=dict(zip(unique_params,axs))

for i in indices:
    L,sigma_e2,N,V_s,mu,a2,theta,rep=params[i]
    a=np.sqrt(a2)

    ax=fig_dict[(L,N,V_s,mu,a2)]
    
    parts=ax.violinplot(
            Vg_sims[i]/(Vg_sims[i]+1),
            positions=[np.log10(sigma_e2+offset)],
            widths=0.25,showmeans=True)

    for pc in parts['bodies']:
        pc.set_color('C0')
        pc.set_alpha(0.5)
   
    parts['cbars'].set_color('C0')
    parts['cmeans'].set_color('C0')
    parts['cmins'].set_color('C0')
    parts['cmaxes'].set_color('C0')

    #variances for table 2
    if sigma_e2==1e-3:
        h2=Vg_sims[i]/(Vg_sims[i]+1)
        print(L,V_s,sigma_e2,np.var(h2))
        print('    ',np.std(h2)/np.mean(h2))
        print('    ',np.mean(h2))

    ax.set_ylim([0,1])
    ax.set_title(r'$L=$'+str(L)+r'$,V_s=$'+str(V_s),y=.99,fontsize=11)
                    
    h2_LB=Vg_LB(mu,L,V_s)/(1+Vg_LB(mu,L,V_s))
    ax.axhline(y=h2_LB,color='k',ls='--',label=r'Latter-Bulmer $(4L\mu V_s)$')
    ax.axhspan(0.1,0.6,color='k',alpha=0.02)

sig=np.unique(np.array(params)[:,1])

for _ in fig_dict.keys():
    ax=fig_dict[_]
    L,N,V_s,mu,a2=_
    Vg_numerical=np.array([
        Vg_pred_consistent(1e-1,N,mu,a,L,sigma_e2,V_s)
        for sigma_e2 in sig])

    ax.plot(
            np.log10(sig+offset),
            Vg_numerical/(Vg_numerical+1),
            'k:',label=r'Diff, approx.',alpha=0.7)

    scaleVg=Vg_LB(mu,L,V_s)+np.array([np.sqrt(V_s*sigma_e2) for sigma_e2 in sig])
    ax.plot(
            np.log10(sig+offset),
            scaleVg/(scaleVg+1),
            'k', label=r'$V_g=4L\mu V_s+\sqrt{V_s\sigma^2}$',alpha=0.7)


axs[2].set_ylabel(r'Heritability $h^2$',fontsize=14)
axs[2].yaxis.set_label_coords(-0.2,1.1)

tick_list=[r'$0.0$',' ',r'$0.2$',' ',r'$0.4$',' ',r'$0.6$',' ',r'$0.8$']
#axs[0].set_yticklabels(tick_list,fontsize=12)
#axs[2].set_yticklabels(tick_list,fontsize=12)
axs[1].set_yticklabels([])
axs[3].set_yticklabels([])

tick_list=['',r'$0$', r'$10^{-4}$',  r'$10^{-3}$', r'$10^{-2}$']
axs[2].set_xticklabels(tick_list,fontsize=12)
axs[3].set_xticklabels(tick_list,fontsize=12)
axs[0].set_xticklabels([])
axs[1].set_xticklabels([])

axs[2].set_xlabel(r'Fluctuation intensity $\sigma^2$',x=1.15,fontsize=14)

#remove duplicate legend entries
handles, labels = axs[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axs[0].legend(by_label.values(), by_label.keys(), loc=[0.01,.7],fontsize=9)

#plt.savefig('violinplot_N'+str(N)+'_a2'+str(a2)+'_th'+str(theta)+'_mu'+str(mu)+'.pdf',
#            bbox_inches='tight')

##################################
#%%
#Latter-Bulmer_fluctuation predictions 
import matplotlib.patches as mpatches

mu=6.6e-9
sigma2=1e-2
N=10000

plt.figure(figsize=[3,2])

Ls=np.exp(np.linspace(np.log(1.2e4),np.log(1.2e8),20))
plt.plot(np.log10(Ls),
         Vg_LB(mu,Ls,20)/(1+Vg_LB(mu,Ls,20)),'--',
         label=r'$V_s=20, \sigma^2=0$ (LB)') 
plt.plot(np.log10(Ls), Vg_LB(mu,Ls,5)/(1+Vg_LB(mu,Ls,5)),'--',
         label=r'$V_s=5,$  $\sigma^2=0$ (LB)')

#rescale for computational stability
Ls=1e-3*Ls
mu=mu*1e3

plt.gca().set_prop_cycle(None)

h2_theory=[]
for _ in Ls:
    temp=Vg_pred_consistent(1e-1,10000,mu,0.3,_,sigma2,20)
    h2_theory.append(temp/(1+temp))

plt.plot(np.log10(Ls)+3,
         h2_theory,'b:',label=r'$V_s=20, \sigma^2=10^{-2}$ (Diff.)')

h2_theory=[]
for _ in Ls:
    temp=Vg_pred_consistent(1e-1,10000,mu,0.3,_,sigma2,5)
    h2_theory.append(temp/(1+temp))

plt.plot(np.log10(Ls)+3,
         h2_theory,'r:',label=r'$V_s=5,$  $\sigma^2=10^{-2}$ (Diff.)')

unique_params=set([_[:1]+_[2:6] for _ in params 
                   if _[-3]==0.1 and _[2]==N and _[1]==1e-2])
indices=[_ for _ in range(len(params)) 
         if params[_][-3]==0.1 and params[_][2]==N and params[_][1]==1e-2]

for i in indices:
    L,sigma_e2,N,V_s,mu,a2,theta,rep=params[i]
    
    if V_s==5:
        violin_displace=-0.05
        col="Red"
    else:
        violin_displace=0.05
        col="Blue"
    
    parts=plt.violinplot(
            Vg_sims[i]/(Vg_sims[i]+1),
            positions=[np.log10(1e3*L)+violin_displace],
            widths=0.1,showmeans=True)
    
    for _ in parts['bodies']: 
        _.set_color(col)
        _.set_edgecolor(col)

    for partname in ('cbars','cmins','cmaxes','cmeans'):
            vp = parts[partname]
            vp.set_edgecolor(col)
            vp.set_linewidth(1)

plt.axhspan(0.1,0.6,color='k',alpha=0.1)

plt.xlabel(r'Target size (fraction of genome)',fontsize=10)
plt.ylabel(r'Heritability',fontsize=10)

plt.gca().set_xticklabels(
        ['',r'$10^{-4}$','',r'$10^{-3}$','',r'$10^{-2}$','',r'$10^{-1}$','',r'$1$'],
        fontsize=10)
plt.yticks(fontsize=10)

handles, labels = plt.gca().get_legend_handles_labels()
patch = mpatches.Patch(color='blue', alpha=0.5,
                       label=r'$V_s=20,\sigma^2=10^{-2}$ (Sim.)')
handles.append(patch)
patch = mpatches.Patch(color='red', alpha=0.5,
                       label=r'$V_s=5,$  $\sigma^2=10^{-2}$ (Sim.)')
handles.append(patch)
plt.legend(loc='upper left',handles=handles, fontsize=4.5)

plt.savefig('LB_fluc_'+str(N)+'_a2'+str(a2)+'_th'+str(theta)+'.pdf',bbox_inches='tight')


#########################################
#%%
#Autocorrelation timescale and change in Vg

import pandas as pd

hist1=split_traj(np.loadtxt('hist_001_100.txt'))
delt=np.loadtxt('delta_hist_001_100.txt')
delt_pd=pd.Series(delt)
Vg=2*np.sum(0.1*hist1[:,:]*(1-hist1[:,:]),1)

mean_tau=np.mean(5/Vg[1:])

fig, axs=plt.subplots(3,1,figsize=[3,6])

autocor=np.array([delt_pd.autocorr(lag=_) for _ in range(400)])
axs[0].hist(5/Vg[100:],100) #exclude initial condition which has Vg\approx 0
axs[0].set_xlim([0,400])
axs[0].axvline(mean_tau,color='k')
axs[0].set_yticklabels([])
#axs[0].set_xticklabels([])
axs[0].set_ylabel('Frequency')
axs[0].set_title(r'Autocov. timescale $V_s/V_g$',fontsize=8,pad=0)
axs[0].annotate(r'$\langle V_s/V_g\rangle$',[0.15,0.5],xycoords='axes fraction')

axs[1].plot([np.mean(np.abs(Vg[tau:]-Vg[:-tau])/Vg[tau:]) for tau in range(1,400)]); 
axs[1].axvline(mean_tau,color='k')
axs[1].set_xlim([0,400])
#axs[1].set_xlabel(r'Time difference ($\tau$)')
axs[1].set_ylabel(r'$\mathbb{E}[|V_g(t+\tau)-V_g(t)|]/V_g(t)$')
axs[1].set_xticklabels([])

axs[2].plot(autocor);
axs[2].plot(np.exp(-np.arange(400)/mean_tau),'k--'); 
axs[2].axvline(mean_tau,color='k')
axs[2].set_xlim([0,400])
axs[2].set_xlabel(r'Time difference ($\tau$)')
axs[2].set_ylabel(r'$\langle \delta_{t+\tau},\delta_t\rangle/\langle \delta_t,\delta_t\rangle$')

plt.savefig('VG_autocorr.pdf',bbox_inches='tight')

#%%
########################################
#Exploratory plots
#Not in manuscript

#small offset to avoid log(0)
offset=1e-5

#index of variable on x axis
xvar=1

#log x axis option
logx=True

indices=[_ for _ in range(len(params)) if params[_][-3]==0.1]

for i in indices:

    plt.figure(str([_ for ind,_ in enumerate(params[i]) if ind != xvar]))
    ax=plt.gca() 

    L,sigma_e2,N,V_s,mu,a2,theta,rep=params[i]
    a=np.sqrt(a2)
    
    #Vg_theory=scipy.optimize.fsolve(
    #        lambda x: Vg_theory_opt(x,N,a,sigma_e2,L,mu,V_s),0.025)[0]
    
    Vg_numerical=Vg_pred_consistent(2e-1,N,mu,a,L,sigma_e2,V_s)
    
    h2_LB=Vg_LB(mu,L,V_s)/(1+Vg_LB(mu,L,V_s))
    ax.axhline(y=h2_LB,color='k',ls='--')
    ax.axhspan(0.1,0.6,color='k',alpha=0.02)

    #Simulated heritability violinplots
    #Environmental variance = 1
    if logx==True:  
        ax.violinplot(
                Vg_sims[i]/(Vg_sims[i]+1),
                positions=[np.log10(params[i][xvar]+offset)],
                widths=2e-1,showmeans=True)
        
        #ax.plot(
        #        np.log10(params[i][xvar]+offset),
        #        Vg_theory/(Vg_theory+1),
        #        'ko',fillstyle='none',markersize=8,
        #        label=r'Theory analytical',alpha=0.7)
        
        ax.plot(
                np.log10(params[i][xvar]+offset),
                Vg_numerical/(Vg_numerical+1),
                'kx',markersize=10,
                label=r'Diffusion approximation MSB',alpha=0.7)
        
    else:
        ax.violinplot(
                Vg_sims[i]/(Vg_sims[i]+1),
                positions=[params[i][xvar]+offset],
                widths=1e-3,showmeans=True)
        
        ax.plot(
                params[i][xvar]+offset, Vg_theory/(Vg_theory+1),
                'ko',fillstyle='none',markersize=8,
                label=r'Theory analytical',alpha=0.7)
        
        ax.plot(
                params[i][xvar]+offset, Vg_numerical/(Vg_numerical+1),
                'kx',markersize=10,
                label=r'Theory numerical',alpha=0.7)
       
    ax.set_ylim([0,0.8])

for _ in plt.get_figlabels():
    plt.figure(_)
    plt.savefig('/home/jason/git/fluctuating_optimum/'+_+'.png')

#plt.close('all')

###################
#%%
#Moving optimum picture
#not used in manuscript

x=np.linspace(-2,2.1,1000)

fig,ax=plt.subplots(figsize=[4,2.7])

ax.plot(x,np.exp(-(x-0.6)**2),'r',label='Fitness')
ax.plot(x,np.exp(-10*x**2),'b',label='Population')

ax.set_xticklabels([])
ax.set_xlabel(r'Trait value',fontsize=14)

#ax.set_yticklabels([])
ax.set_ylabel(r'Distribution',fontsize=14)

xpos=0.4
plt.annotate(s=r'', xy=(-xpos+0.6,1.05), xytext=(xpos+0.6,1.05), arrowprops=dict(arrowstyle='<->',linewidth='2'))

plt.annotate(s='Trait\noptimum',xy=(0.6,1),xytext=(0.22,.6),fontsize=12,arrowprops=dict(arrowstyle='->'))

plt.ylim([0,1.1])

plt.tight_layout()

plt.legend(loc='upper left',frameon=False)

plt.savefig("stablizing_fluc.eps")

#%%
#Chasing optimum picture

x=np.linspace(-2,2.1,1000)

fig,ax=plt.subplots(figsize=[4,2.7])

ax.plot(x,np.exp(-(x-1.5)**2),'r',label='Fitness')
ax.plot(x,np.exp(-10*x**2),'b',label='Population')

ax.set_xticklabels([])
ax.set_xlabel(r'Trait value',fontsize=14)

#ax.set_yticklabels([])
ax.set_ylabel(r'Distribution',fontsize=14)

xpos=0.4
plt.annotate(s='', xy=(-0.5,1.0), xytext=(0.8,1.0), 
             arrowprops=dict(arrowstyle='<-',linewidth='2'))

plt.annotate(s=r'Rate $\propto V_g$',xy=(0.,1.025), xytext=(-0.4,1.025))
plt.ylim([0,1.1])
plt.tight_layout()
plt.legend(loc='upper left',frameon=False)

plt.savefig("stablizing_lande.eps")


######################################
#%%
#Stablizing picture
#not used in manuscript

x=np.linspace(-2,2.1,1000)

fig,ax=plt.subplots(figsize=[4,2.7])

ax.plot(x,np.exp(-x**2),'r',label='Fitness')
ax.plot(x,np.exp(-10*x**2),'b',label='Population')
ax.set_xticklabels([])
ax.set_xlabel(r'Trait value',fontsize=14)
ax.set_yticklabels([])
ax.set_ylabel(r'Distribution',fontsize=14)

xpos=0.4
plt.annotate(text='', xy=(-xpos,np.exp(-10*xpos**2)), xytext=(xpos,np.exp(-10*xpos**2)), arrowprops=dict(arrowstyle='<->'))
plt.annotate(text=r'$V_g+V_e$',xy=(-0.435,np.exp(-0.15*xpos**2)-0.15),xytext=(-0.435,np.exp(-10*xpos**2)-0.15),fontsize=14)

xpos=0.6
plt.annotate(text='', xy=(-xpos,np.exp(-xpos**2)), xytext=(xpos,np.exp(-xpos**2)), arrowprops=dict(arrowstyle='<->'))
plt.annotate(text=r'$V_s$',xy=(-0.15,np.exp(-xpos**2)-0.15),xytext=(-0.15,np.exp(-xpos**2)-0.15),fontsize=14)

plt.annotate(text=r'Trait optimum',xy=(0,1),xytext=(0,1.1),fontsize=12,arrowprops=dict(arrowstyle='->'))

plt.tight_layout()

plt.legend(loc='upper left',frameon=False)

#plt.savefig("stablizing.eps")

###################
#%%
#Latter-Bulmer predictions
#not used in manuscript

mu=6.6e-9

Ls=np.linspace(1.2e4,1.2e8,10000)
plt.plot(np.log10(Ls),Vg_LB(mu,Ls,20)/(1+Vg_LB(mu,Ls,20)),label=r'$V_s=20V_e$')
plt.plot(np.log10(Ls),Vg_LB(mu,Ls,5)/(1+Vg_LB(mu,Ls,5)),label=r'$V_s=5V_e$')

plt.fill_between(np.log10(Ls),[0.1],[0.6],alpha=0.5)

plt.xlabel(r'Target size (fraction of euchromatic genome)',fontsize=14)
plt.ylabel(r'Heritability',fontsize=14)

plt.gca().set_xticklabels(['',r'$10^{-4}$','',r'$10^{-3}$','',r'$10^{-2}$','',r'$10^{-1}$','',r'$1$'],fontsize=14)
plt.yticks(fontsize=14)

plt.legend(fontsize=14)


plt.ylim([0,1])
plt.tight_layout()
#plt.savefig("LB.pdf")

#%%
########################################
#sigma2 dependency excess only

fig, axs=plt.subplots(3,2,figsize=[3.5,5])
fig.subplots_adjust(hspace=0.35)
top=0.5

fname="Vg_sims_sig"
with open(fname,'r') as fin:
    params=eval(fin.readline()[2:-1])
    #parameter format: L,sigma_e2,N,V_s,mu,a2,theta,rep
Vg_sims=np.loadtxt(fname)

for i,par in enumerate(params):
    L,sigma_e2,N,V_s,mu,a2,theta,rep=par

    Vg_sims[i]=Vg_sims[i]-Vg_LB(mu,L,V_s)

sig=np.array(params)[:,1]
axs[0,0].violinplot(np.transpose(Vg_sims),
                    positions=np.log10(sig+1e-5),
                    showmeans=True, widths=.4)

Vg_numerical=np.array([
    Vg_pred_consistent(1e-1,N,mu,np.sqrt(a2),L,sigma_e2,V_s)-Vg_LB(mu,L,V_s)
    for sigma_e2 in sig])
axs[0,0].plot(
        np.log10(sig+1e-5),
        Vg_numerical,
        'k:', label=r'Diff. approx.')
    
axs[0,0].plot(np.log10(sig+1e-5),np.sqrt(sig*V_s),'k',label=r'$\sqrt{V_s\sigma^2}$')

axs[0,0].set_ylim([-0.05,top])
#axs[0,0].set_xlabel(r'$\sigma^2$',x=1.1,labelpad=-12)
axs[0,0].set_xlabel(r'$\sigma^2$',labelpad=-1)
axs[0,0].set_xticks([-5,-4,-3,-2])
tick_list=[r'$0$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$']
axs[0,0].set_xticklabels(tick_list,fontsize=8)
handles, labels = axs[0,0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
axs[0,0].legend(by_label.values(), by_label.keys(), loc=[0.01,.65],fontsize=6)


fname="Vg_sims_V"
with open(fname,'r') as fin:
    params=eval(fin.readline()[2:-1])
    #parameter format: L,sigma_e2,N,V_s,mu,a2,theta,rep
Vg_sims=np.loadtxt(fname)

for i,par in enumerate(params):
    L,sigma_e2,N,V_s,mu,a2,theta,rep=par

    Vg_sims[i]=Vg_sims[i]-Vg_LB(mu,L,V_s)

V=np.array(params)[:,3]
axs[0,1].violinplot(np.transpose(Vg_sims),positions=V,showmeans=True, widths=2)

Vg_numerical=np.array([
    Vg_pred_consistent(1e-1,N,mu,np.sqrt(a2),L,sigma_e2,V_s)-Vg_LB(mu,L,V_s)
    for V_s in V])
axs[0,1].plot(V, Vg_numerical, 'k:')

axs[0,1].plot(V,np.sqrt(sigma_e2*V),'k')

axs[0,1].set_ylim([-0.05,top])
axs[0,1].set_xticks([1,5,10,15,20])
axs[0,1].set_xlabel(r'$V_s$',labelpad=-1)
axs[0,1].set_yticklabels([])


fname="Vg_sims_a"
with open(fname,'r') as fin:
    params=eval(fin.readline()[2:-1])
    #parameter format: L,sigma_e2,N,V_s,mu,a2,theta,rep
Vg_sims=np.loadtxt(fname)

for i,par in enumerate(params):
    L,sigma_e2,N,V_s,mu,a2,theta,rep=par

    Vg_sims[i]=Vg_sims[i]-Vg_LB(mu,L,V_s)
    
a2s=np.array(params)[:,5]
axs[1,0].violinplot(np.transpose(Vg_sims),
                    positions=np.log10(a2s),
                    showmeans=True, widths=.2)

Vg_numerical=np.array([
    Vg_pred_consistent(1e-1,N,mu,np.sqrt(a2),L,sigma_e2,V_s)-Vg_LB(mu,L,V_s)
    for a2 in a2s])
axs[1,0].plot(np.log10(a2s), Vg_numerical, 'k:')

axs[1,0].plot(np.log10(a2s),len(a2s)*[np.sqrt(sigma_e2*V_s)],'k')

axs[1,0].set_ylim([-0.05,top])
axs[1,0].set_ylabel(r'Extra $V_g$ due to fluctuations')
axs[1,0].set_xlabel(r'$a^2$',labelpad=-1)
axs[1,0].set_xticks([-3,-2,-1])
tick_list=[r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$']
axs[1,0].set_xticklabels(tick_list,fontsize=8)

fname="Vg_sims_N"
with open(fname,'r') as fin:
    params=eval(fin.readline()[2:-1])
    #parameter format: L,sigma_e2,N,V_s,mu,a2,theta,rep
Vg_sims=np.loadtxt(fname)

for i,par in enumerate(params):
    L,sigma_e2,N,V_s,mu,a2,theta,rep=par

    Vg_sims[i]=Vg_sims[i]-Vg_LB(mu,L,V_s)

Ns=np.array(params)[:,2]
axs[1,1].violinplot(np.transpose(Vg_sims),
                    positions=np.log10(Ns),
                    showmeans=True, widths=.3)

Vg_numerical=np.array([
    Vg_pred_consistent(1e-1,N,mu,np.sqrt(a2),L,sigma_e2,V_s)-Vg_LB(mu,L,V_s)
    for N in Ns])
axs[1,1].plot(np.log10(Ns), Vg_numerical, 'k:')

axs[1,1].plot(np.log10(Ns),len(Ns)*[np.sqrt(sigma_e2*V_s)],'k')
axs[1,1].set_ylim([-0.05,top])
axs[1,1].set_xlabel(r'$2N$',labelpad=-1)
axs[1,1].set_yticklabels([])
axs[1,1].set_xticks([2,3,4,5])
tick_list=[r'$10^2$', r'$10^{3}$', r'$10^{4}$', r'$10^{5}$']
axs[1,1].set_xticklabels(tick_list,fontsize=8)


fname="Vg_sims_L"
with open(fname,'r') as fin:
    params=eval(fin.readline()[2:-1])
    #parameter format: L,sigma_e2,N,V_s,mu,a2,theta,rep
Vg_sims=np.loadtxt(fname)

for i,par in enumerate(params):
    L,sigma_e2,N,V_s,mu,a2,theta,rep=par

    Vg_sims[i]=Vg_sims[i]-Vg_LB(mu,L,V_s)

Ls=np.array(params)[:,0]
axs[2,0].violinplot(np.transpose(Vg_sims),
                    positions=np.log10(Ls),
                    showmeans=True, widths=.2)

Vg_numerical=np.array([
    Vg_pred_consistent(1e-1,N,mu,np.sqrt(a2),L,sigma_e2,V_s)-Vg_LB(mu,L,V_s)
    for L in Ls])

axs[2,0].plot(np.log10(Ls), Vg_numerical, 'k:')

#axs[2,0].plot(np.log10(Ls),len(Ls)*[np.sqrt(sigma_e2*V_s)],'k')
axs[2,0].plot(np.log10(Ls),np.sqrt(Ls*sigma_e2*V_s)/np.sqrt(100),'k')

axs[2,0].set_ylim([-0.05,top])
axs[2,0].set_xlabel(r'$L$',labelpad=-1)
axs[2,0].set_xticks([1,2,3])
tick_list=[r'$10$', r'$10^{2}$', r'$10^{3}$']
axs[2,0].set_xticklabels(tick_list,fontsize=8)


fname="Vg_sims_mu"
with open(fname,'r') as fin:
    params=eval(fin.readline()[2:-1])
    #parameter format: L,sigma_e2,N,V_s,mu,a2,theta,rep
Vg_sims=np.loadtxt(fname)

for i,par in enumerate(params):
    L,sigma_e2,N,V_s,mu,a2,theta,rep=par

    Vg_sims[i]=Vg_sims[i]-Vg_LB(mu,L,V_s)

mus=np.array(params)[:-2,4]
axs[2,1].violinplot(np.transpose(Vg_sims[:-2]),
                    positions=np.log10(mus),
                    showmeans=True,
                    widths=.3)

Vg_numerical=np.array([
    Vg_pred_consistent(1e-1,N,mu,np.sqrt(a2),L,sigma_e2,V_s)-Vg_LB(mu,L,V_s)
    for mu in mus])
axs[2,1].plot(np.log10(mus), Vg_numerical, 'k:')

axs[2,1].plot(np.log10(mus),len(mus)*[np.sqrt(sigma_e2*V_s)],'k')

axs[2,1].set_ylim([-0.05,top])
axs[2,1].set_xlabel(r'$\mu$',labelpad=-1)
axs[2,1].set_yticklabels([])
axs[2,1].set_xticks([-7,-6,-5])
tick_list=[r'$10^{-7}$', r'$10^{-6}$', r'$10^{-5}$']
axs[2,1].set_xticklabels(tick_list,fontsize=8)

plt.savefig('param_dep.pdf',bbox_inches='tight')
