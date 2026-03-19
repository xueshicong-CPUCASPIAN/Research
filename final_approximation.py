import numpy as np
import matplotlib.pyplot as plt
N=10000
sigma2=1e-2
a2=1e-1
Vs=5
ss=a2/(2*Vs)
L=1000
mu=5e-6

def sigma_a2(a2, sigma2, Vg):
    return a2*sigma2/Vg**2

def bbar(ss, a2, sigma2, Vg):
    b=2*ss/sigma_a2(a2, sigma2, Vg)
    return b/(1+d(a2, sigma2, Vg))

def d(a2,sigma2,Vg):
    return -.5 + np.sqrt(1/4 + 1/(N*sigma_a2(a2,sigma2,Vg)))

C=4*N*L*a2*mu

plt.figure()
x=np.linspace(1e-2,1,1000)
plt.plot(x,C*d(a2,sigma2,x)**bbar(ss,a2,sigma2,x)\
        /(bbar(ss,a2, sigma2, x) -1) \
        *(d(a2,sigma2,x)**(1-bbar(ss,a2,sigma2,x)) \
        -(0.5+d(a2,sigma2,x))**(1-bbar(ss,a2,sigma2,x))) \
        )


plt.plot(x,1\
        /(bbar(ss,a2, sigma2, x) -1) \
        *(1 \
        -(1+0.5/d(a2,sigma2,x))**(1-bbar(ss,a2,sigma2,x))) \
        )

#plt.plot(x,1\
#        /(bbar(ss,a2, sigma2, x) -1) \
#        *(d(a2,sigma2,x)**(1-bbar(ss,a2,sigma2,x)) \
#        -(0.5+d(a2,sigma2,x))**(1-bbar(ss,a2,sigma2,x))) \
#        ,'k')
#    
#plt.plot(x,-C*d(a2,sigma2,x)**bbar(ss,a2, sigma2, x)\
#        /(bbar(ss,a2, sigma2, x) -1) \
#        *(1/2))
#
#
#plt.plot(x,C*d(a2,sigma2,x)**bbar(ss,a2, sigma2, x)\
#        *(1/2))
#


plt.plot(x,x)
plt.ylim([-0.0,5.0])

#plt.savefig('/home/jason/git/fluctuating_optimum/temp.pdf')

#plt.plot(x,np.log(x**-0*4*N*100*a2*5e-6*d(a2,sigma2,x)**bbar(ss,a2,sigma2,x)\
#        /(d(a2,sigma2,x)-1) \
#        *(d(a2,sigma2,x)**(1-bbar(ss,a2,sigma2,x)) \
#        -(0.5+d(a2,sigma2,x))**(1-bbar(ss,a2,sigma2,x)))) \
#        )
#
#plt.plot(x,np.log(x**-0*4*N*100*a2*5e-6*d(a2,sigma2,0.01)**bbar(ss,a2,sigma2,x)\
#        /(d(a2,sigma2,x)-1) \
#        *(d(a2,sigma2,x)**(1-bbar(ss,a2,sigma2,x)) \
#        -(0.5+d(a2,sigma2,x))**(1-bbar(ss,a2,sigma2,x)))) \
#        )
#
#plt.plot(x,np.log(x))
#plt.hlines(x[0],x[-1],0)

#plt.plot([0,0.2],[0.31,0.066])

#plt.plot(x,x**(2*x**2))
#plt.plot(x,1-4.5*x**2)#*np.log(x))
