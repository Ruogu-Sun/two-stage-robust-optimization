import numpy as np

m=4
n=3
Gamma=0.6

Kapacity=np.random.uniform(200, 700, size=m)
d_bot=np.random.randint(10, 500, size=n)
alpha=np.random.uniform(0.1, 0.5, size=n)
d_tilde=alpha*d_bot

print(d_bot,Gamma,d_tilde)
feasibility=sum(d_bot+Gamma*d_tilde)
print(feasibility)

def generate_Ay(m,Kapacity):
    if len(Kapacity)==m:
        K=np.diag(Kapacity)
        block2=-np.diag([-1 for _ in range(m)])
        block3=np.array(([0 for _ in range(m)]+[1 for _ in range(m)],))
        middle=np.concatenate((K,block2)).T
        print(middle.shape,block3.shape)
        return np.concatenate((middle,block3))

    else:raise ValueError


print(generate_Ay(m,Kapacity))