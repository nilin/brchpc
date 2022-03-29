import sys
import jax
import jax.numpy as jnp
import time

def log(*x,printtime=True):
  x=' '.join([str(s) for s in x])
  t=time.ctime(time.time()) if printtime else ''
  string=x+' | '+t
  print(string)
  with open('log','a') as f:
    f.write(string+'\n')

log(jax.devices()[0])

n=int(sys.argv[1])
m=int(sys.argv[2])
its=int(sys.argv[3])
log(n)
key=jax.random.PRNGKey(0)
key0,*keys=jax.random.split(key,3*its)
A=[jax.random.normal(keys[2*i],(n,m)) for i in range(its)]
B=[jax.random.normal(keys[2*i+1],(m,n)) for i in range(its)]
t0=time.perf_counter()
for i in range(its):
	log(A[i].shape,'*',B[i].shape,'| iteration ',i,' | ',round(i/(time.perf_counter()-t0)),' per second')
	C=jnp.matmul(A[i],B[i])


t1=time.perf_counter()
log(t1-t0)
log(its)
