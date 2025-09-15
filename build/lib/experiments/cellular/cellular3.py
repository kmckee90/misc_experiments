import jax
import jax.numpy as jnp
import jax.random as random
import jax.nn as nn
from PIL import Image
import numpy as np

key = random.PRNGKey(123)

@jax.jit
def gen_data(key):
    dim = DIM
    x = jnp.zeros((dim, dim), jnp.float32)
    idx = random.randint(key, (4,), 0, dim)
    x = x.at[idx[0],:].set(1)
    x = x.at[:,idx[1]].set(1)
    x = x.at[idx[2],:].set(1)
    x = x.at[:,idx[3]].set(1)
    return x

DIM = 12
BIAS = -2
data_dim = DIM
d_i = data_dim*data_dim
d_h = 64
d_o = d_i
L = 1
lr = 0.1
decay = 0.0
temp = 0.01


#Model pars
key, subkey = random.split(key)
data_vals = random.uniform(key, (data_dim, data_dim))
initial_sd = 1.0
_WS = random.normal(key, (L-1, d_h, d_h))*initial_sd
key, subkey = random.split(key)
_WB = random.normal(key, (L-1, d_h, d_h))*initial_sd
key, subkey = random.split(key)
_WSi = random.normal(key, (d_i, d_h))*initial_sd
key, subkey = random.split(key)
_WSo = random.normal(key, (d_h, d_o))*initial_sd
key, subkey = random.split(key)
_WBi = random.normal(key, (d_i, d_h))*initial_sd

S = jnp.zeros((L, d_h))
B = jnp.zeros((L, d_h))

#Forward
@jax.jit
def forward(key, X, Y, WSi, WSo, WBi, WS, WB):#, temp=1.0, lr=0.001, decay=0.0):
    w_decay = 1-decay
    Si = X.flatten() #Input
    Bi = Y.flatten() #Target
    S = jnp.zeros((L, d_h))
    B = jnp.zeros((L, d_h))

    #FORWARD
    h = nn.sigmoid(Si @ WSi / temp + BIAS)
    S = S.at[0].set(h)
    for i in range(L-1):
        h = nn.sigmoid(S[i] @ WS[i] / temp + BIAS)
        S = S.at[i+1].set(h)
    h = nn.sigmoid(S[-1] @ WSo / temp + BIAS)
    So = h

    #BACKWARD
    h = nn.sigmoid(Bi @ WBi / temp + BIAS)
    B = B.at[0].set(h)
    for i in range(L-1):
        h = nn.sigmoid(B[i] @ WB[i] / temp + BIAS)
        B = B.at[i+1].set(h)

    #Weight updates: this is confusing af
    WSi = w_decay*WSi + lr*(jnp.outer(Si, B[-1]) - jnp.outer(Si, S[0]))
    WSo = w_decay*WSo + lr*(jnp.outer(S[0], Bi) - jnp.outer(S[-1], So))
    WBi = w_decay*WBi + lr*(jnp.outer(Bi, B[0]) - jnp.outer(So, S[-1]))      
    WS = w_decay*WS + lr*(jax.vmap(jnp.outer)(S[:-1], jnp.flip(B, 0)[1:]) - jax.vmap(jnp.outer)(S[:-1], S[1:]))
    WB = w_decay*WB + lr*(jax.vmap(jnp.outer)(B[:-1], B[1:]) - jax.vmap(jnp.outer)(jnp.flip(S, 0)[:-1], jnp.flip(S, 0)[1:]))
    return So, Bi, B, S, WSi, WSo, WBi, WS, WB


mse = 0
log_iter = 500
for iter in range(500000):
    #Initial hidden activities
    key, subkey = random.split(key)
    _X = gen_data(key)
    _X = _X * data_vals #Continuous value variation
    _X = _X.at[0,0].set(1.0) #sacrifice one for easy bias implementation
    _Y = _X.T
    
    key, subkey = random.split(key)
    _So, _Bi, B, S, _WSi, _WSo, _WBi, _WS, _WB = forward(key, _X, _Y, _WSi, _WSo, _WBi, _WS, _WB)

    #Log    
    mse = mse + jnp.abs(_Y.flatten()-_So).mean()/log_iter
    if iter % log_iter == 0:
        print(f"Iter:{iter}, MAE:{mse}")
        mse = 0

    if iter % (log_iter) == 0:
        imgS = np.array(_So.reshape(data_dim, data_dim))
        imgB = np.array(_Bi.reshape(data_dim, data_dim))
        imgS[imgS<0] = 0
        imgS = (imgS / imgS.max() * 255).astype('uint8')
        imgB = (imgB / imgB.max() * 255).astype('uint8')
        imgS = Image.fromarray(imgS)  
        imgB = Image.fromarray(imgB) 
        imgS.save('cellular/imgS.png')
        imgB.save('cellular/imgB.png')
