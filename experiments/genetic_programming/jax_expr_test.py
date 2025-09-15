import time

import jax
import jax.numpy as jnp

expr = "4*jax.random.bernoulli(key, p=jax.nn.sigmoid(2.0*(x@W)-1))"
n = 15000


def func2(key):
    x = jax.random.normal(key, (n,))
    W = jax.random.normal(key, (n, n))
    Y = eval(expr)
    return Y


@jax.jit
def func3(key):
    x = jax.random.normal(key, (n,))
    W = jax.random.normal(key, (n, n))
    Y = eval(expr)
    return Y


this_time = int(time.time() * 10)
key = jax.random.PRNGKey(this_time)
u = jax.random.normal(key, (n,))
V = jax.random.normal(key, (n, n))


t0 = time.time()
func0(u, V, key)
print(time.time() - t0)

t0 = time.time()
func1(u, V, key)
print(time.time() - t0)


this_time = int(time.time() * 10)
key = jax.random.PRNGKey(this_time)
t0 = time.time()
func2(key)
print(time.time() - t0)


this_time = int(time.time() * 10)
key = jax.random.PRNGKey(this_time)
t0 = time.time()
func3(key)
print(time.time() - t0)


def new_func(eqs, exprs):
    @jax.jit
    def run_func():
        state = {}
        state["V_0"] = jnp.ones((3,))
        for step in range(5):
            for i, leaf_var in enumerate(eqs):
                print(state)
                state[leaf_var] = eval(exprs[i])
        return state["V_Y"]

    return run_func()


funcs = []
for i in range(10):
    exprV0 = "state['V_0']+i/10"
    exprVY = "state['V_0']"
    eqs = ["V_0", "V_Y"]
    exprs = [exprV0, exprVY]

    def run_func():
        state = {}
        state["V_0"] = jnp.ones((3,))
        for step in range(5):
            for i, leaf_var in enumerate(eqs):
                print(state)
                state[leaf_var] = eval(exprs[i])
        return state["V_Y"]

    funcs.append(run_func)


def run_this_func(f):
    return f()


jax.vmap(run_this_func)(funcs)
jnp.array(funcs)

jax.pmap(run_this_func)(funcs)

jnp.array("s", dtype=str)

import numpy as np

sa = np.array(["a", "b"])
np.int32(sa)


# Define some sample functions
def f(x):
    return x * 2


def g(x):
    return x + 10


def h(x):
    return x**2


# List of functions
funcs = [f, g, h]

# Example inputs corresponding to each function
inputs = jnp.array([1, 2, 3])


# Wrapper function to apply a function to an input
def apply_func(func, x):
    return func(x)


# Vectorize the wrapper function across multiple inputs and functions
parallel_apply = jax.vmap(apply_func, in_axes=(0, 0))

# Run the functions in parallel
outputs = parallel_apply(jnp.array(funcs), inputs)

print(outputs)  # Expected output: array([ 2, 12, 9])


# JIT compile the functions
f_jit = jax.jit(f)
g_jit = jax.jit(g)
h_jit = jax.jit(h)

# Inputs for each function
inputs = [1, 2, 3]

# Execute each JIT-compiled function with its input
outputs = [f_jit(inputs[0]), g_jit(inputs[1]), h_jit(inputs[2])]


#this is the population life:
#Take exprs

#Nothing static here
def iterate_state(state, funcs):
    new_state = state
    for F in funcs:
        k = list(F)[0]
        f = F[k]
        new_state[k] = f(**state)
    return new_state

def forward(x, y, state, funcs):
    state["X"] = x
    state["Y"] = y
    state = iterate_state(state, funcs)
    return state

#This is all generated
variables = ["X","Y", "V", "W", "VY"]
eqs = ["V", "V", "V", "W", "VY"]
exprs = ["jax.nn.sigmoid(X @ W)", "jax.nn.relu(V @ W)+X", "jax.nn.tanh(V @ W + X)", "W+0.1", "V @ W"]
state = {"V":jnp.zeros((3,))+.1, "W":jnp.zeros((3,3))+.1, "VY":jnp.zeros((3,))}

#Compiling:
exprs_lambda = [f"lambda {', '.join(variables)}: {x}" for x in exprs]
funcs = [{name: eval(x)} for name, x in zip(eqs, exprs_lambda)]

#Problem operates here:
X = jnp.zeros((3,))
Y = jnp.zeros((3,))
#Run several times?
state = forward(X, Y, state, funcs)




#What is the order for running things?
# Initialize state
# Initialize funcs


def run_eqs(state, exprs):
    funcs = [eval(f"lambda X: {x}") for x in exprs]
    for eq in eqs:




@jax.jit
def run_expr(idx, x, w):
    return jax.lax.switch(idx, funcs, (x, w))


key = jax.random.PRNGKey(123)
_X = jax.random.normal(key, (3,)) + 1
_W = jax.random.normal(key, (3, 3))

idxes = jnp.arange(len(funcs))
jax.vmap(run_expr, in_axes=(0, None, None))(idxes, _X, _W)

run_expr(2, _X, _W)

# evolutionary funcs need to take the input, output, and supply fitness
function_code =    """
lambda X, Y:
    X = 0.1 + X
    Y = 0.9*Y + X
"""

funcs = [exec(function_code)]

def create_function():
    # Define function code with multiple lines
    function_code = """
def new_func0(x):
    result = x + 10
    result *= 2
    return result
"""
    exec(function_code, globals(), locals())
    return locals()['new_func0']

# Get the dynamically created function
my_function = create_function()
print(my_function(5))  # Output will be 30
