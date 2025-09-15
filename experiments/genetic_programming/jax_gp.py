import random
import time
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np

D = 3

POP_SIZE = 20
K_ELITES = 4
N_NEW_MEMBERS = 2
GENERATIONS = 10000
TASK_N_EPOCH = 50
TASK_N_ITER = 500
N_OPS = 10

MAX_EQS = 20
MAX_DEPTH = 10
P_DELETION = 0.025
P_INSERTION = 0.025
P_CROSSOVER = 1.0


# Operators with special exception handling to keep GP well behaved.
def gp_transpose(x):
    x = jnp.array(x)
    x = jnp.nan_to_num(x)
    return x.T


def gp_matmul_inner(x, y):
    x = jnp.array(x)
    x = jnp.nan_to_num(x)
    y = jnp.array(y)
    y = jnp.nan_to_num(y)
    x = gp_matrix(x)
    y = gp_matrix(y)
    return x.T @ y  # type: ignore


def gp_matmul_outer(x, y):
    x = jnp.array(x)
    x = jnp.nan_to_num(x)
    y = jnp.array(y)
    y = jnp.nan_to_num(y)
    x = gp_matrix(x)
    y = gp_matrix(y)
    return x @ y.T  # type: ignore


def gp_matmul(x, y):
    x = jnp.array(x)
    x = jnp.nan_to_num(x)
    y = jnp.array(y)
    y = jnp.nan_to_num(y)
    x = gp_matrix(x)
    y = gp_matrix(y)
    return x @ y  # type: ignore


def gp_identity(x):
    x = jnp.array(x)
    x = jnp.nan_to_num(x)
    return x


def gp_logabs(x):
    x = jnp.array(x)
    x = jnp.nan_to_num(x)
    return jnp.log(jnp.abs(x) + 1e-12)


def gp_sigmoid(x):
    x = jnp.array(x)
    x = jnp.nan_to_num(x)
    return jax.nn.sigmoid(x)


def gp_divide(x, y):
    x = jnp.array(x)
    x = jnp.nan_to_num(x)
    y = jnp.array(y)
    y = jnp.nan_to_num(y) + 1.0
    z = jnp.array(x / y)
    return z


def gp_abspower(x, y):
    x = jnp.array(x)
    x = jnp.nan_to_num(x) + 1.0
    y = jnp.array(y)
    y = jnp.nan_to_num(y)
    z = jnp.power(jnp.abs(x), y)
    return z


def gp_relu(x):
    x = jnp.nan_to_num(x)
    x = jnp.array(x)
    return jax.nn.relu(x)


def gp_softmax(x):
    x = jnp.array(x)
    x = jnp.nan_to_num(x)
    x = gp_matrix(x)
    return jax.nn.softmax(x, 0)  # type: ignore


def gp_sigmoid_bernoulli(x):
    p = gp_sigmoid(x)
    key = jax.random.PRNGKey(int(time.time() * 1000))
    z = jax.random.binomial(key, 1, p=p, shape=p.shape)
    return z


def gp_tranpose(x):
    x = jnp.array(x)
    x = jnp.nan_to_num(x)
    return x.T


# NEED TO FIX THESE CONDITIONALS
# def gp_scalar(x):
#     x = jnp.array(x)
#     x = jnp.nan_to_num(x)
#     if len(x.shape) == 0:
#         return x
#     if len(x.shape) == 1:
#         return jnp.array(x.mean())
#     if len(x.shape) == 2:
#         return jnp.array(x.mean())


# def gp_vector(x):
#     x = jnp.array(x)
#     x = jnp.nan_to_num(x)
#     if len(x.shape) == 0:
#         return jnp.zeros((D,)) + x
#     if len(x.shape) == 1:
#         return x
#     if len(x.shape) == 2:
#         return x.mean(0)


# def gp_matrix(x):
#     x = jnp.nan_to_num(x)
#     if len(x.shape) == 0:
#         return jnp.zeros((D, D)) + x
#     if len(x.shape) == 1:
#         return jnp.zeros((D, D)) + x
#     if len(x.shape) == 2:
#         return x


def gp_scalar(x):
    x = jnp.array(x)
    x = jnp.nan_to_num(x)
    return x.mean()


def gp_vector(x):
    x = jnp.array(x)
    x = jnp.nan_to_num(x)
    x = jnp.zeros((D, D)) + x
    return jnp.mean(x, 0)


def gp_matrix(x):
    x = jnp.nan_to_num(x)
    return jnp.zeros((D, D)) + x


# Generate a number of M V and S
def generate_leaf_vars():
    leaf_vars = []
    var_nums = np.random.randint(1, MAX_EQS, (3,))
    leaf_vars += [f"M_{i}" for i in range(var_nums[0])]
    leaf_vars += [f"V_{i}" for i in range(var_nums[1])]
    scalars = [f"S_{i}" for i in range(var_nums[2])]
    scalars_val = []
    for s in scalars:
        float_val = np.random.randint(0, 10, (1,)).item()
        float_exp = np.random.randint(-4, 3, (1,)).item()
        float_str = f"{float_val}e{float_exp}"
        float_str = list(float_str)
        float_str[2] = "m" if float_str[2] == "-" else float_str[2]
        float_str = "".join(float_str)
        s = s + "_" + float_str
        scalars_val.append(s)
    scalars = scalars_val
    leaf_vars += scalars
    return leaf_vars


# Now sample them with replacement some number of times.
def choose_eqs(leaf_vars, max_eqs=10):
    num_eqs = np.random.randint(1, max_eqs, (1,))
    eqs = list(np.random.choice(leaf_vars, num_eqs, replace=True))
    eqs.append("V_Y")
    return eqs


# Collect leaves
def collect_terminal_keys(d, terminals=None):
    if terminals is None:
        terminals = []
    for key, value in d.items():
        if len(value) > 0:
            # If the value is another dict, recurse into it
            collect_terminal_keys(value, terminals)
        else:
            # If the value is not a dict, this is a terminal key
            terminals.append(key)
    return terminals


# Parse tree as an expression
def tree_to_expression(tree):
    def traverse(node, parent_key=""):
        if not node:
            return parent_key
        if isinstance(node, dict):
            keys = list(node.keys())
            if len(keys) == 0:
                return parent_key
            elif len(keys) == 1:
                return parent_key + "(" + traverse(node[keys[0]], keys[0]) + ")"
            elif len(keys) == 2:
                # Order: Left, Node, Right
                return (
                    parent_key + "(" + traverse(node[keys[0]], keys[0]) + "," + traverse(node[keys[1]], keys[1]) + ")"
                )
            else:
                # Handles cases where there are more than two children (not typical for binary trees)
                result = ""
                for key in keys[:-1]:
                    result += traverse(node[key])
                result += keys[-1] + traverse(node[keys[-1]])
                return result
        return ""

    # Start traversal from the root node
    if tree:
        return traverse(tree)
    return ""


# Generate tree from a set of leaf variables
def generate_tree(symbol_sets, max_depth=5):
    def rtree(depth):
        if depth == 0:
            return {np.random.choice(symbol_sets[0]): {}}
        new_node, node_type = random_node(symbol_sets)
        if node_type in [0, 3]:
            return {new_node: {}}
        elif node_type == 1:
            this_node = {}
            this_node.update(rtree(depth - 1))  # type: ignore
            return {new_node: this_node}
        elif node_type == 2:
            this_node = {}
            left_node = rtree(depth - 1)
            right_node = rtree(depth - 1)
            if right_node.keys() == left_node.keys():
                right_node = {"gp_identity": right_node}
            this_node.update(left_node)
            this_node.update(right_node)
            return {new_node: this_node}

    return rtree(max_depth)


# Random node selection
def random_node(symbol_sets):
    # set_idx = jax.random.randint(0,len(symbol_sets),(1,)).item()
    set_idx = np.random.choice([0, 1, 2, 3])
    sel_type = np.random.choice(symbol_sets[set_idx])
    return sel_type, set_idx


# Print tree
def print_tree(d, prefix=""):
    for key, value in d.items():
        current_path = f"{prefix}.{key}" if prefix else key
        print(current_path)
        if isinstance(value, dict):
            print_tree(value, current_path)


# Turn a particular tree into an expression and run it.
def run_tree(tree):
    print_tree(tree)
    expr = tree_to_expression(tree)
    return eval(expr)


# Needed to constrain types.
def var_to_final_op(var):
    var_type = var[0]
    match var_type:
        case "S":
            return "gp_scalar"
        case "V":
            return "gp_vector"
        case "M":
            return "gp_matrix"


# So now we have leaf_vars and eqs. Add operators
symbols_ops1 = [
    "gp_logabs",
    "jnp.exp",
    "jnp.tanh",
    "gp_sigmoid",
    "gp_relu",
    "gp_softmax",
    # "gp_sigmoid_bernoulli",
    "jnp.sign",
    "gp_transpose",
    "gp_matrix",
    "gp_vector",
    "gp_scalar",
]
symbols_ops2 = [
    "jnp.add",
    "jnp.subtract",
    "jnp.multiply",
    "gp_abspower",
    "gp_divide",
    "gp_matmul_inner",
    "gp_matmul_outer",
    "gp_matmul",
]
symbols_fixed = ["X", "Y"]


# The task is structured as follows:
# First present X with Y set to 0 for a few iterations (minus phase)
# Then present X with Y not zero for a few iterations (plus phase)
# Use the minus phase V_Y.


def initialize_state(varnames):
    state = {}
    for s in varnames:
        parts = s.split("_")
        if len(parts) > 1:
            vtype = parts[0]
        else:
            continue
        match vtype:
            case "S":
                vargs = parts[2:]
                arg0 = vargs[0]  # Assuming the number is always the last part
                arg0 = list(arg0)
                arg0[2] = "-" if arg0[2] == "m" else arg0[2]
                arg0 = "".join(arg0)
                state[s] = jnp.array(float(arg0))
            case "M":
                key = jax.random.PRNGKey(int(time.time() * 1000))
                state[s] = jax.random.normal(key, (D, D))
            case "V":
                key = jax.random.PRNGKey(int(time.time() * 1000))
                state[s] = jax.random.normal(key, (D,))
            case _:
                TypeError(f"{s} not parsed.")
    return state


def generate_model():
    # Run entire system of expressions
    leaf_vars = generate_leaf_vars() + ["V_Y"]
    symbol_sets = [leaf_vars, symbols_ops1, symbols_ops2, symbols_fixed]
    eqs = choose_eqs(leaf_vars, max_eqs=MAX_EQS)
    tree_set = [generate_tree(symbol_sets, max_depth=MAX_DEPTH) for _ in eqs]
    tree_set_final = [{var_to_final_op(var): tree} for var, tree in zip(eqs, tree_set, strict=True)]
    leaf_vars = leaf_vars + ["X", "Y"]
    exprs = [tree_to_expression(x) for x in tree_set_final]
    exprs_lambda = [f"lambda {', '.join(leaf_vars)}: {x}" for x in exprs]
    funcs = [{name: eval(x)} for name, x in zip(eqs, exprs_lambda, strict=True)]
    state = initialize_state(leaf_vars)
    return state, funcs, exprs, leaf_vars, eqs, tree_set


def build_model_from_genome(leaf_vars, eqs, tree_set):
    # Run entire system of expressions
    Gf = [{var_to_final_op(var): tree} for var, tree in zip(eqs, tree_set, strict=True)]
    exprs = [tree_to_expression(x) for x in Gf]
    exprs_lambda = [f"lambda {', '.join(leaf_vars)}: {x}" for x in exprs]
    funcs = [{name: eval(x)} for name, x in zip(eqs, exprs_lambda, strict=True)]
    state = initialize_state(leaf_vars)
    return state, funcs, exprs, leaf_vars, eqs, tree_set


def collect_nodes(dict_tree, node_list):
    # Append the current node to the list
    node_list.append(dict_tree)
    # Recursively collect nodes from child dictionaries
    for key, value in dict_tree.items():
        if isinstance(value, dict) and len(value) > 0:
            collect_nodes(value, node_list)


# MUTATION
def crossover(tree1, tree2):
    nodes1 = []
    collect_nodes(tree1, nodes1)
    nodes2 = []
    collect_nodes(tree2, nodes2)
    node_swap_1 = random.choice(nodes1)
    node_swap_2 = random.choice(nodes2)
    remaining_key_1 = "n"
    remaining_key_2 = "n"
    if len(node_swap_1) < 2:
        branch1 = deepcopy(node_swap_1)
        node_swap_1.clear()
    else:
        node_swap_name_1 = random.choice(list(node_swap_1))
        branch1 = deepcopy({node_swap_name_1: node_swap_1[node_swap_name_1]})
        del node_swap_1[node_swap_name_1]
        remaining_key_1 = list(node_swap_1)[0]
    if len(node_swap_2) < 2:
        branch2 = deepcopy(node_swap_2)
        node_swap_2.clear()

    else:
        node_swap_name_2 = random.choice(list(node_swap_2))
        branch2 = deepcopy({node_swap_name_2: node_swap_2[node_swap_name_2]})
        del node_swap_2[node_swap_name_2]
        remaining_key_2 = list(node_swap_2)[0]

    # Handle the exception of having the same name:
    if remaining_key_1 == list(branch2)[0]:
        if remaining_key_1 == "gp_identity":
            while list(branch2)[0] == "gp_identity":
                branch2 = branch2["gp_identity"]
        else:
            branch2 = {"gp_identity": branch2}
    if remaining_key_2 == list(branch1)[0]:
        if remaining_key_2 == "gp_identity":
            while list(branch1)[0] == "gp_identity":
                branch1 = branch1["gp_identity"]
        else:
            branch1 = {"gp_identity": branch1}
    node_swap_1.update(branch2)
    node_swap_2.update(branch1)


def deletion(tree):
    nodes = []
    collect_nodes(tree, nodes)
    node_del = random.choice(nodes)
    if len(node_del) == 1:
        node_del.clear()
        node_del.update({"jnp.zeros((1,))": {}})
    else:
        node_del.clear()
        node_del.update({"jnp.zeros((1,))": {}, f"jnp.zeros(({D},))": {}})


def eq_insertion(eq1, eq2, G1, G2):
    idx = random.randint(0, len(eq2) - 1)
    eq2_swap = deepcopy(eq2[idx])
    g2_swap = deepcopy(G2[idx])
    idx_insert = random.randint(0, len(eq1) - 1)
    eq1.insert(idx_insert, eq2_swap)
    G1.insert(idx_insert, g2_swap)


def eq_deletion(eq, G):
    if len(eq) > 1:
        idx = random.randint(0, len(eq) - 2)
        _ = eq.pop(idx)
        _ = G.pop(idx)


def eq_crossover(eq1, eq2, G1, G2):
    if len(eq1) > 1 and len(eq2) > 1:
        eq1_swap_idx = random.randint(0, len(eq1) - 2)
        eq2_swap_idx = random.randint(0, len(eq2) - 2)

        def _swap(x1, x2):
            x1_swap = deepcopy(x1[eq1_swap_idx])
            x2_swap = deepcopy(x2[eq2_swap_idx])
            x2.insert(eq2_swap_idx, x1_swap)
            x1.insert(eq1_swap_idx, x2_swap)
            _ = x1.pop(eq1_swap_idx + 1)
            _ = x2.pop(eq2_swap_idx + 1)

        _swap(eq1, eq2)
        _swap(G1, G2)


population = [generate_model() for _ in range(POP_SIZE)]
key = jax.random.PRNGKey(int(time.time() * 1000))


for gen in range(GENERATIONS):
    print("GENERATION:", gen)
    print(len(population))
    states = [state for state, _, _, _, _, _ in population]
    funcs = [func for _, func, _, _, _, _ in population]
    eqs = [eqs for _, _, _, _, eqs, _ in population]
    leaves = [leaves for _, _, _, leaves, _, _ in population]
    genomes = [genome for _, _, _, _, _, genome in population]

    @jax.jit
    def problem(key, states):
        def iterate_state(state, idx):
            new_state = state
            for F in funcs[idx]:
                k = list(F)[0]
                f = F[k]
                new_state[k] = f(**state)
            return new_state

        def forward(x, y, state, idx):
            new_state = state
            new_state["X"] = x
            new_state["Y"] = y
            new_state = iterate_state(new_state, idx)
            return new_state

        fitnesses = []
        for idx, state in enumerate(states):
            state["X"] = jnp.zeros((D,))
            state["Y"] = jnp.zeros((D,))

            def problem_scan(carry, _):
                key, state_scan, b = carry
                key, subkey = jax.random.split(key)
                X = jax.random.normal(subkey, (D,))
                Y = X * 0
                state_scan = forward(X, Y, state_scan, idx)
                V_Y_eval = state_scan["V_Y"]
                key, subkey = jax.random.split(key)
                err = jax.random.normal(subkey, (D,)) * 0.5
                Y = X * b + err + 5
                fitness = jnp.mean((V_Y_eval - Y) ** 2)  # type: ignore
                state_scan = forward(X, Y, state_scan, idx)
                carry = (key, state_scan, b)
                return carry, fitness

            def problem_scan_2(carry, _):
                key, state_scan, b = carry
                key, subkey = jax.random.split(key)
                X = jax.random.normal(subkey, (D,))
                Y = jnp.zeros((D,))
                state_scan = forward(X, Y, state_scan, idx)
                V_Y_eval = state_scan["V_Y"]
                key, subkey = jax.random.split(key)
                err = jax.random.normal(subkey, (D,)) * 0.5
                Y = b * X + err
                fitness = jnp.mean((V_Y_eval - Y) ** 2)  # type: ignore
                state_scan = forward(X, Y, state_scan, idx)
                carry = (key, state_scan, b)
                return carry, fitness

            def epoch_scan(carry, _):
                key = carry
                key, subkey = jax.random.split(key)
                b = jax.random.uniform(subkey, (1,), minval=-5, maxval=5)
                key, subkey = jax.random.split(key)
                B = jnp.array([b[0]])
                B = jax.random.choice(key, B)
                carry_in = (subkey, state, B)
                carry_out, fitness = jax.lax.scan(problem_scan_2, init=carry_in, length=TASK_N_ITER)
                key = carry_out[0]
                carry = key
                return carry, fitness.mean()

            _, fitness = jax.lax.scan(epoch_scan, init=key, length=TASK_N_EPOCH)
            fitnesses.append(fitness.mean())
            # fit_total = []
            # for epoch in range(TASK_N_EPOCH):
            # key, subkey = jax.random.split(key)
            # b = jax.random.uniform(key, (1,), minval=-10, maxval=10)
            # carry = (key, state, b)
            # carry, fitness = jax.lax.scan(problem_scan, init=carry, length=TASK_N_ITER)
            # fitnesses.append(fitness.mean())

        return fitnesses

    print("Running generation")
    t0 = time.time()
    key, subkey = jax.random.split(key)
    fits = problem(subkey, states)
    print("Run time:", time.time() - t0)

    # To get a new generation,
    fits = -jnp.stack(fits)
    print("BEST_FITNESS:", jnp.max(fits))

    # Take top K winners.
    topk = jax.lax.top_k(fits, K_ELITES)[1]
    top1 = fits.argmax()

    print("Best model:")
    for v, eq in zip(population[top1][4], population[top1][2], strict=True):
        print(f"\t{v}={eq}")

    new_population = [population[idx] for idx in topk]
    for _ in range(N_NEW_MEMBERS):
        new_population.append(generate_model())
    n_children = POP_SIZE - K_ELITES - N_NEW_MEMBERS
    children = []
    for _ in range(n_children):
        m1 = deepcopy(random.choice(new_population))
        m2 = deepcopy(random.choice(new_population))
        leaves = list(set.union(set(m1[3]), set(m2[3])))
        for _ in range(N_OPS):
            if random.uniform(0, 1) < P_CROSSOVER:
                eq_crossover(m1[4], m2[4], m1[5], m2[5])
            if random.uniform(0, 1) < P_DELETION:
                eq_deletion(m1[4], m1[5])
            if random.uniform(0, 1) < P_INSERTION:
                eq_insertion(m1[4], m2[4], m1[5], m2[5])

        if len(m1[5]) > 1:
            for _ in range(N_OPS):
                t1, t2 = random.sample(m1[5], k=2)
                if random.uniform(0, 1) < P_CROSSOVER:
                    crossover(t1, t2)
                if random.uniform(0, 1) < P_DELETION:
                    deletion(t1)
                if random.uniform(0, 1) < P_DELETION:
                    deletion(t2)

        children.append(build_model_from_genome(leaves, m1[4], m1[5]))
    population = new_population + children

"""
S_2_5e2 = gp_abspower(V_0, S_1_9em4)
V_0 = X
V_Y = X * jnp.exp(gp_sigmoid(S_2_5e2))-jnp.exp(X)@Y
"""

"""
S_2_5e2=(gp_scalar(gp_abspower(V_3,V_0)))
V_0=(gp_vector(gp_sigmoid(X)))
V_Y= X@exp(gp_sigmoid(S_2_5e2)))-exp(V_0)@Y
"""
