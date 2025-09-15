import time

import numpy as np

D = 3


# Operators with special exception handling to keep GP well behaved.
def gp_matmul(x, y):
    x = np.array(x)
    x = np.nan_to_num(x)
    y = np.array(y)
    y = np.nan_to_num(y)

    if len(x.shape) > 0 and len(y.shape) > 0:
        return x @ y
    else:
        return x * y


def gp_identity(x):
    x = np.array(x)
    x = np.nan_to_num(x)

    return x


def gp_logabs(x):
    x = np.array(x)
    x = np.nan_to_num(x)

    return np.log(np.abs(x) + 1e-12)


def gp_sigmoid(x):
    x = np.array(x)
    x = np.nan_to_num(x)

    return 1.0 / (1.0 + np.exp(-x))


def gp_divide(x, y):
    x = np.array(x)
    x = np.nan_to_num(x)
    y = np.array(y)
    y = np.nan_to_num(y) + 1.0

    z = np.array(x / y)
    return z


def gp_abspower(x, y):
    x = np.array(x)
    x = np.nan_to_num(x) + 1.0
    y = np.array(y)
    y = np.nan_to_num(y)

    z = np.power(np.abs(x), y)
    return z


def gp_relu(x):
    x = np.nan_to_num(x)
    x = np.array(x)
    x[x < 0] = 0
    return x


def gp_softmax(x):
    x = np.array(x)
    x = np.nan_to_num(x)

    xsum = np.sum(np.exp(x), 0)
    return np.exp(x) / xsum


def gp_sigmoid_bernoulli(x):
    p = gp_sigmoid(x)
    z = np.random.binomial(1, p=p, size=p.shape)
    return z


def gp_scalar(x):
    x = np.array(x)
    x = np.nan_to_num(x)
    if len(x.shape) == 0:
        return x
    if len(x.shape) == 1:
        return np.array(x.mean())
    if len(x.shape) == 2:
        return np.array(x.mean())


def gp_vector(x):
    x = np.array(x)
    x = np.nan_to_num(x)
    if len(x.shape) == 0:
        return np.zeros((D,)) + x
    if len(x.shape) == 1:
        return x
    if len(x.shape) == 2:
        return x.mean(0)


def gp_matrix(x):
    x = np.nan_to_num(x)
    if len(x.shape) == 0:
        return np.zeros((D, D)) + x
    if len(x.shape) == 1:
        return np.zeros((D, D)) + x
    if len(x.shape) == 2:
        return x


# Generate a number of M V and S
def generate_leaf_vars():
    leaf_vars = []
    var_nums = np.random.randint(1, 5, (3,))
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
    # set_idx = np.random.randint(0,len(symbol_sets),(1,)).item()
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
    "np.exp",
    "np.tanh",
    "gp_sigmoid",
    "gp_relu",
    # "gp_softmax",
    "gp_sigmoid_bernoulli",
    "np.sign",
]
symbols_ops2 = [
    "np.add",
    "np.subtract",
    "np.multiply",
    "gp_abspower",
    "gp_divide",
    # "gp_matmul",
]
symbols_fixed = ["X", "Y"]


# The task is structured as follows:
# First present X with Y set to 0 for a few iterations (minus phase)
# Then present X with Y not zero for a few iterations (plus phase)
# Use the minus phase V_Y.


def initialize_state(varnames, d=3):
    state = {}
    for s in varnames:
        parts = s.split("_")
        if len(parts) > 1:
            vtype = parts[0]
        else:
            # print(f"Skipping {parts[0]}")
            continue
        match vtype:
            case "S":
                vargs = parts[2:]
                arg0 = vargs[0]  # Assuming the number is always the last part
                arg0 = list(arg0)
                arg0[2] = "-" if arg0[2] == "m" else arg0[2]
                arg0 = "".join(arg0)
                state[s] = np.array(float(arg0))
            case "M":
                state[s] = np.random.normal(0, 1, (d, d))
            case "V":
                state[s] = np.random.normal(0, 1, (d,))
            case _:
                TypeError(f"{s} not parsed.")
    return state


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


def generate_model():
    # Run entire system of expressions
    leaf_vars = generate_leaf_vars()
    symbol_sets = [leaf_vars, symbols_ops1, symbols_ops2, symbols_fixed]
    eqs = choose_eqs(leaf_vars, max_eqs=10)
    G = [generate_tree(symbol_sets, max_depth=5) for _ in eqs]
    Gf = [{var_to_final_op(var): tree} for var, tree in zip(eqs, G, strict=True)]

    leaf_vars = leaf_vars + ["X", "Y", "V_Y"]
    exprs = [tree_to_expression(x) for x in Gf]
    exprs_lambda = [f"lambda {', '.join(leaf_vars)}: {x}" for x in exprs]
    funcs = [{name: eval(x)} for name, x in zip(eqs, exprs_lambda, strict=True)]
    state = initialize_state(leaf_vars, d=3)
    return state, funcs, exprs, eqs


# Try some different models.
def problem(state, funcs):
    for epoch in range(50):
        print(epoch)
        b = np.random.uniform(-5, 5, (1,))
        fit = []  # fitness is evaluated as of the last episode.
        for _ in range(200):
            X = np.random.normal(0, 1, (D,))
            Y = X * 0
            state = forward(X, Y, state, funcs)
            V_Y_eval = state["V_Y"]

            Y = X * b
            state = forward(X, Y, state, funcs)

            fitness = -np.mean((V_Y_eval - Y) ** 2)  # type: ignore
            fit.append(fitness)
    return np.mean(fit)


population = [generate_model() for _ in range(50)]

print("Running generation")
t0 = time.time()

scores = []
for state, func, exprs, eqs in population:
    # print("Model:")
    # for eq, expr in zip(eqs, exprs, strict=True):
    # print(f"\t{eq}={expr}")
    score = problem(state, func)
    scores.append(score)
    # print("Fitness:", score)
print(scores)

print("Run time:", time.time() - t0)

# For regression:
# Need to produce a function Y = model(X)
# We run 'n' iterations of Y=mX for each value of 'm'. Give the final evaluation.
# Evaluate total fitness -MSE(Y, VY)


# How his works:
# (1) Define set of leaf_vars (M_0, M_1, V_0, V_1, V_2, S_0, S_1)
# (2) Generate system chromosome by sampling leaf_vars with replacement, to put them in a certain order with repeats
# (3) Generate tree per var with leaves drawing from random var set with replacement.


# To do:
# Interface with input X and output Y.
#    Include both as leaf variables.
#    Include Y as (last?) equation.
#        How to interface dimensionality of X and Y
# Need a separate system: X, fX, Y, fY, and fU and fV?
#   fX and fY map X and Y to  leaf node dimensions.
#   fZ maps leaf node dimensions to Y.
#   Do they evolve separately from the rest?
# Problem is, things have different output formats and activations (RL vs supervised).
# If we don't want to resort to other standard training methods (sort of defeating the whole purpose)
# then we have to be able to evolve operations over the input and output weights that can possibly train them.
# The delta rule is the simplest example.

# One thought is that we can break the input/output apart into overlapping pieces of size D.
# Then simply include those pieces as leaf nodes.
# What if size of output < D? Pad the output.
# This leaves open the possibility of convolution type weight sharing.
# What about Conv2D?
# An interface for breaking say, Y (3, 255, 255) into matrices that flatten to D-sized vectors.
# Should be possible from the 1D case without many steps.
# This is fair enough for input, what about outputting to dimension Y?

# For now, I will probably just use multi-armed bandit with a length D tag for input purposes
# and do some simple operation to obtain the policy.
# Or even better: just do a D dimensional X->Y prediction problem.


# Evolution questions
# How are nodes selected for genetic operators?


# How to evaluate? Have X, Y, V_Y.
