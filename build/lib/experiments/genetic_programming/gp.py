import jax
import jax.numpy as jnp
import jax.random as random
import jax.nn as nn
from PIL import Image
import numpy as np

key = random.PRNGKey(123)
d = 8
X = random.normal(key, (d,))
Y = 2*X

#Example genome: Linear model with the delta rule
G2 = {}
G2["np.add"]= {"M_W0":{}, "np.multiply":{}}
G2["np.add"]["np.multiply"]={"S_LR_0001":{}, "np.matmul":{}}
G2["np.add"]["np.multiply"]["np.matmul"]={"X":{},"np.subtract":{}}
G2["np.add"]["np.multiply"]["np.matmul"]["np.subtract"]={"Y":{}, "np.matmul":{}}
G2["np.add"]["np.multiply"]["np.matmul"]["np.subtract"]["np.matmul"]={"M_W0":{}, "X":{}}

G1 = {}
G1["np.matmul"]={"X":{}, "M_W0":{}}

G = {"M_W0":G1, "Y":G2}



#Decode the above.
#First pull out terminals
def collect_terminal_keys(d, terminals=None):
    if terminals is None:
        terminals = []
    for key, value in d.items():
        if len(value)>0:
            # If the value is another dict, recurse into it
            collect_terminal_keys(value, terminals)
        else:
            # If the value is not a dict, this is a terminal key
            terminals.append(key)
    return terminals



#Can parse codes this way:
def parse_var(s: str):
    parts = s.split('_')
    if len(parts)>1:
        vtype = parts[0]
        vname = parts[1]
        vargs = parts[2:]
        vname = s
    else:
        print(f"Skipping {parts[0]}")
        return None
    match vtype:
        case "S":
            arg0 = vargs[0]  # Assuming the number is always the last part
            arg0 = arg0[0]+"."+arg0[1:] if arg0[0] == '0' else arg0
            globals()[vname]=np.array(float(arg0)) 
        case "M":
            globals()[vname]=np.random.normal(0,1,(d, d))
        case "V":
            globals()[vname]=np.random.normal(0,1, (d,))
        case _:
            TypeError(f"{vname} not parsed.")
            


def custom_traverse(tree):
    def traverse(node, parent_key=''):
        if not node:
            return parent_key
        if isinstance(node, dict):
            keys = list(node.keys())
            if len(keys) == 0:
                return parent_key
            elif len(keys) == 1:
                return parent_key + '(' + traverse(node[keys[0]], keys[0]) + ')'
            elif len(keys) == 2:
                # Order: Left, Node, Right
                return parent_key + '('+traverse(node[keys[0]], keys[0])+ "," + traverse(node[keys[1]], keys[1])+')'
            else:
                # Handles cases where there are more than two children (not typical for binary trees)
                result = ''
                for key in keys[:-1]:
                    result += traverse(node[key])
                result += keys[-1] + traverse(node[keys[-1]])
                return result
        return ''
    
    # Start traversal from the root node
    if tree:
        return traverse(tree)
    return ''

def gp_matmul(x, y):
    if len(x.shape) > 0 and len(y.shape) > 0:
        return x @ y
    else:
        return x * y
    
def gp_identity(x):
    return x

def gp_logabs(x):
    return np.log(np.abs(x)+1e-12)

def gp_sigmoid(x):
    return 1/(1+np.exp(-x))

def gp_divide(x, y):
    z = np.array(x/y)
    z[np.abs(z)==np.Infinity] = 0.0
    return z

def gp_relu(x):
    x = np.array(x)
    x[x<0]=0
    return x

def gp_softmax(x):
    return np.exp(x)/np.sum(np.exp(x), 0)

def gp_sigmoid_bernoulli(x):
    x = np.array(x)
    x = np.nan_to_num(x)
    p = gp_sigmoid(x)
    z = np.random.binomial(1, p=p, size=p.shape)
    return z

#This will be better done in jax to deal with array depth
# def gp_softmax_categorical(x):
#     p = gp_softmax(x)
#     z = np.random.multinomial(1, pvals=p)
#     return z


symbols_obj = ["M_","V_","S_"]
symbols_ops1 = ["gp_logabs", "np.exp", "np.tanh", "gp_sigmoid", "gp_relu", "gp_softmax", "gp_sigmoid_bernoulli"]
symbols_ops2 = ["np.add", "np.subtract", "np.multiply", "np.power", "gp_divide", "gp_matmul"]
symbols_fixed = ["X","Y"]
symbol_sets = [symbols_obj, symbols_ops1, symbols_ops2, symbols_fixed]


def random_node():
    set_idx = np.random.randint(0,len(symbol_sets),(1,)).item()
    # set_idx = np.random.choice([0,2])
    sel_type = np.random.choice(symbol_sets[set_idx])
    if set_idx == 0:
        binary_name = np.random.randint(0,2,(3,))
        binary_str = ''.join(binary_name.astype(str).astype(int).astype(str))
        sel_type_out = sel_type + binary_str
        if sel_type == "S_":
            binary_name = np.random.randint(0,2,(4,))
            binary_str = ''.join(binary_name.astype(str).astype(int).astype(str))
            sel_type_out = sel_type_out + '_'+binary_str
        sel_type = sel_type_out
    return sel_type, set_idx


def generate_tree(max_depth):
    def rtree(depth):
        if depth == 0:
            return {"S_END_000":{}}
        new_node, node_type = random_node()
        if node_type in [0,3]:
            return {new_node: {}}
        elif node_type == 1:
            this_node = {}
            this_node.update(rtree(depth - 1))
            return {new_node: this_node}
        elif node_type == 2:
            this_node = {}
            left_node = rtree(depth - 1)
            if list(left_node)[0] == "S_END_000": #If left node is 0, then so is right node.
                right_node = {"S_END2_000":{}}
            else:
                right_node = rtree(depth - 1)        
                if right_node.keys() == left_node.keys():
                    right_node = {"gp_identity": right_node}   
            this_node.update(left_node)
            this_node.update(right_node)
            return {new_node: this_node}
    return rtree(max_depth)


def print_dict_keys(d, prefix=''):
    for key, value in d.items():
        current_path = f"{prefix}.{key}" if prefix else key
        print(current_path)
        if isinstance(value, dict):
            print_dict_keys(value, current_path)
            
def run_tree(tree):
    print_dict_keys(tree)
    terminals = collect_terminal_keys(tree)
    terminals = list(set(terminals))
    for var in terminals:
        parse_var(var)
    exprs = []
    for expr in list(tree):
        exprs.append(custom_traverse(tree[expr]))
    for var, expr in zip(list(G), exprs):
        globals()[var] = eval(expr)
    print("Success!")
    return exprs

for i in range(5):
    G0 = generate_tree(5)
    print(G0)
    G = {"Y": G0}
    run_tree(G)

trees = []
for i in range(5):
    trees.append(generate_tree(5))

print(trees)

#Questions
# How are variables split up and defined?
#   1) Generate a finite set of variables first then fill each out with a tree, selecting from the var list.
#       Introduce new variables through mutation. Perhaps mutate in tree then define as a new tree. 
#       Don't generate new variables for leaf nodes as currently done, just draw from the pool of initially generated leaves.
#       This makes it likely that every leaf is used, most used recurrently.
#       Order of variables may be important too.
#       What about duplicating variables in order then?
#       e.g. [A, B, A, C, B, A, C] each being reused.
#       Could independently evolve a linear chromosome for that purpose.
#       So: Define initial variables (indexing by iterator) (M_0, M_1, V_0, V_1, V_2, S_0, S_1) and initialize them.
#       Define random order that they are computed w.r.t each other.
#       [M_1(V_2, S_1), V_0(V_0, S_0, S_1), M_1(M_1, V_0, V_2), V_0(V_1, S_0, S_1), etc.] with whatever repeats. Could even generate it this way with args that control local leaf selection.
#       Then generate the tree for each: M1 -> multiply -> V2, S1.
#       Benefit of this approach is that I don't need S_END and I can use sampling without replacement to avoid problems.
# How are nodes selected for operators?

