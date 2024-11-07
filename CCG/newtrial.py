from gurobipy import *
import numpy as np

""" The input parameter """
facility_num = 3
customer_num = 3
fixed_cost = [400, 414, 326]
unit_capacity_cost = [18, 25, 20]
trans_cost = [[22, 33, 24],
              [33, 23, 30],
              [20, 25, 27]]
max_capacity = 800

demand_nominal = [206, 274, 220]
demand_var = [40, 40, 40]

""" build initial master problem """
""" Create variables """
master = Model('master problem')
x_master = {}
z = {}
y = {}
eta = master.addVar(lb=0, vtype=GRB.CONTINUOUS, name='eta')

for i in range(facility_num):
    y[i] = master.addVar(vtype=GRB.BINARY, name='y_%s' % (i))
    z[i] = master.addVar(lb=0, vtype=GRB.CONTINUOUS, name='z_%s' % (i))

""" Set objective """
obj = LinExpr()
for i in range(facility_num):
    obj.addTerms(fixed_cost[i], y[i])
    obj.addTerms(unit_capacity_cost[i], z[i])
obj.addTerms(1, eta)

master.setObjective(obj, GRB.MINIMIZE)

""" Add Constraints  """
# cons 1
for i in range(facility_num):
    master.addConstr(z[i] <= max_capacity * y[i])

""" Add initial value Constraints  """
# create new variables x
iter_cnt = 0
for i in range(facility_num):
    for j in range(customer_num):
        x_master[iter_cnt, i, j] = master.addVar(lb=0
                                                 , ub=GRB.INFINITY
                                                 , vtype=GRB.CONTINUOUS
                                                 , name='x_%s_%s_%s' % (iter_cnt, i, j))
# create new constraints
expr = LinExpr()
for i in range(facility_num):
    for j in range(customer_num):
        expr.addTerms(trans_cost[i][j], x_master[iter_cnt, i, j])
master.addConstr(eta >= expr)

expr = LinExpr()
for i in range(facility_num):
    expr.addTerms(1, z[i])
master.addConstr(expr >= 772)  # 206 + 274 + 220 + 40 * 1.8

""" solve the model and output """
master.optimize()
print('Obj = {}'.format(master.ObjVal))
print('-----  location ----- ')
for key in z.keys():
    print('facility : {}, location: {}, capacity: {}'.format(key, y[key].x, z[key].x))


def print_sub_sol(model, d, g, x):
    d_sol = {}
    if(model.status != 2):
        print('The problem is infeasible or unbounded!')
        print('Status: {}'.format(model.status))
        d_sol[0] = 0
        d_sol[1] = 0
        d_sol[2] = 0
    else:
        print('Obj(sub) : {}'.format(model.ObjVal), end='\t | ')
        for key in d.keys():
            # print('demand: {}, perturbation = {}'.format(d[key].x, g[key].x))
            d_sol[key] = d[key].x
    return d_sol


""" Column-and-constraint generation """

LB = -np.inf
UB = np.inf
iter_cnt = 0
max_iter = 30
cut_pool = {}
eps = 0.001
Gap = np.inf

z_sol = {}
for key in z.keys():
    z_sol[key] = z[key].x
# print(z_sol)

""" solve the master problem and update bound """
master.optimize()

""" 
 Update the Lower bound 
"""
LB = master.ObjVal
print('LB: {}'.format(LB))



''' create the subproblem '''
subProblem = Model('sub problem')
''' create the subproblem '''""" set objective """
sub_obj = LinExpr()
for i in range(facility_num):
    for j in range(customer_num):
        sub_obj.addTerms(trans_cost[i][j], x[i, j])
subProblem.setObjective(sub_obj, GRB.MAXIMIZE)

""" add constraints to subproblem """
# cons 1
for i in range(facility_num):
    expr = LinExpr()
    for j in range(customer_num):
        expr.addTerms(1, x[i, j])
    subProblem.addConstr(expr <= z_sol[i], name='sub_capacity_1_z_%s' % i)

# cons 2
for j in range(facility_num):
    expr = LinExpr()
    for i in range(customer_num):
        expr.addTerms(1, x[i, j])
    subProblem.addConstr(expr >= d[j])

# cons 3
for i in range(facility_num):
    for j in range(customer_num):
        subProblem.addConstr(pi[i] - theta[j] <= trans_cost[i][j])

""" demand constraints """
for j in range(customer_num):
    subProblem.addConstr(d[j] == demand_nominal[j] + g[j] * demand_var[j])

subProblem.addConstr(g[0] + g[1] + g[2] <= 1.8)
subProblem.addConstr(g[0] + g[1] <= 1.2)