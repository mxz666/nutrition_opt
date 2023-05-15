import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
import gurobipy as gp
from gurobipy import GRB
from tabulate import tabulate
import numpy as np

sto_nsch = gp.Model('Stochastic Programming')

def nutrition_opt(Wd, M, b, II, machine_speed, workload, setup_time, clean_time, weight, due_time, ec_m, ep_m,
                  E_limit, p_horizon, start_A, start_B, **kwargs):

    n_orders = list(range(1, len(due_time)))
    print(f"n_orders={n_orders}")

    n_0_orders = list(range(0, len(due_time)))
    print(f"n_0_orders={n_0_orders}")

    m_machines = list(range(0, len(machine_speed)))
    print(f"m_machines={m_machines}")

    t_types = list(range(1, len(machine_speed[0])))
    # t_types = list(range(1, (T + 1)))
    print(f"t_types={t_types}")

    t_0_types = list(range(0, len(machine_speed[0])))
    # t_0_types = list(range(0, (T + 1)))
    print(f"t_0_types={t_0_types}")

    # seq = sto_nsch.addVars(m_machines, n_0_orders, n_orders, p_horizon,
    #                        lb=0, ub=1, vtype=GRB.BINARY, name="seq")
    # o_md = sto_nsch.addVars(m_machines, p_horizon, lb=0, ub=1,
    #                         vtype=GRB.BINARY, name="o_md")
    # a_mnd = sto_nsch.addVars(m_machines, n_orders, p_horizon, lb=0, ub=1, vtype=GRB.BINARY,
    #                          name='a_mnd')
    #
    # #################################
    # # continuous variable
    # delta = sto_nsch.addVars(m_machines, n_orders, p_horizon,
    #                          lb=0, ub=float("1"), vtype=GRB.CONTINUOUS, name="delta")
    # sp_nd = sto_nsch.addVars(n_0_orders, p_horizon, lb=0, ub=float("1"), vtype=GRB.CONTINUOUS, name="sp_nd")
    # p_mnd = sto_nsch.addVars(m_machines, n_orders, p_horizon,
    #                          lb=0, ub=float("inf"), vtype=GRB.CONTINUOUS, name="p_mnd")
    # c_n = sto_nsch.addVars(n_orders, lb=0,
    #                        ub=float("inf"), vtype=GRB.CONTINUOUS, name="c_n")
    # c_mn = sto_nsch.addVars(m_machines, n_orders,
    #                         lb=0, ub=float("inf"), vtype=GRB.CONTINUOUS, name="c_mn")
    # c_mnd = sto_nsch.addVars(m_machines, n_0_orders, p_horizon,
    #                          lb=0, ub=float("inf"), vtype=GRB.CONTINUOUS, name="c_mnd")
    # t_n = sto_nsch.addVars(n_orders, lb=0, ub=float("inf"), vtype=GRB.CONTINUOUS, name="t_n")
    # setup = sto_nsch.addVars(m_machines, n_0_orders, n_orders, p_horizon,
    #                          lb=0, ub=float("inf"), vtype=GRB.CONTINUOUS, name="setup")
    # ec_md = sto_nsch.addVars(m_machines, p_horizon, lb=0,
    #                          ub=float("inf"), vtype=GRB.CONTINUOUS, name="ec_md")
    # ep_md = sto_nsch.addVars(m_machines, p_horizon, lb=0,
    #                          ub=float("inf"), vtype=GRB.CONTINUOUS, name="ep_md")

    # large enough constant
    B = 10000000
    Wd.insert(0, [1]*len(m_machines)) # 因为模型里面的d实际是从1开始的，这就使得原始的Wd少了一个元素，所以添加了d=0是全开机这个选择（不影响，因为开始已经使得所有的订单完工时间是0了）
    print("Wd", Wd)
    Wd_new = [] #这里主要是为了MPC中预测的方便，所以改成了一维数据
    for d in p_horizon:
        if Wd[d] == [1]*len(m_machines):
            Wd_new.append(1)
        else: 
            Wd_new.append(0)
   
    Wd_new_d = pd.Series(Wd_new)
    print("Wd_new",Wd_new)
   

    #available_machine_d = pd.Series(available_machine)

    # generate output file
    file_name = 'sto_nutrition_opt_engine_log' + '.txt'
    file_object = open(file_name, 'a')
    file_object.truncate(0)
    # generate output file

    passed_days = [0]
    counter = 0
    com_pred = []
    temp=[]
    delta = sto_nsch.addVars(m_machines, n_orders, p_horizon,
                             lb=0, ub=float("1"), vtype=GRB.CONTINUOUS, name="delta")
    for n in n_orders:
        sto_nsch.addConstr(gp.quicksum(delta[m, n, d] for m in m_machines for d in p_horizon) == 1, 'whole_workload')

    for d in p_horizon:
        seq = sto_nsch.addVars(m_machines, n_0_orders, n_orders, p_horizon,
                               lb=0, ub=1, vtype=GRB.BINARY, name="seq")
        o_md = sto_nsch.addVars(m_machines, p_horizon, lb=0, ub=1,
                                vtype=GRB.BINARY, name="o_md")
        a_mnd = sto_nsch.addVars(m_machines, n_orders, p_horizon, lb=0, ub=1, vtype=GRB.BINARY,
                                 name='a_mnd')

        #################################
        # continuous variable
        sp_nd = sto_nsch.addVars(n_0_orders, p_horizon, lb=0, ub=float("1"), vtype=GRB.CONTINUOUS, name="sp_nd")
        p_mnd = sto_nsch.addVars(m_machines, n_orders, p_horizon,
                                 lb=0, ub=float("inf"), vtype=GRB.CONTINUOUS, name="p_mnd")
        c_n = sto_nsch.addVars(n_orders, lb=0,
                               ub=float("inf"), vtype=GRB.CONTINUOUS, name="c_n")
        c_mn = sto_nsch.addVars(m_machines, n_orders,
                                lb=0, ub=float("inf"), vtype=GRB.CONTINUOUS, name="c_mn")
        c_mnd = sto_nsch.addVars(m_machines, n_0_orders, p_horizon,
                                 lb=0, ub=float("inf"), vtype=GRB.CONTINUOUS, name="c_mnd")
        t_n = sto_nsch.addVars(n_orders, lb=0, ub=float("inf"), vtype=GRB.CONTINUOUS, name="t_n")
        setup = sto_nsch.addVars(m_machines, n_0_orders, n_orders, p_horizon,
                                 lb=0, ub=float("inf"), vtype=GRB.CONTINUOUS, name="setup")
        ec_md = sto_nsch.addVars(m_machines, p_horizon, lb=0,
                                 ub=float("inf"), vtype=GRB.CONTINUOUS, name="ec_md")
        ep_md = sto_nsch.addVars(m_machines, p_horizon, lb=0,
                                 ub=float("inf"), vtype=GRB.CONTINUOUS, name="ep_md")
            # binary variable
        # seq = sto_nsch.addVars(m_machines, n_0_orders, n_orders, list(range(0, len(p_horizon)-d+1)),
        #                     lb=0, ub=1, vtype=GRB.BINARY, name="seq")
        # o_md = sto_nsch.addVars(m_machines,list(range(0, len(p_horizon)-d+1)), lb=0, ub=1,
        #                         vtype=GRB.BINARY, name="o_md")
        # a_mnd = sto_nsch.addVars(m_machines, n_orders, list(range(0, len(p_horizon)-d+1)), lb=0, ub=1, vtype=GRB.BINARY, name='a_mnd')
        #
        # #################################
        # # continuous variable
        # delta = sto_nsch.addVars(m_machines, n_orders, list(range(0, len(p_horizon)-d+1)),
        #                         lb=0, ub=float("1"), vtype=GRB.CONTINUOUS, name="delta")
        # sp_nd = sto_nsch.addVars(n_0_orders, p_horizon, lb=0, ub=float("1"), vtype=GRB.CONTINUOUS, name="sp_nd")
        # p_mnd = sto_nsch.addVars(m_machines, n_orders, list(range(0, len(p_horizon)-d+1)),
        #                         lb=0, ub=float("inf"), vtype=GRB.CONTINUOUS, name="p_mnd")
        # c_n = sto_nsch.addVars(n_orders, lb=0,
        #                     ub=float("inf"), vtype=GRB.CONTINUOUS, name="c_n")
        # c_mn = sto_nsch.addVars(m_machines, n_orders,
        #                         lb=0, ub=float("inf"), vtype=GRB.CONTINUOUS, name="c_mn")
        # c_mnd = sto_nsch.addVars(m_machines, n_0_orders, list(range(0, len(p_horizon)-d+1)),
        #                         lb=0, ub=float("inf"), vtype=GRB.CONTINUOUS, name="c_mnd")
        # t_n = sto_nsch.addVars(n_orders, lb=0, ub=float("inf"), vtype=GRB.CONTINUOUS, name="t_n")
        # setup = sto_nsch.addVars(m_machines, n_0_orders, n_orders, list(range(0, len(p_horizon)-d+1)),
        #                         lb=0, ub=float("inf"), vtype=GRB.CONTINUOUS, name="setup")
        # ec_md = sto_nsch.addVars(m_machines, list(range(0, len(p_horizon)-d+1)), lb=0,
        #                         ub=float("inf"), vtype=GRB.CONTINUOUS, name="ec_md")
        # ep_md = sto_nsch.addVars(m_machines, list(range(0, len(p_horizon)-d+1)), lb=0,
        #                         ub=float("inf"), vtype=GRB.CONTINUOUS, name="ep_md")
        counter += 1
        temp = [Wd_new[d]]
        missing_keys = [d for d in p_horizon if d not in temp]
        print(f"Missing keys: {missing_keys}")
        passed_days.append(counter)
        passed_day_list = list(range(1, len(passed_days)))
        ets3 = ExponentialSmoothing(Wd_new_d, trend=None, seasonal=None, seasonal_periods=p_horizon)        
        r3 = ets3.fit()
        pred3 = r3.predict(start=1, end=len(p_horizon)-d)
        init_pred = np.array(np.around(pred3))
        temp = temp + list(init_pred) #60+60=120

        pick_len = len(temp)
        #new_temp = temp[pick_len - start_A:pick_len - start_B] #60

        sto_nsch.setObjective(gp.quicksum(weight[n] * t_n[n] for n in n_orders), GRB.MINIMIZE)

        # missing_keys = [(n, d) for n in n_orders for d in p_horizon if (n, d) not in sp_nd]
        # print(f"Missing keys: {missing_keys}")

        for n in n_orders:
            sto_nsch.addConstr(
                sp_nd[n, d] == 1 - gp.quicksum(delta[m, n, dd] for m in m_machines for dd in passed_day_list),
                'remain_workload')

        for n in n_orders:
            sto_nsch.addConstr(gp.quicksum(delta[m, n, dd] for m in m_machines for dd in passed_day_list) >= 0.01 * d,
                               'whole_workload')

        for m in m_machines:
            for dd in list(range(1, len(p_horizon)-d+1)):
                sto_nsch.addConstr(o_md[m, dd] <= temp[dd], 'machine_open')
            
        for m in m_machines:
            for n in n_orders:
                for dd in list(range(1, len(p_horizon)-d+1)):
                    sto_nsch.addConstr(delta[m, n, dd] <= o_md[m, dd], 'machine_on_UB')
                    sto_nsch.addConstr(delta[m, n, dd] <= a_mnd[m, n, dd], 'order_assigned')

        for dd in list(range(1, len(p_horizon)-d+1)):
            sto_nsch.addConstr(gp.quicksum((ep_md[m, dd] + ec_md[m, dd]) for m in m_machines) <= E_limit, 'energy_limit')
            sto_nsch.addConstr(gp.quicksum((ep_md[m, dd] + ec_md[m, dd]) for m in m_machines) >= E_limit / (M * 2),
                               'energy_limit2')
        
        for m in m_machines:
            for n in n_orders:
                for dd in list(range(1, len(p_horizon)-d+1)):
                    sto_nsch.addConstr(p_mnd[m, n, dd] == gp.quicksum((workload[n] * delta[m, n, dd] * II[n][t] / machine_speed[m][t]) for t in t_types), 'processing_time')
                
        for m in m_machines:
           for dd in list(range(1, len(p_horizon)-d+1)):
                sto_nsch.addConstr(ep_md[m, dd] == gp.quicksum((p_mnd[m, n, dd] * ep_m[m]) for n in n_orders),'energy_processing_per_day')

        for m in m_machines:
            for n in n_orders:
                for dd in list(range(1, len(p_horizon)-d+1)):
                    sto_nsch.addConstr(seq[m, n, n, dd] == 0, 'self_adjacency')

        for m in m_machines:
            for n2 in n_orders:
                for dd in list(range(1, len(p_horizon)-d+1)):
                    sto_nsch.addConstr(gp.quicksum(seq[m, n1, n2, dd]
                                                   for n1 in n_0_orders) <= B * delta[m, n2, dd], 'adjacency_assignment_01')
                    sto_nsch.addConstr(gp.quicksum(seq[m, n1, n2, dd]
                                                   for n1 in n_0_orders) >= delta[m, n2, dd], 'adjacency_assignment_02')

        for m in m_machines:
            for n1 in n_0_orders:
                for dd in list(range(1, len(p_horizon)-d+1)):
                    sto_nsch.addConstr(gp.quicksum(seq[m, n1, n2, dd] for n2 in n_orders) <= 1, 'one_successor')

        for m in m_machines:
            for n in n_orders:
                for dd in list(range(1, len(p_horizon)-d+1)):
                    sto_nsch.addConstr(setup[m, 0, n, dd] == 0, 'dummy_order_setup_time')

        for m in m_machines:
            for dd in list(range(1, len(p_horizon)-d+1)):
                sto_nsch.addConstr(gp.quicksum(p_mnd[m, n, dd] for n in n_orders) +
                               gp.quicksum(setup[m, n1, n2, dd] for n1 in n_0_orders for n2 in n_orders) <= b,
                               'total_working_time')

        for n in n_orders:
            for dd in list(range(1, len(p_horizon)-d+1)):
                sto_nsch.addConstr(gp.quicksum(a_mnd[m, n, dd] for m in m_machines) <= 1, "one_machine1")

        for m in m_machines:
            for dd in list(range(1, len(p_horizon)-d+1)):
                sto_nsch.addConstr(ec_md[m, dd] == (gp.quicksum(setup[m, n1, n2, dd] for n1 in n_0_orders for n2 in n_orders)
                                               + clean_time[m] * o_md[m, dd]) * ec_m[m], 'energy_clean_time')

        for m in m_machines:
            for n1 in n_orders:
                for n2 in n_orders:
                    for dd in list(range(1, len(p_horizon)-d+1)):
                        sto_nsch.addConstr(setup[m, n1, n2, dd] == gp.quicksum((seq[m, n1, n2, dd] * II[n1][t1] * II[n2][t2] *
                                                                           setup_time[m][t1][t2]) for t1 in t_types
                                                                          for t2 in t_types), 'setup')

        for m in m_machines:
            for dd in list(range(1, len(p_horizon)-d+1)):
                sto_nsch.addConstr(c_mnd[m, 0, dd] == 16 * (d - 1), 'completion_time_dummy_order')
                for n in n_orders:
                    sto_nsch.addConstr(c_mnd[m, n, dd] >= 16 * (d - 1) * a_mnd[m, n, dd], 'completion_time_daily_UB')
                    for n2 in n_orders:
                        for n1 in n_0_orders:
                            sto_nsch.addConstr(c_mnd[m, n2, dd] + B * (1 - seq[m, n1, n2, dd]) >=
                                            c_mnd[m, n1, dd] + setup[m, n1, n2, dd] + p_mnd[m, n2, dd],
                                            'residence_time_adjacency_LB')
                        
        for m in m_machines:
            for n in n_orders:
                sto_nsch.addConstr(c_mn[m, n] == gp.max_(c_mnd[m, n, dd] for dd in list(range(1, len(p_horizon)-d+1))), 'c_mn')

        for n in n_orders:
            sto_nsch.addConstr(c_n[n] == gp.max_(c_mn[m, n] for m in m_machines), 'c_n')

        for m in m_machines:
            for dd in list(range(1, len(p_horizon)-d+1)):
                for n1 in n_orders:
                    for n2 in n_orders:
                        if n1 == n2:
                            continue
                        else:
                            sto_nsch.addConstr((gp.quicksum(seq[m, n0, n1, dd] for n0 in n_0_orders
                                                            if (n0 != n1) and (n0 != n2))) >= seq[m, n1, n2, dd],
                                            'at_least_one_predecessor')

        for n in n_orders:
            sto_nsch.addConstr(t_n[n] >= c_n[n] - due_time[n], 'tardiness1')
            sto_nsch.addConstr(t_n[n] >= 0, 'tardiness2')

        sto_nsch.setParam('Timelimit', 120)
        sto_nsch.resetParams()
        # sto_nsch.feasRelaxS(0,False,False,True)
        sto_nsch.setParam('OutputFlag', 0)
        # # sto_nsch.write('nutrition_optimization.lp')
        # sto_nsch.setParam('LogToConsole', 1)
        # sto_nsch.setParam('LogFile', 'output.log')
        sto_nsch.optimize()
        # check optimization status
        if sto_nsch.status == GRB.Status.OPTIMAL:
            print('Optimal objective value: %g' % sto_nsch.objVal)
        elif sto_nsch.status == GRB.Status.INFEASIBLE:
            sto_nsch.computeIIS()
            sto_nsch.write('model.ilp')
            print('Model is infeasible; see model.ilp for infeasibility proof.')
        elif sto_nsch.status == GRB.Status.INF_OR_UNBD:
            print('Model is infeasible or unbounded')
        elif sto_nsch.status == GRB.Status.UNBOUNDED:
            print('Model is unbounded')
        else:
            print('Optimization ended with status %d' % sto_nsch.status)

        # nSolution = sto_nsch.SolCount
        # print('number of solution stored:' + str(nSolution))

        schedule = []
        tardy_order = []
        tardy_time = []
        for m in m_machines:
            for n2 in n_orders:
                if sum(seq[m, n1, n2, d].X for n1 in n_0_orders) >= 0.99:
                    temp_oid = n2
                    temp_machine = m
                    temp_day = d
                    temp_start = round(c_mnd[m, n2, d].X - p_mnd[m, n2, d].X, 2)
                    temp_finish = round(c_mnd[m, n2, d].X, 2)
                    temp_cmn = round(c_mn[m, n2].X, 2)
                    temp_cn = round(c_n[n2].X, 2)
                    temp_due_time = due_time[n2]
                    temp_processing = round(p_mnd[m, n2, d].X, 4)
                    temp_tardiness = max(0.0, round(t_n[n2].X, 2))
                    temp_surplus = round(sp_nd[n2, d].X, 4)
                    temp_type = []
                    temp_portion = []
                    for t in t_types:
                        if II[n2][t] == 1.0:
                            temp_type = t
                            temp_portion = round(delta[m, n2, d].X, 4)
                            break
                    schedule.append([temp_oid, temp_machine, temp_start, temp_finish, temp_cmn, temp_cn,
                                     temp_due_time, temp_processing, temp_tardiness, temp_type, temp_portion,
                                     temp_surplus, temp_day])
        schedule.sort(key=lambda row: (row[0], row[12]))

        if d in passed_day_list:
            # --------------------------------
            # output generation
            # --------------------------------
            file_object.write(f"\n------------------------------------------------------------\n")
            file_object.write(f"\tsolution overview day {d}\n")
            file_object.write(f"------------------------------------------------------------\n")
            file_object.write("Optimization Engine: GUROBI\n")
            file_object.write("Objective Function Choice: Minimize total weighted tardiness\n")
            file_object.write(f"The optimal objective value is {sto_nsch.ObjVal: 10.2f}\n")

            for n in n_orders:
                if t_n[n].X > 0.0:
                    tardy_order.append(n)

            if tardy_order:
                file_object.write("\n---------------------------------\n")
                file_object.write(f"\ntardy orders summary day {d}\n")
                file_object.write("---------------------------------\n")
                file_object.write(f"Current schedule has a total of {len(tardy_order)} late orders.\n\n")
                ####z整数是3的，浮点数是3f
                for n in tardy_order:
                    file_object.write(f"order {n} with weight {weight[n]} "
                                      f"has tardy time {t_n[n].x:10.2f}\n")
                    tardy_time.append(t_n[n].x)
                file_object.write(f"\nThe maximum tardy time {max(tardy_time): 10.2f} \n")
            else:
                file_object.write("\nThere are NO tardy orders!\n")

        if schedule:
            header_schedule = ['order_id', 'machine_id', 'start_time', 'finish_time', 'cmn', 'cn',
                               'due_time', 'processing_time', 'tardiness', 'type', 'portion', 'remain_portion', 'day']
            print(tabulate(schedule, headers=header_schedule, tablefmt="pretty"))
        print('\n-------------------------------------------------------------------------------------------')
    file_object.close()
    return schedule


