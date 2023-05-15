import datetime
import random
import numpy as np
import pandas as pd
from nutrition_opt_MPC import nutrition_opt

#配置参数
'''读取机器状态参数'''
'''读取订单生产配置参数'''
save_result_dir = './result/machine_order_params_file/'
file = save_result_dir + 'machine_order_params.xls'
c_qn_n_data = pd.read_excel(file,sheet_name='c_qn_n',header=None,skiprows=0)
c_v_mt_n_data = pd.read_excel(file,sheet_name='c_v_mt_n',header=None,skiprows=0)
c_dn_n_data = pd.read_excel(file,sheet_name='c_dn_n',header=None,skiprows=0)
c_wn_n_data = pd.read_excel(file,sheet_name='c_wn_n',header=None,skiprows=0)
c_type_n_data = pd.read_excel(file,sheet_name='c_type_n',header=None,skiprows=0)
c_e_mc_n_data = pd.read_excel(file,sheet_name='c_e_mc_n',header=None,skiprows=0)
c_e_mp_n_data = pd.read_excel(file,sheet_name='c_e_mp_n',header=None,skiprows=0)
clt_time_matrix_data = pd.read_excel(file,sheet_name='clt_time_matrix',header=None,skiprows=0)
c_clt_max_data = pd.read_excel(file,sheet_name='c_clt_max_n',header=None,skiprows=0)
c_clt_m_data = pd.read_excel(file,sheet_name='c_clt_m_list',header=None,skiprows=0)
c_E_data = pd.read_excel(file,sheet_name='E',header=None,skiprows=0)
c_machine_nums_data = pd.read_excel(file,sheet_name='machine_nums',header=None,skiprows=0)
c_order_nums_data = pd.read_excel(file,sheet_name='order_nums',header=None,skiprows=0)
c_II_data = pd.read_excel(file,sheet_name='II',header=None,skiprows=0)
c_day_num_data = pd.read_excel(file,sheet_name='day_num',header=None,skiprows=0)
start = datetime.datetime.now()

N = (c_order_nums_data.values).reshape(-1).tolist()[0]   # num of orders, dummy order has order index 0
M = (c_machine_nums_data.values).reshape(-1).tolist()[0]   # num of machines
T = len((c_type_n_data.values).reshape(-1).tolist())  # num of product types, dummy order has type index 0
D = (c_day_num_data.values).reshape(-1).tolist()[0]   # days of work
b = 16  # working time everyday

scenario_num = 3 # numbers of  scenario
prob_s =[0.8, 0.15, 0.05] # probability of scenario
start_A = int(3 * D / 2)
start_B = start_A - D

n_orders = list(range(1, (N + 1)))  # n order set
n_0_orders = list(range(0, (N + 1)))  # add dummy order 0 to order set
m_machines = list(range(0, M))  # m machines set, note that machine list start with machine 0
t_types = list(range(1, (T + 1)))  # t types
t_0_types = list(range(0, (T + 1)))  # add dummy type 0 to type set
p_horizon = list(range(1, D+1))


'''读取机器状态参数'''
file = './result/machine_runing_state_file/machine_runing_state_days.xls'
machine_runing_state_data = pd.read_excel(file,header=None,skiprows=0)
c_Wd_data=[]
for d in range(0,len(machine_runing_state_data[0])):
    m_data = []
    for m in range(M):
        if machine_runing_state_data.values[d][m] == True:
            m_data += [1]
        else:
            m_data += [0]
    c_Wd_data.append(m_data)
# machine can work or not
Wd = c_Wd_data #每天工厂是否允许工作


# s_scenario_num = list(range(0, scenario_num))
# a = np.floor(M / 2)
# x = [M, a, 0]
# available_machine = []
# for d in p_horizon:
#     available_machine.append(machine_runing_state_data[d].tolist().count(True))
# available_machine = np.array(available_machine)

machine_speed_tmp = c_v_mt_n_data.values.tolist()
machine_speed_tmp = list(map(list, zip(*machine_speed_tmp)))
machine_speed = []
for i in range(M):
    machine_speed.append([0.01] + machine_speed_tmp[i])
# randomly generate setup time
setup_time_tmp = (clt_time_matrix_data.values).reshape(M, N, N).tolist()  
setup_time = [[[random.randint(100,120)/1000000 for t1 in t_0_types] for t2 in t_0_types] for i in m_machines]
for i in m_machines:
    for t1 in t_0_types:
        for t2 in t_0_types:
            if t1 == t2:
                setup_time[i][t1][t2] = 0  # setting diagonal to zero
for i in range(M):
    for t1 in range(1, N+1):
        for t2 in range(1, N+1):
            setup_time[i][t1][t2] = setup_time_tmp[i][t1-1][t2-1]
# print(setup_time)
# randomly generate workload

workload = [0.] + (c_qn_n_data.values).reshape(-1).tolist()

# randomly generated due date
due_time = [0.] + (c_dn_n_data.values).reshape(-1).tolist() #订单完成截止时间

# energy consumed daily
E_limit = (c_E_data.values).reshape(-1).tolist()[0]        #每天最大能源供应

# randomly generate daily clean time in hrs
clean_time = (c_clt_max_data.values).reshape(-1).tolist()
# randomly generate weight for each order
weight = [0.] + (c_wn_n_data.values).reshape(-1).tolist()

# energy related constant
ec_m = (c_e_mc_n_data.values).reshape(-1).tolist() #机器每小时清洗能量消耗+
ep_m = (c_e_mp_n_data.values).reshape(-1).tolist() #机器每小时加工能量消耗

# randomly generate order type assignment,这里每个II的取值要么为0要么为1，
# 也就是订单n包含产品t的话，对应的II的值为1，否则为0。这里应该把II看成一个是II_n*t的矩阵，
# 那么II_0t全为0，那么如果订单1包含了第t种产品，然后接下来II矩阵的第一行第t列的值为1。以此类推
II = (c_II_data.values).reshape(len(n_0_orders), len(t_0_types)).tolist()
print("II ", II)

print("\n------------------------------------------")
print("\tOPTIMIZATION BEGINS")
print("------------------------------------------")

schedule = nutrition_opt(Wd, M, b, II, machine_speed, workload, setup_time, clean_time, weight, due_time, ec_m, ep_m, E_limit, p_horizon, start_A, start_B)


print("\n------------------------------------------")
print("\tOPTIMIZATION FINISHES")
print("------------------------------------------")
end = datetime.datetime.now()
print('耗时{}'.format(end - start))

