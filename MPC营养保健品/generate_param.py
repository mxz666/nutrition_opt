# -*- coding: UTF-8 -*-
import numpy as np
import xlwt  # 负责写excel
import os
import shutil
import random
import itertools
import operator
from functools import reduce
import pandas as pd
import time
import copy

def filter_longtime_list(list_before, clt_time_matrix, machine_idx):
    list_provious = copy.deepcopy(list_before)
    list_filter = []
    for i in range(len(list_provious)-1):
        if len(list_provious[i]) < 2:
            continue
        else:
            for j in range(i+1,len(list_provious)):
                if j >= len(list_provious):
                    break
                else:
                    if len(list_provious[i]) != len(list_provious[j]):
                        continue
                    else:
                        set_c = set(list_provious[i]) & set(list_provious[j])
                        list_c = list(set_c)
                        if len(list_c) == len(list_provious[i]):
                            clc_time_sum_i = 0
                            clc_time_sum_j = 0
                            for k1 in range(len(list_provious[i])-1):
                                idx1 = list_provious[i][k1]-1
                                idx2 = list_provious[i][k1+1]-1
                                clc_time_sum_i += clt_time_matrix[machine_idx][idx1][idx2]
                                idx1 = list_provious[j][k1]-1
                                idx2 = list_provious[j][k1+1]-1
                                clc_time_sum_j += clt_time_matrix[machine_idx][idx1][idx2]
                            if clc_time_sum_i <= clc_time_sum_j:
                                list_filter.append(list_provious[j])
                            else:
                                list_filter.append(list_provious[i])
                                break
    # print("list need filtered len ", len(list_filter))
    for list_i in list_filter:
        if list_i in list_provious:
            list_provious.remove(list_i)
    print("1list after filter len ", (list_provious), " len ", len(list_provious))
    return list_provious
def filter_order_biger_than_n(list_before):
    list_provious = copy.deepcopy(list_before)
    list_filter = []
    for sub_list in list_provious:
        if len(sub_list) > 2:
            list_filter.append(sub_list)
    print("list_filter ", list_filter)
    for list_i in list_filter:
        if list_i in list_provious:
            list_provious.remove(list_i)
    print("2list after filter len ", (list_provious), " len ", len(list_provious))
    return list_provious


'''一、产生机器开工排班参数'''
start_time = time.time()
threshold = 1.0#开机率0.6,0.8,1.0
order_nums = 2 #订单数量
machine_nums = 3
day_num = 20  #尽可能大于机器加工的最大运行天数
run_mechines_days = []
for dn in range(day_num):
    run_mechines = []
    rnd = np.random.uniform()
    if rnd > threshold:
        run_mechines=[False]*machine_nums
    else:
        run_mechines=[True]*machine_nums 
    # if run_mechines==[False]*machine_nums:
    #     run_mechines=[True]*(machine_nums-1)+[False]  #避免全部为False,需要根据mechine_num具体修改
    #     print("all state is False ", run_mechines)
    run_mechines_days.append(run_mechines)
#print("run_mechines ", run_mechines)
save_result_dir = './result/machine_runing_state_file/'
if os.path.exists(save_result_dir):
    shutil.rmtree(save_result_dir)
x_excel_output = run_mechines_days
filename =xlwt.Workbook() #创建工作簿
sheet1 = filename.add_sheet('输入数据',cell_overwrite_ok=True) #创建sheet
[h,l]=[day_num, machine_nums] #h为行数，l为列数
print("h", h, " l ", l)
for i in range (h):
    for j in range (l):
        sheet1.write(i,j,str(x_excel_output[i][j]))
if not os.path.exists(save_result_dir):
    os.makedirs(save_result_dir)
filename.save(save_result_dir + 'machine_runing_state_days' + '.xls')

'''二、产生机器和订单相关的起始参数'''
p_hours = 16     #每天可用加工时长,单位小时
E = [(165.- 9.*order_nums)*machine_nums]        #每天最大能源供应，暂时考虑充足供应
#1. order params
qn_avg = 400. #均值， 可修改
qn_var = 100. #方差，可修改
c_qn_n = []
for i in range(order_nums):
    rand_val = float(int(random.gauss(qn_avg,qn_var)/10)*10) #gauss(a,b),a表示均值，b表示方差
    if rand_val <= 0:
            rand_val = qn_avg
    c_qn_n.append(rand_val)
print("c_qn_n is ", c_qn_n)
save_result_dir = './result/machine_order_params_file/'
if os.path.exists(save_result_dir):
    shutil.rmtree(save_result_dir)
if not os.path.exists(save_result_dir):
    os.makedirs(save_result_dir)
x_excel_output = c_qn_n
filename =xlwt.Workbook() #创建工作簿
sheet1 = filename.add_sheet('c_qn_n',cell_overwrite_ok=True) #创建sheet
l=order_nums #l为列数
print("l ", l)
for i in range (l):
    sheet1.write(0,i,str(x_excel_output[i]))

v_mt_avg = 5.  #均值， 可修改
v_mt_var = 1 #方差，可修改
c_v_mt_list = []
for i in range(order_nums):
    clt_rnd = []
    for j in range(machine_nums):
        rand_val = float(int(random.gauss(v_mt_avg,v_mt_var)*100)/100) #gauss(a,b),a表示均值，b表示方差
        if rand_val <= 0:
            rand_val = v_mt_avg
        clt_rnd.append(rand_val)
    c_v_mt_list.append(clt_rnd)
print("c_v_mt_list is ", c_v_mt_list)
x_excel_output = c_v_mt_list
sheet2 = filename.add_sheet('c_v_mt_n',cell_overwrite_ok=True) #创建sheet
[h,l]=[order_nums, machine_nums] #h为行数，l为列数
print("h", h, " l ", l)
for i in range (h):
    for j in range (l):
        sheet2.write(i,j,str(x_excel_output[i][j]))

dn_avg = 800. * order_nums / machine_nums / (17 - order_nums) #均值， 可修改，表示天数
dn_var = 80 * order_nums / machine_nums / (17 - order_nums) #方差，可修改
c_dn_n = []
for i in range(order_nums):
    rand_val = float(int(random.gauss(dn_avg,dn_var))) * p_hours #gauss(a,b),a表示均值，b表示方差
    if rand_val <= 0:
        rand_val = dn_avg
    c_dn_n.append(rand_val) #订单完成截止时间
print("c_dn_n is ", c_dn_n)
x_excel_output = c_dn_n
sheet3 = filename.add_sheet('c_dn_n',cell_overwrite_ok=True) #创建sheet
l=order_nums #l为列数
print("l ", l)
for i in range (l):
    sheet3.write(0,i,str(x_excel_output[i]))

wn_avg = 0.5 #均值， 可修改
wn_var = 0.1 #方差，可修改
c_wn_n = []
for i in range(order_nums):
    rand_val = float(int(random.gauss(wn_avg,wn_var)*100)/100) #gauss(a,b),a表示均值，b表示方差
    if rand_val <= 0:
        rand_val = wn_avg
    c_wn_n.append(rand_val)
print("c_wn_n is ", c_wn_n)
x_excel_output = c_wn_n
sheet4 = filename.add_sheet('c_wn_n',cell_overwrite_ok=True) #创建sheet
l=order_nums #l为列数
print("l ", l)
for i in range (l):
    sheet4.write(0,i,str(x_excel_output[i]))

c_type_n = list(np.linspace(1,order_nums,order_nums,dtype=int))#在本程序里面订单数=type数，这个主要是因为MPDQN的程序里面setup对应的是订单，而规划模型里面setup对应的是type，为了统一写代码方便，所以这样定。可以理解成N个订单对应的type捡出来，其他的type就忽略了
print("c_type_n is ", c_type_n)
x_excel_output = c_type_n
sheet5 = filename.add_sheet('c_type_n',cell_overwrite_ok=True) #创建sheet
l=order_nums #l为列数
print("l ", l)
for i in range (l):
    sheet5.write(0,i,str(x_excel_output[i]))

#machine params
e_mc_avg = 1.0 #均值， 可修改
e_mc_var = 0.1 #方差，可修改
c_e_mc_n = []
for i in range(machine_nums):
    rand_val = float(int(random.gauss(e_mc_avg,e_mc_var)*10)/10) #gauss(a,b),a表示均值，b表示方差
    if rand_val <= 0:
        rand_val = e_mc_avg
    c_e_mc_n.append(rand_val) #机器每小时清洗能量消耗
print("c_e_mc_n is ", c_e_mc_n)
x_excel_output = c_e_mc_n
sheet6 = filename.add_sheet('c_e_mc_n',cell_overwrite_ok=True) #创建sheet
l=machine_nums #l为列数
print("l ", l)
for i in range (l):
    sheet6.write(0,i,str(x_excel_output[i]))

e_mp_avg = 10. #均值， 可修改
e_mp_var = 1. #方差，可修改
c_e_mp_n = []
for i in range(machine_nums):
    rand_val = float(int(random.gauss(e_mp_avg,e_mp_var)*10)/10) #gauss(a,b),a表示均值，b表示方差
    if rand_val <= 0:
        rand_val = e_mp_avg
    c_e_mp_n.append(rand_val) #机器每小时加工能量消耗
print("c_e_mp_n is ", c_e_mp_n)
x_excel_output = c_e_mp_n
sheet7 = filename.add_sheet('c_e_mp_n',cell_overwrite_ok=True) #创建sheet
l=machine_nums #l为列数
print("l ", l)
for i in range (l):
    sheet7.write(0,i,str(x_excel_output[i]))

clt_time_avg = 1.  #均值， 可修改
clt_time_var = 0.1 #方差，可修改
clt_time_matrix = []
for i in range(machine_nums):
    clt_rnd_jk = []
    for j in range(order_nums):
        clt_rnd_k = []
        for k in range(order_nums):
            rand_val = float(int(random.gauss(clt_time_avg,clt_time_var)*100)/100) #gauss(a,b),a表示均值，b表示方差
            if rand_val <= 0:          #不等于零的切换时间要大于DT
                rand_val = clt_time_avg
            if j==k:
                rand_val = 0.
            clt_rnd_k.append(rand_val)
        clt_rnd_jk.append(clt_rnd_k)
    clt_time_matrix.append(clt_rnd_jk)  
print("clt_time_matrix is ", clt_time_matrix)
x_excel_output = clt_time_matrix
sheet8 = filename.add_sheet('clt_time_matrix',cell_overwrite_ok=True) #创建sheet
[h,l1,l2]=[machine_nums, order_nums, order_nums] #h为行数，l1,12为列数
print("h", h, " l1 ", l1, " l2 ", l2)
for i in range (h):
    for j in range (l1):
        for k in range (l2):
            sheet8.write(i,j*order_nums+k,str(x_excel_output[i][j][k]))

clt_mn_avg = 5  #均值， 可修改
clt_mn_var = 0.5 #方差，可修改
c_clt_m_list = []
clt_rnd = []
for i in range(machine_nums):
    rand_val = float(int(random.gauss(clt_mn_avg,clt_mn_var)*100)/100)
    clt_rnd.append(rand_val)
c_clt_m_list = [clt_rnd]*day_num
c_clt_max_n = clt_rnd
# print("len clt ", len(c_clt_m_list), " c_clt_m_list ", np.array(c_clt_m_list))
x_excel_output = c_clt_max_n
sheet9 = filename.add_sheet('c_clt_max_n',cell_overwrite_ok=True) #创建sheet
l=machine_nums #l为列数
print("l ", l)
for i in range (l):
    sheet9.write(0,i,str(x_excel_output[i]))

x_excel_output = c_clt_m_list
sheet10 = filename.add_sheet('c_clt_m_list',cell_overwrite_ok=True) #创建sheet
[h,l]=[day_num, machine_nums] #h为行数，l1,12为列数
print("h", h, " l ", l)
for i in range (h):
    for j in range (l):
        sheet10.write(i,j,str(x_excel_output[i][j]))

x_excel_output = E
sheet11 = filename.add_sheet('E',cell_overwrite_ok=True) #创建sheet
sheet11.write(0,0,str(x_excel_output[0]))

x_excel_output = machine_nums
sheet12 = filename.add_sheet('machine_nums',cell_overwrite_ok=True) #创建sheet
sheet12.write(0,0,str(x_excel_output))

x_excel_output = order_nums
sheet13 = filename.add_sheet('order_nums',cell_overwrite_ok=True) #创建sheet
sheet13.write(0,0,str(x_excel_output))


# randomly generate order type assignment,这里每个II的取值要么为0要么为1，
# 也就是订单n包含产品t的话，对应的II的值为1，否则为0。这里应该把II看成一个是II_n*t的矩阵，
# 那么II_0t全为0，那么如果订单1包含了第t种产品，然后接下来II矩阵的第一行第t列的值为1。以此类推
N = order_nums   # num of orders, dummy order has order index 0
T = len(c_type_n)  # num of product types, dummy order has type index 0
t_0_types = list(range(0, (T + 1)))  # add dummy type 0 to type set
n_0_orders = list(range(0, (N + 1)))  # add dummy order 0 to order set
II = [[0 for t in t_0_types] for n in n_0_orders] #产生MPC算法需要的订单包含的产品类别矩阵
for n in n_0_orders:
    for t in t_0_types:
        if n == t:
            II[n][t] = 1
            break
print("II ", II)
x_excel_output = II
sheet14 = filename.add_sheet('II',cell_overwrite_ok=True) #创建sheet
[h,l]=[len(n_0_orders), len(t_0_types)] #h为行数，l1,12为列数
print("h", h, " l ", l)
for i in range (h):
    for j in range (l):
        sheet14.write(i,j,str(x_excel_output[i][j]))

'''生成所有的action，并保存为文件'''
list1 = [0]+list(np.arange(1, order_nums+1))
list2 = []
list3 = []
for i in range(1, len(list1)+1):
    iter1 = itertools.permutations(list1, i)
    list2.append(list(iter1))
list3 = reduce(operator.add, list2)
list3_reset = []
list3_reset.append([0])
for l1_num in range(len(list3)): #这个for循环主要是变了数据的格式
    if 0 in list3[l1_num]:
        continue
    else:
        list3_reset.append(list(list3[l1_num]))
print("list3_reset len ", len(list3_reset))
machine_order_list = []
tmp_machine_order = []
len_laf = 0
for k1 in range(1, machine_nums+1):  # 比如一共有2个可能的action 1，2，有3台机器，则第一台机器[1 1 1 1 2 2 2 2]，第2台机器[1 1 2 2]*2，第3台机器[1 2]*4，把这三个list纵向放在一起，每一列单独拿出来作为一个完整的方案，所有的就是全部的可能方案
    list_after_filter1 = filter_longtime_list(list3_reset, clt_time_matrix, k1-1)
    list_after_filter2 = filter_order_biger_than_n(list_after_filter1)
    len_laf = len(list_after_filter2)
    tmp_machine_order = [val for val in list_after_filter2 for m_num in range(len_laf**(machine_nums-k1))]*(len_laf**(k1-1))
    machine_order_list.append(tmp_machine_order)
print("machine_order_list ", len(machine_order_list[0]))
start_time1 = time.time()
action_list = []
for a_num in range(len_laf**machine_nums):
    action_order_list = []
    list_1Dim = []
    for o_num in range(machine_nums):
        list_1Dim += machine_order_list[o_num][a_num]
        action_order_list.append(machine_order_list[o_num][a_num])
    duplicate_result = pd.value_counts(list_1Dim)
    if 0 in duplicate_result.index:
        duplicate_result = duplicate_result.drop([0]) #删除index为0的行
    if (duplicate_result.values.sum() > len(duplicate_result)):
        continue
    if action_order_list == [[0]]*machine_nums:   #删除全为0的action
        print("zero action_order_list ", [[0]]*machine_nums)
        continue
    action_list.append(action_order_list)
print("action list len ", len(action_list))
end_time1 = time.time()
print("Took %.2f seconds" % (end_time1 - start_time1))
action_list_write=np.array(action_list,dtype=object)
np.save(save_result_dir + 'action_list.npy',action_list_write)   # 保存为.npy格式

x_excel_output = list1
sheet15 = filename.add_sheet('list1',cell_overwrite_ok=True) #创建sheet
l=order_nums+1 #l为列数
print("l ", l)
for i in range (l):
    sheet15.write(0,i,str(x_excel_output[i]))
    
x_excel_output = day_num
sheet16 = filename.add_sheet('day_num',cell_overwrite_ok=True) #创建sheet
sheet16.write(0,0,str(x_excel_output))

filename.save(save_result_dir + 'machine_order_params' + '.xls')
end_time = time.time()
print("Took %.2f seconds" % (end_time - start_time))