
# coding: utf-8

# In[61]:


import pandas as pd
import numpy as np

# set distance
age_diff = 10

# use pandas to read CSV as it is able to automatically convert str -> float
df = pd.read_csv("after_combat.txt")
df = df.sample(frac=1).reset_index(drop=True) # shuffle, return 100% data, and keep shuffled index in origianl order

df = df.dropna()  # drop null
data = df.as_matrix() # or .values or np.array(df)
header = df.columns.values


# In[62]:


age_col = data[:, 1].astype(np.int)
age_col.shape


# In[63]:


# a grouping helper that can divide ranges based on distance, min&max values
# params: min, max, difference
# ex: if min = 16, max = 101, distance = 10
#     then range = [(10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 70), (70, 80), (80, 90), (90, 100), (100, 110)]
def group_range(min_v, max_v, diff):
    def divider(age, offset=5):
        return age - (age % 10) + offset
    r1 = range(divider(min_v), divider(max_v, 0), diff)  # 20 since the second element of range() is excluded. And also need to +10 to cover the max_v. ex: needs 20 to cover 12. So 10+10=20
    r2 = range(divider(min_v, 15), divider(max_v, 0), diff)
    return [(b, e) for b, e in zip(r1, r2)]

min_age, max_age = np.min(age_col), np.max(age_col)
age_group = group_range(min_age, max_age, age_diff)
age_group


# In[64]:


def count_age_in_range(begin_age, end_age, ages):
    # [begin_age, end_age)
    return (np.logical_and.reduce([ages >= begin_age, ages < end_age])).sum()

ages_num_lst = [count_age_in_range(b, e, age_col) for (b, e) in age_group]

# test if number is correct
correct = "age counter is " + ("accurate" if sum(ages_num_lst) == age_col.size else "inaccurate")
correct


# In[65]:


# printer for printing out number by ranges
def age_group_printer(age_group, ages_num_lst):
    diff = age_group[0][1] - age_group[0][0]
    print("age distance = {0}, min age = {1}, max age = {2}\n".format(diff, min_age, max_age))

    for i, (b, e) in enumerate(age_group):
        print("number of ages within [{0} ~ {1}) = {2}".format(b, e, ages_num_lst[i]))

age_group_printer(age_group, ages_num_lst)


# In[66]:


# age filtering
# use age_col, to target_col where the element satisfies age condition(within range) to be true
# ex: for age within [20, 10) accroding to grouping table, num = 31. The number of True in target_col should be 31
# and the numbre of rows in target_arr should be 31 also

# target_col = np.logical_and.reduce([age_col >= 10, age_col < 20])
# target_arr = data[target_col]
# target_arr.shape    # => (31, 48)
# then do the same thing to other age groups, and append tog


# In[67]:


# target_col = np.logical_and.reduce([age_col >= 20, age_col < 30])
# target_arr = data[target_col]
# target_arr.shape


# In[68]:


def filter_age_group(select_num,age_group,ages_num_lst,data):
  data_set = np.zeros(len(data[0])) # a 1-d array with 48 zeros, used to initialize
  data_index = [0 for i in range(len(ages_num_lst))] # an array with group number of zeros
  for d in data[0:]:
    for i,(b,e) in enumerate(age_group):
      if data_index[i] < select_num:
        if b <= d[1] and e > d[1]:
          data_set = np.vstack((data_set,d))
          data_index[i] += 1
          break
  return data_set[1:]
  
final_data = filter_age_group(100,age_group,ages_num_lst,data)
final_data.shape


# In[69]:


data[0:]


# In[70]:


# testing
age_col = final_data[:, 1].astype(np.int)
ages_num_lst = [count_age_in_range(b, e, age_col) for (b, e) in age_group]
age_group_printer(age_group, ages_num_lst)
# the result: 


# In[71]:


final_data = np.vstack((header,final_data))
final_data.shape
np.savetxt("age_kept_output.txt", final_data, delimiter=',', fmt='%s')


# In[72]:


# to obtain filtered rows
combine_data = np.vstack((data, final_data))
combine_df = pd.DataFrame(combine_data)
# combine_df.shape # (2306, 48)  624+1681+1=2306
filtered_df = combine_df.drop_duplicates(keep=False)  # or inplace=True
filtered_df.reset_index()
# filtered_df.shape # (1058, 48)  Correct. extra 1 is for column in DataFrame
filtered_data = filtered_df.as_matrix()
filtered_data = np.vstack((header,filtered_data))
np.savetxt("age_filtered_output.txt", filtered_data, delimiter=',', fmt='%s')

