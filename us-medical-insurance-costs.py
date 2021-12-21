#!/usr/bin/env python
# coding: utf-8

# # U.S. Medical Insurance Costs

# In[308]:


import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[309]:


df_insurance = pd.read_csv("insurance.csv") #this pd.read_csv method lets create df directly from .csv file
df_insurance.index = range(1,1339) #sets index in proper range, to avoid calculation problems with row[0]


# In[310]:


print(df_insurance)


# ## **Determining the max, min and the average value for each column:**

# In[311]:


def max_min_mean(column):
    max = df_insurance[column].max()
    min = df_insurance[column].min()
    mean = round(df_insurance[column].mean(),2)
    
    return "{column}: max value is {max}, min value is {min}, average value is {mean}".format(max=max, min=min, mean=mean, column=column)

[max_min_mean('age'), max_min_mean('bmi'), max_min_mean('charges'), max_min_mean('children')] #run max_min_mean method with every column, we need


# ## **Verifying representation of each age group.**

# In[312]:


def age_representation_calc():
    age_representation = [0,0,0,0,0]
    for age in df_insurance['age']:
        if age <= 25:
            age_representation[0] += 1
        elif age <=35:
            age_representation[1] += 1
        elif age <= 45:
            age_representation[2] += 1
        elif age <= 55:
            age_representation[3] += 1
        else:
            age_representation[4] += 1
    return age_representation


# In[313]:


age_representation = age_representation_calc()


# In[314]:


fig, ax = plt.subplots(figsize = (10,8))

x = np.array(["0-25", "26-35", "36-45", "46-55", "56<"])
y = np.array(age_representation)

ax.set_title('\nAge groups representation', fontweight ='bold', fontsize = 15)
ax.set_xlabel('\nPopulation of each age groups', fontweight ='bold', fontsize = 15)
ax.set_ylabel('Age groups', fontweight ='bold', fontsize = 18)
 
plt.barh(x, y, color='#ffcc66', edgecolor='#331a00')
plt.show()


# ## **Verifying representation of each sex.**

# In[315]:


def sex_representation_calc():
    male = round((df_insurance['sex'].value_counts().male/ 1338 )*100,2)
    female = round((df_insurance['sex'].value_counts().female/ 1338 )*100,2)
    return "{column}: in this dataset there is {male}% of males, and {female}% of females.".format(column='sex', male=male, female=female)


# In[316]:


sex_representation_calc()


# ## **Verifying representation of bmi sections for respondents.**

# In[317]:


def bmi_representation_calc():
    bmi_representation = [0,0,0,0,0,0]
    for bmi in df_insurance['bmi']:
        if bmi <=18:
            bmi_representation[0] += 1
        elif bmi <= 25:
            bmi_representation[1] += 1
        elif bmi <= 30:
            bmi_representation[2] += 1
        elif bmi <= 34:
            bmi_representation[3] += 1
        elif bmi <= 40:
            bmi_representation[4] += 1
        else:
            bmi_representation[5] += 1
    return bmi_representation


# In[318]:


bmi_representation = bmi_representation_calc()


# In[319]:


labels = ['Underweight', 'Normal weight', 'Overweight', 'Obese 1st degree', 'Obese 2nd degree', 'Obese 3rd degree']
sizes = bmi_representation_calc()
explode = (0, 0.2, 0, 0, 0, 0)
fig1, ax1 = plt.subplots(figsize = (10,8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle = 0, shadow=True )


plt.show()


# ## **Verifying representation of each region.**

# In[320]:


def region_representation_calc():
    southeast = round((df_insurance['region'].value_counts().southeast / 1338) * 100, 2)
    southwest = round((df_insurance['region'].value_counts().southwest / 1338) * 100, 2)
    northeast = round((df_insurance['region'].value_counts().northeast / 1338) * 100, 2)
    northwest = round((df_insurance['region'].value_counts().northwest / 1338) * 100, 2)
    return [southeast, southwest, northeast, northwest]


# In[321]:


region_representation_calc()[0] ### do poprawienia zwrotka na tekstowa dla wszystkich wartości


# In[322]:


fig2, ax2 = plt.subplots(figsize = (10,8))
x = ['South East', 'South West', 'North East', 'North West']
y = region_representation_calc()

ax2.set_xlabel('\nReqions of U.S.', fontweight ='bold', fontsize = 15)
ax2.set_ylabel('\nPercentage of each region\'s representation', fontweight ='bold', fontsize = 13)
ax2.bar(x, y, width=0.4, color=['lightgreen', 'green', 'darkolivegreen', 'darkgreen'], )

plt.show()


# ## **Smokers share** 

# In[323]:


def smokers_share_calc():
    smokers_quantity = df_insurance['smoker'].value_counts().yes
    smokers_share = round((df_insurance['smoker'].value_counts().yes / 1338) * 100, 2)
    return "{smokers_quantity} ({smokers_share}%) of respondents smoke.".format(smokers_share=smokers_share, smokers_quantity=smokers_quantity)


# In[324]:


smokers_share = smokers_share_calc()


# In[325]:


labels = ['Smokers', 'Non smokers']
smokers = [20.48, (100-20.48)]
explode = (0.2,0)
figsize = (10,8)
fig1, ax1 = plt.subplots(figsize=(10,8))
ax1.pie(smokers, explode=explode, labels=labels, autopct='%1.1f%%', startangle = 90, shadow=True, colors = ["red", "green"])


plt.show()


# ## **Smokers in each age groups**

# In[326]:


smokers_only = df_insurance[df_insurance["smoker"] == 'yes'] #zwraca df z samymi palaczami
print(smokers_only)


# In[327]:


non_smokers_only = df_insurance[df_insurance["smoker"] == 'no']


# In[328]:


def smokers_age_representation_calc(smokers_only):
    smokers_age_representation = [0,0,0,0,0]
    for age in smokers_only['age']:
        if age <= 25:
            smokers_age_representation[0] += 1
        elif age <=35:
            smokers_age_representation[1] += 1
        elif age <= 45:
            smokers_age_representation[2] += 1
        elif age <= 55:
            smokers_age_representation[3] += 1
        else:
            smokers_age_representation[4] += 1
    return smokers_age_representation

def non_smokers_age_representation_calc(non_smokers_only):
    non_smokers_age_representation = [0,0,0,0,0]
    for age in non_smokers_only['age']:
        if age <= 25:
            non_smokers_age_representation[0] += 1
        elif age <=35:
            non_smokers_age_representation[1] += 1
        elif age <= 45:
            non_smokers_age_representation[2] += 1
        elif age <= 55:
            non_smokers_age_representation[3] += 1
        else:
            non_smokers_age_representation[4] += 1
    return non_smokers_age_representation


# In[329]:


smokers_age_representation_calc(smokers_only)


# In[330]:


non_smokers_age_representation_calc(non_smokers_only)


# In[331]:


barWidth = 0.25
fig3 = plt.subplots(figsize =(10, 8))

S = smokers_age_representation_calc(smokers_only)
NS = non_smokers_age_representation_calc(non_smokers_only)

br_S = np.arange(len(S))
br_NS = [x + barWidth for x in br_S]

plt.bar(br_S, S, color='red', width=barWidth, label='Smokers', edgecolor='black')
plt.bar(br_NS, NS, color='lightgreen', width=barWidth, label='Non Smokers', edgecolor='black')

plt.xlabel('Smokers and non smokers in each age group', fontweight ='bold', fontsize = 15)
plt.ylabel('Population', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(S))], ['< 25', '26-35', '36-45', '46-55', '56 <'])

plt.legend()
plt.show()


# ## Smokers in regions

# In[332]:


smokers_only = df_insurance[df_insurance["smoker"] == 'yes']
print(smokers_only)


# In[333]:


def region_population_calc():
    region_population = [0, 0, 0, 0]
    for region in df_insurance["region"]:
        if region == "southwest":
            region_population[0] += 1
        elif region == "southeast":
            region_population[1] += 1
        elif region == "northwest":
            region_population[2] += 1
        else:
            region_population[3] += 1
    return region_population


# In[334]:


region_population = region_population_calc()


# In[335]:


print(region_population)


# In[336]:


print(str(region_population[0]) + " respondents live in South West of U.S., " + str(region_population[1]) + " respondents live in South East of U.S., " + str(region_population[2]) + " respondents live in North West of U.S., " + str(region_population[3]) + " respondents live in North East of U.S.")


# In[337]:


def smokers_per_region_calc():
    smokers_per_region = [0, 0, 0, 0]
    for region in smokers_only["region"]:
        if region == "southwest":
            smokers_per_region[0] += 1
        elif region == "southeast":
            smokers_per_region[1] += 1
        elif region == "northwest":
            smokers_per_region[2] += 1
        else:
            smokers_per_region[3] += 1
    return smokers_per_region


# In[338]:


smokers_per_region = smokers_per_region_calc()


# In[339]:


def smokers_per_region_percentage_calc():
    south_west_smokers_percentage = round((smokers_per_region[0] / region_population[0]), 2)
    south_east_smokers_percentage = round((smokers_per_region[1] / region_population[1]), 2)
    north_west_smokers_percentage = round((smokers_per_region[2] / region_population[2]), 2)
    north_east_smokers_percentage = round((smokers_per_region[3] / region_population[3]), 2)
    
    return south_west_smokers_percentage, south_east_smokers_percentage, north_west_smokers_percentage, north_east_smokers_percentage
    


# In[340]:


smokers_per_region_percentage = list(smokers_per_region_percentage_calc())


# In[341]:


print(smokers_per_region_percentage)


# In[342]:


south_west_smokers_percentage = [18, (100-18)]
south_east_smokers_percentage = [25, (100-25)]
north_west_smokers_percentage = [18, (100-18)]
north_east_smokers_percentage = [21, (100-21)]

labels = ['Smokers', 'Non Smokers']

fig = plt.figure(figsize=(10,8), dpi = 144)
ax1 = fig.add_subplot(221)
ax1.pie(south_west_smokers_percentage, labels = labels, explode = (0.2, 0), shadow = True, colors = ["red", "green"], startangle = 90, autopct='%1.1f%%')
ax1.set_title("Smokers percentage in South West")

ax2 = fig.add_subplot(222)
ax2.pie(south_east_smokers_percentage, labels = labels, explode = (0.2, 0), shadow = True, colors = ["red", "green"], startangle= -90, autopct='%1.1f%%')
ax2.set_title("Smokers percentage in South East")

ax3 = fig.add_subplot(223)
ax3.pie(north_west_smokers_percentage, labels = labels, explode = (0.2, 0), shadow = True, colors = ["red", "green"], startangle = 90, autopct='%1.1f%%')
ax3.set_title("Smokers percentage in North West")

ax4 = fig.add_subplot(224)
ax4.pie(north_east_smokers_percentage, labels = labels, explode = (0.2, 0), shadow = True, colors = ["red", "green"], startangle = -90, autopct='%1.1f%%')
ax4.set_title("Smokers percentage in North East")


plt.show()


# ## Smokers in each sex group

# In[343]:


females_only = df_insurance[df_insurance['sex'] == "female"]


# In[344]:


males_only = df_insurance[df_insurance['sex'] == 'male']


# In[345]:


non_smoker_females_only = females_only['smoker'].value_counts().no


# In[346]:


non_smoker_males_only = males_only['smoker'].value_counts().no


# In[347]:


smoker_females_only = females_only['smoker'].value_counts().yes


# In[348]:


smoker_males_only = males_only['smoker'].value_counts().yes


# In[349]:


labels = ['Females', 'Males']
smokers = [smoker_females_only, smoker_males_only]
non_smokers = [non_smoker_females_only, non_smoker_males_only]
width = 0.35

fig, ax = plt.subplots(figsize=(10, 8))

ax.bar(labels, non_smokers, width, label = "Non Smokers", color = "green", edgecolor='black')
ax.bar(labels, smokers, width, label = "Smokers", color = "red", edgecolor='black', bottom=non_smokers)

for bar in ax.patches:
    height = bar.get_height()
    width = bar.get_width()
    x = bar.get_x()
    y = bar.get_y()
    label_text = height
    label_x = x + width / 2
    label_y = y + height / 2
    ax.text(label_x, label_y, label_text, ha='center', va='center', fontsize=15, fontweight = 'bold', color='white')

ax.set_ylabel("Number of smokers and nonsmokers\n", fontsize=15)
ax.set_title("Smokers in each sex group\n", fontsize=20, fontweight = "bold")
ax.legend(loc = 'center')


plt.show()


# ## Smoking and number of children

# In[350]:


df_insurance_by_children = df_insurance.groupby(['children'])


# In[351]:


def children_counter_calc():
    children_counter = [0, 0, 0, 0, 0, 0]
    for children in df_insurance['children']:
        if children == 0:
            children_counter[0] += 1
        elif children == 1:
            children_counter[1] += 1
        elif children == 2:
            children_counter[2] += 1
        elif children == 3:
            children_counter[3] += 1
        elif children == 4:
            children_counter[4] += 1
        else:
            children_counter[5] += 1
    return children_counter
                


# In[352]:


print(list(children_counter_calc()))


# In[353]:


def percentage_of_smokers_and_children_calc():
    smokers_and_children = [0, 0, 0, 0, 0, 0]
    children_0 = df_insurance[df_insurance['children'] == 0]
    smokers_and_children[0] = round((children_0['smoker'].value_counts().yes/children_counter[0]) * 100, 2)
    
    children_1 = df_insurance[df_insurance['children'] == 1]
    smokers_and_children[1] = round((children_1['smoker'].value_counts().yes/children_counter[1]) * 100, 2)
    
    children_2 = df_insurance[df_insurance['children'] == 2]
    smokers_and_children[2] = round((children_2['smoker'].value_counts().yes/children_counter[2]) * 100, 2)
    
    children_3 = df_insurance[df_insurance['children'] == 3]
    smokers_and_children[3] = round((children_3['smoker'].value_counts().yes/children_counter[3]) * 100, 2)
    
    children_4 = df_insurance[df_insurance['children'] == 4]
    smokers_and_children[4] = round((children_4['smoker'].value_counts().yes/children_counter[4]) * 100, 2)
    
    children_5 = df_insurance[df_insurance['children'] == 5]
    smokers_and_children[5] = round((children_5['smoker'].value_counts().yes/children_counter[5]) * 100, 2)
    
    return smokers_and_children


# In[354]:


print(percentage_of_smokers_and_children_calc())


# In[355]:


fig, ax1 = plt.subplots(figsize = (10,8))
x = ['0', '1', '2', '3', '4', '5']
y = children_counter_calc()

ax1.set_xlabel('\nNumber of children', fontweight ='bold', fontsize = 15)
ax1.set_ylabel('Number of respondents\nwith certain amount of children\n', fontweight ='bold', fontsize = 15)
ax1.bar(x, y, width=0.4, color=['lightgreen', 'green', 'darkolivegreen', 'darkgreen'], )

ax2 = ax1.twinx()

z = percentage_of_smokers_and_children_calc()
ax2.set_ylabel('\nPercentage of Smokers', fontweight = 'bold', fontsize = 15)
ax2.plot(x, z, linewidth=3.0)

plt.show()


# ## Smoker in each BMI group

# In[356]:


def smokers_in_bmi_groups_calc():
    smokers_in_bmi_groups = [0, 0, 0, 0, 0, 0]
    smokers_in_uderweight = df_insurance[df_insurance['bmi'] <= 18]
    smokers_in_bmi_groups[0] = round((smokers_in_uderweight['smoker'].value_counts().yes /bmi_representation[0])*100, 2)
    
    smokers_in_normal_weight = df_insurance[df_insurance['bmi'] <= 25]
    smokers_in_bmi_groups[1] = round((smokers_in_normal_weight['smoker'].value_counts().yes / bmi_representation[1])*100, 2)
    
    smokers_in_overweight = df_insurance[df_insurance['bmi'] <= 30]
    smokers_in_bmi_groups[2] = round((smokers_in_overweight['smoker'].value_counts().yes / bmi_representation[2])*100, 2)
    
    smokers_in_obes_1st = df_insurance[df_insurance['bmi'] <= 35]
    smokers_in_bmi_groups[3] = round((smokers_in_obes_1st['smoker'].value_counts().yes / bmi_representation[3])*100, 2)
    
    smokers_in_obes_2nd = df_insurance[df_insurance['bmi'] <= 40]
    smokers_in_bmi_groups[4] = round((smokers_in_obes_2nd['smoker'].value_counts().yes / bmi_representation[4])*100, 2)
    
    smokers_in_obes_3rd = df_insurance[df_insurance['bmi'] > 40]
    smokers_in_bmi_groups[5] = round((smokers_in_obes_3rd['smoker'].value_counts().yes / bmi_representation[5])*100, 2)
    
    return smokers_in_bmi_groups


# In[357]:


smokers_in_bmi_groups = smokers_in_bmi_groups_calc()


# In[358]:


def non_smokers_in_bmi_groups_calc():
    non_smokers_in_bmi_groups = [0, 0, 0, 0, 0, 0]
    non_smokers_in_uderweight = df_insurance[df_insurance['bmi'] <= 18]
    non_smokers_in_bmi_groups[0] = round((non_smokers_in_uderweight['smoker'].value_counts().no /bmi_representation[0])*100, 2)
    
    non_smokers_in_normal_weight = df_insurance[df_insurance['bmi'] <= 25]
    non_smokers_in_bmi_groups[1] = round((non_smokers_in_normal_weight['smoker'].value_counts().no / bmi_representation[1])*100, 2)
    
    non_smokers_in_overweight = df_insurance[df_insurance['bmi'] <= 30]
    non_smokers_in_bmi_groups[2] = round((non_smokers_in_overweight['smoker'].value_counts().no / bmi_representation[2])*100, 2)
    
    non_smokers_in_obes_1st = df_insurance[df_insurance['bmi'] <= 35]
    non_smokers_in_bmi_groups[3] = round((non_smokers_in_obes_1st['smoker'].value_counts().no / bmi_representation[3])*100, 2)
    
    non_smokers_in_obes_2nd = df_insurance[df_insurance['bmi'] <= 40]
    non_smokers_in_bmi_groups[4] = round((non_smokers_in_obes_2nd['smoker'].value_counts().no / bmi_representation[4])*100, 2)
    
    non_smokers_in_obes_3rd = df_insurance[df_insurance['bmi'] > 40]
    non_smokers_in_bmi_groups[5] = round((non_smokers_in_obes_3rd['smoker'].value_counts().no / bmi_representation[5])*100, 2)
    
    return non_smokers_in_bmi_groups


# In[359]:


non_smokers_in_bmi_groups = non_smokers_in_bmi_groups_calc()


# In[360]:


fig = plt.figure(figsize=(12,8), dpi = 144)
fig.suptitle('Percentage of smokers and non smokers in each BMI group', fontsize=15, fontweight='bold')

labels = ['Underweight', 'Normal weight', 'Overweight', 'Obestity 1st degree', 'Obesity 2nd degree', 'Obestity 3rd degree']
smokers_values = smokers_in_bmi_groups_calc()
non_smokers_values = non_smokers_in_bmi_groups_calc()
colors = ['#FF0000', '#0000FF', '#FFFF00', '#ADFF2F', '#FFA500']

ax1 = fig.add_subplot(221)
ax1 = plt.pie(smokers_values, labels=labels, autopct='%1.1f%%', pctdistance = 0.75, startangle=90)
my_circle = plt.Circle((0,0), 0.5, color = 'white')
p = plt.gcf()
p.gca().add_artist(my_circle)

ax2 = fig.add_subplot(222)
ax2 = plt.pie(non_smokers_values, labels=labels, autopct='%1.1f%%', pctdistance = 0.75, startangle=90)
my_circle_2 = plt.Circle((0,0), 0.5, color = 'white')
p.gca().add_artist(my_circle_2)


plt.show()

