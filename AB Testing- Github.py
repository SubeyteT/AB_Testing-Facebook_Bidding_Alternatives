########### CASE:

# A website wants to promote on Facebook. Facebook has two types of bidding alternatives.
# Both has different advantages, both are useful for different needs.
# AB testing is a perfect method to analyze better the matching bidding.

########### VARIABLES:

# Impression – Ad views
# Click – Indicates the number of clicks on the displayed ad.
# Purchase – Indicates the number of products purchased after the ads clicked.
# Earning – Income after purchases

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
from helpers.helpers import check_df

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.expand_frame_repr', False)

df_ctrl = pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Control Group")
df_test = pd.read_excel("datasets/ab_testing.xlsx", sheet_name="Test Group")

df_ctrl.describe().T
df_test.describe().T

df_test.head()
df_ctrl.head()

df_ctrl["group"] = "control"
df_test["group"] = "test"

df = pd.concat([df_ctrl, df_test], axis=0)

df["don_oranı"] = df["Purchase"] / df["Impression"]

df.head()
check_df(df)

####################################
# TASK 1: Define the hypothesis of the A/B test.
####################################

# H0: Ask as purchase between Maxbidding (A, control) and Avgbidding (B, test) there is no
# statistically significant difference.
# H1: ... there is a difference.

df[df["group"] == "control"].mean()  # Purchase        550.89406
# Impression   101711.44907
# Click          5100.65737
# Purchase        550.89406
# Earning        1908.56830
# don_oranı         0.00558  dthere are more clicks even though it is less viewed but the purchase is down

# 0.00558

df[df["group"] == "test"].mean()     # Purchase        582.10610
# Impression   120512.41176
# Click          3967.54976
# Purchase        582.10610
# Earning        2514.89073
# don_oranı         0.00492 more views, less clicks, more purchases.
# # it can be said that the target audience where the ad is shown is better

####################################
# TASK 2: Perform the hypothesis test. Comment on whether the results are statistically significant.
####################################

###############
# Normality Assumption: provided.
###############

# H0: Assumption of normal distribution is provided.
# H1:..not provided.

# p<0.05 HO: Rejected , p>0.05 HO: Not rejected

test_stat, pvalue = shapiro(df.loc[df["group"] == "control"]["don_oranı"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.9857, p-value = 0.8844

test_stat, pvalue = shapiro(df.loc[df["group"] == "control"]["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.9773, p-value = 0.5891

test_stat, pvalue = shapiro(df.loc[df["group"] == "test"]["don_oranı"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.9714, p-value = 0.3980

test_stat, pvalue = shapiro(df.loc[df["group"] == "test"]["Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.9589, p-value = 0.1541

# p > 0.05, H0 Not rejected

#################
# Assumption of Variance Homogeneity: provided.
#################

# H0: Variances are homogeneous.
# H1: ... is not.

test_stat, pvalue = levene(df.loc[df["group"] == "control", "don_oranı"],
                           df.loc[df["group"] == "test", "don_oranı"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.0736, p-value = 0.7868  H0: Not rejected

test_stat, pvalue = levene(df.loc[df["group"] == "control", "Purchase"],
                           df.loc[df["group"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 2.6393, p-value = 0.1083  H0: Not rejected


# Both the normality assumption and the homogeneity of
# variance assumption are provided since H0 hypotheses cannot be rejected.
# To measure the hypotheses since both hypotheses are provided
# Two-Sample T-Test, which is a parametric test, is required.

#################
# INDEPENDENT TWO SAMPLE T TEST (parametric test) as assumptions are provided
#################

test_stat, pvalue = ttest_ind(df.loc[df["group"] == "control", "Purchase"],
                              df.loc[df["group"] == "test", "Purchase"],
                              equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = -0.9416, p-value = 0.3493   H0: Not rejected.
# there is no statistically significant difference between the two methods.


# Although there is no significant difference in purchasing, earnings, number of ad views,
# Since there does not appear to be a proportional increase between values such as
# buying
# It is also necessary to calculate via Purchase/ Impression.

test_stat, pvalue = ttest_ind(df.loc[df["group"] == "control", "don_oranı"],
                              df.loc[df["group"] == "test", "don_oranı"],
                              equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

# Test Stat = 1.9423, p-value = 0.0557  H0: NOT REJECTED
####################################
# TASK 3: Which test did you use, state the reasons.
####################################

# Assumptions of normality and homogeneity of variance were provided by Shapiro and Levene tests.
# It is understood that both groups have a normal distribution and their variances are homogeneously distributed.
# Since these two prerequisites are met, the Two Sample T Test, which is a parametric test, was applied on the data set.
# This test result shows that we cannot reject the H0 hypothesis, Avg.bidding and Max. bidding
# showed that there was no significant difference in terms of sales rates.

####################################
# TASK 4: Based on your answer in Task 2, what is your advice to the client?
####################################
plt.figure(figsize=(10,8), dpi= 80)
sns.pairplot(df, kind="scatter", hue="group", plot_kws=dict(s=80, edgecolor="white", linewidth=2.5))
plt.show()

# Although ist in terms of purchasing between the two methods. die. Although there is no significant difference
# The conversion rate has a value of p: very close to 0.05, which is an observation made in a short time on the dataset.
# means that there may be a difference over time.
# Depending on the conversion rate, the increase in earnings should also be checked.

####################################
# 2. AB TEST:
####################################
# H0: # H0: ist in terms of gain between Maxbidding (A, control) and Avgbidding (B, test). die.
    # There is no significant difference.
# H0: # H1: ... there is no difference.

# Assumption of Normality: provided.
#################

# H0: Assumption of normal distribution is provided.
# H1:..not provided.


test_stat, pvalue = shapiro(df.loc[df["group"] == "control"]["Earning"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# TTest Stat = 0.9756, p-value = 0.5306

test_stat, pvalue = shapiro(df.loc[df["group"] == "test"]["Earning"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.9780, p-value = 0.6163
# Assumption of Variance Homogeneity: provided.


#################

# H0: Variances are homogeneous.
# H1: ... is not.

test_stat, pvalue = levene(df.loc[df["group"] == "control", "Earning"],
                           df.loc[df["group"] == "test", "Earning"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = 0.3532, p-value = 0.5540  H0: NOT REJECTED

# AB TEST:

test_stat, pvalue = ttest_ind(df.loc[df["group"] == "control", "Earning"],
                              df.loc[df["group"] == "test", "Earning"],
                              equal_var=True)
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
# Test Stat = -9.2545, p-value = 0.0000

## p value < 0.05, H0: REJECTED

sns.lineplot(x = "don_oranı", y = "Earning", hue = "group", data = df);
plt.show()


# CONCLUSION:

# Max bidding method targets customers who can be found with the determined max price.
# In this case, users above the specified fee cannot be accessed, but those who fall far below can also be accessed.
# Avg. With bidding, it reaches users from both segments by balancing the above and below the determined price.
# a more strategic advertising fee will be made.
# Therefore, it is also important that the resulting purchase is the result of how many people see the advertisement.
# Although the purchase did not show a statistically significant increase with the new system,
# It is thought that there may be an increase in earnings due to the possible increase in the conversion rate (purchase per click).
# For this reason, which will bring more profit than the result obtained after the 2nd EU test.
# It is recommended to use the AVERAGE BIDDING method.