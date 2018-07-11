from  scipy.stats.stats import pearsonr
import numpy as np
### 3-state model correlation analysis
x_1 = np.array([0.78, 0.76, 0.73, 0.75, 0.73, 0.74])
y_1 = np.array([0.56, 0.54, 0.51, 0.53, 0.52, 0.52])

x_2 = np.array([0.1,  0.11, 0.12, 0.11, 0.12, 0.12])
y_2 = np.array([0.24, 0.26, 0.27, 0.25, 0.26, 0.27])

x_3 = np.array([0.12,0.13, 0.15, 0.13,0.15,0.14])
y_3 = np.array([0.2, 0.2,  0.21, 0.21,0.21,0.21])

xx_1 = np.array([0.76, 0.75, 0.72, 0.74, 0.71, 0.72])
yy_1 = np.array([0.51, 0.5, 0.35, 0.47, 0.45, 0.46])

xx_2 = np.array([0.1, 0.12, 0.13, 0.12, 0.16, 0.15])
yy_2 = np.array([0.14, 0.2, 0.26, 0.22, 0.24, 0.29])

xx_3 = np.array([0.14, 0.13, 0.15, 0.14, 0.12, 0.14])
yy_3 = np.array([0.35, 0.3, 0.39, 0.3, 0.32, 0.28])

xxx_1 = np.array([0.8, 0.78, 0.76, 0.77, 0.73, 0.73])
yyy_1 = np.array([0.52, 0.49, 0.47, 0.48, 0.5, 0.5])

xxx_2 = np.array([0.1, 0.11, 0.13, 0.11, 0.14, 0.14])
yyy_2 = np.array([0.21, 0.25, 0.22, 0.23, 0.21, 0.24])

xxx_3 = np.array([0.1, 0.11, 0.11, 0.12, 0.13, 0.13])
yyy_3 = np.array([0.27, 0.26, 0.3, 0.28, 0.29, 0.26])

driving_1_before = np.concatenate([x_1,xx_1,xxx_1])
driving_1_after = np.concatenate([y_1,yy_1,yyy_1])

driving_2_before = np.concatenate([x_2,xx_2,xxx_2])
driving_2_after = np.concatenate([y_2,yy_2,yyy_2])


driving_3_before = np.concatenate([x_3,xx_3,xxx_3])
driving_3_after = np.concatenate([y_3,yy_3,yyy_3])



print pearsonr(driving_1_before, driving_1_after)
print pearsonr(driving_2_before, driving_2_after)
print pearsonr(driving_3_before, driving_3_after)
print pearsonr(driving_1_before, driving_2_after)
print pearsonr(driving_1_before, driving_3_after)
print pearsonr(driving_2_before, driving_3_after)
from scipy.stats import linregress
print linregress(driving_1_before, driving_1_after)

### 5-state model correlation analysis
x_0 = np.array([0.48, 0.46,0.44,0.46,0.45,0.45,0.49, 0.47, 0.41,0.44,0.41,0.41,0.49,0.46,0.48, 0.47, 0.44,0.44])
y_0 = np.array([0.32, 0.3,0.3, 0.31,0.3,0.3,0.29,0.29, 0.2, 0.28, 0.23, 0.25,0.3, 0.28, 0.27, 0.28, 0.27,0.28])

x_1 = np.array([0.23, 0.24,0.25,0.24,0.25,0.25,0.26,0.23,0.26,0.26,0.26,0.25,0.2,0.23,0.24,0.23,0.23, 0.25])
y_1 = np.array([0.24,0.24,0.28,0.24,0.24,0.24,0.34,0.29,0.26,0.26,0.31,0.27,0.26,0.24,0.27,0.26,0.29,0.26])

x_2 = np.array([0.24,0.24,0.24, 0.23,0.24,0.24,0.2,0.24,0.25,0.24,0.26,0.26,0.26,0.26,0.22,0.24,0.25,0.24])
y_2 = np.array([0.29,0.29,0.26,0.27,0.26,0.27,0.17,0.22,0.18,0.21,0.21,0.24,0.27,0.27,0.23,0.25,0.23,0.25])

x_3 = np.array([0.02,0.02,0.03,0.03,0.03,0.03,0.03,0.03,0.04,0.03,0.04,0.04,0.02,0.03,0.05,0.03,0.04,0.04])
y_3 = np.array([0.09,0.1,0.13,0.11,0.13,0.12,0.05,0.09,0.16,0.11,0.12,0.13,0.07,0.1,0.1,0.1,0.1,0.11])

x_4 = np.array([0.03,0.03,0.04,0.03,0.04,0.04, 0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.03,0.01,0.02,0.03,0.03])
y_4 = np.array([0.06,0.06,0.08,0.07,0.08,0.08,0.16,0.12,0.2,0.13,0.12,0.12,0.1,0.1,0.13,0.11,0.11,0.11])
print pearsonr(x_0, y_0)
print pearsonr(x_1, y_1)
print pearsonr(x_2, y_2)
print pearsonr(x_3, y_3)
print pearsonr(x_4, y_4)
print ('--------')
print pearsonr(x_0, y_1)
print pearsonr(x_0, y_2)
print pearsonr(x_0, y_3)
print pearsonr(x_0, y_4)
print ('--------')
print pearsonr(x_1, y_2)
print pearsonr(x_1, y_3)
print pearsonr(x_1, y_4)
print ('-----')
print pearsonr(x_2, y_3)
print pearsonr(x_2, y_4)
print ('-----')
print pearsonr(x_3, y_4)

print ('--------')
print pearsonr(x_0, x_3)
print pearsonr(y_0, y_3)

print pearsonr(x_1, x_3)
print pearsonr(y_1, y_3)

print ("-------")
print pearsonr(x_3,x_4)
print pearsonr(y_3, y_4)

"""
 r : float
        Pearson's correlation coefficient
    p-value : float
        2-tailed p-value
"""