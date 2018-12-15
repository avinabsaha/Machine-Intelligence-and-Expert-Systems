from __future__ import print_function 
import scipy.stats
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


MAX = 1000

# Third Eye Technologies

# Define the Normal Distributions
malignant_dist_third_eye = scipy.stats.norm(37,1)
benign_dist_third_eye = scipy.stats.norm(32,4)

# Arrays for storing values of False Positive, False Negative, True Positive 
FPF = np.zeros(10000)
FNF = np.zeros(10000)
TPF = np.zeros(10000)




for thres in range (0,10000):
        threshold = float(thres)/100
        FPF[thres] = benign_dist_third_eye.cdf(MAX)-benign_dist_third_eye.cdf(threshold)
        FNF[thres] = malignant_dist_third_eye.cdf(threshold)
        TPF[thres] = 1 - FNF[thres]

plt.figure(0)
plt.grid(True)
plt.plot(FPF, TPF,'r',linewidth=2.0) 
plt.xlabel('FPF')
plt.ylabel('TPF')
plt.title('ROC Curve For Third Eye')
plt.savefig('ROC Curve For Third Eye')

# Integrated from reverse direction hence negative sign
AUC_ThirdEye = -integrate.trapz(TPF, FPF)


# Competitor

# Define the Normal Distributions
malignant_dist_competitor = scipy.stats.norm(37,2)
benign_dist_competitor = scipy.stats.norm(32,3)

# Arrays for storing values of False Positive, False Negative, True Positive 
FPF = np.zeros(10000)
FNF = np.zeros(10000)
TPF = np.zeros(10000)


for thres in range (0,10000):
        threshold = float(thres)/100
        FPF[thres] = benign_dist_competitor.cdf(MAX)-benign_dist_competitor.cdf(threshold)
        FNF[thres] = malignant_dist_competitor.cdf(threshold)
        TPF[thres] = 1 - FNF[thres]

plt.figure(1)
plt.grid(True)
plt.plot(FPF, TPF,'b',linewidth=2.0) 
plt.xlabel('FPF')
plt.ylabel('TPF')
plt.title('ROC Curve For Competitor')
plt.savefig('ROC Curve For Competitor')

# Integrated from reverse direction hence negative sign
AUC_Competitor = -integrate.trapz(TPF, FPF)

print("Area Under Curve For Third Eye:",AUC_ThirdEye)
print("Area Under Curve For Competitor:",AUC_Competitor)
if (AUC_ThirdEye >= AUC_Competitor):
        print("Third Eye Technology is preferred")
if (AUC_ThirdEye < AUC_Competitor):
        print("Competitor is preferred")