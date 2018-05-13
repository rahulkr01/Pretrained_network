import re
import numpy as np
import matplotlib.pyplot as plt

f=open("pretrained_loss.txt")
lines=f.readlines()
print(lines)
lines=[ float(re.findall(r"[-+]?\d*\.\d+|\d+",line.split('-')[2])[0]) for line in lines]
arr1=np.arange(len(lines))
lines=np.array(lines)
plt.figure()
plt.plot(arr1,lines)
plt.savefig("pretrained_train_loss.png")
plt.show()

