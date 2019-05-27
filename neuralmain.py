import matplotlib.pyplot as plt
import math
import time
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet 
from pybrain.supervised.trainers import BackpropTrainer

ds = SupervisedDataSet(2,1)

#4 samples of Neural Nets, where the first input (0.8 for 8 hours) represents the hours sleeped, and the second input (0.4 for 4 hours) the hours studied.
#the third of the first sample (0.7 for 7.0 grade)  represents the grade of the student who sleeped 8 hours and sleeped 4 hours. 
#i.e.: -> ((0.8, 0.4), '8 hours sleeped, 4 hours studied'. the (0.7) is the grade of this student. 
ds.addSample((0.8, 0.4), (0.7))
ds.addSample((0.2, 0.4), (0.45))
ds.addSample((1.0, 0.6), (0.75))
ds.addSample((0.6, 0.8), (0.90))

#here whe have the build function for the neural nets
nn = buildNetwork(2, 4, 1, bias=True)

#the trainer function, who train the IA in those respective samples that we created.
trainer = BackpropTrainer(nn, ds)

#creating vectors to save the valors of time that the IA learned each task and quantity of tasks
times = []
tasks = []
task = 1
legendY = []

#this 'for' saves the first 5 tasks and respective time of task learned by the IA.
for i in range(2000):
    while task < 5:
        start = time.perf_counter()
        print(trainer.train())
        end = time.perf_counter()
        time = round(end - start,4)
        times.append(time)
        tasks.append(task)
        task += 1
    
    print(trainer.train())


vez = 1

while vez <= 5:
    valid_sleeped = False
    valid_studied = False
    while valid_sleeped == False:
        sleeped = input('Sleeped (Ex.: 0.4 = 4 hours)\n')
        try:
            sleeped = float(sleeped)
            if sleeped > 1 or sleeped < 0:
                print("Entry numbers in format '0.1' (for 1 hour)")
                print("\n")
            else:
                valid_sleeped = True
        except:
            print("Entry numbers in format '0.1 (for 1 hour)")

    while valid_studied == False:
        studied = input('Studied (Ex.: 0.6 = 6 hours)\n')
        try:
            studied = float(studied)
            if studied > 1 or studied < 0:
               print("Entry numbers greater than 0 and less then 1.0, use the format '0.2' for 2 hours, i.e.")
            else:
                valid_studied = True
        except:
            print("Entry numbers in format '0.1 (for 1 hour)")
            
    z = nn.activate((sleeped, studied))[0] * 10.0
    print('Grade precision: ' + str(z))
    vez += 1
print(tasks,times)
legendX = ['1st Task', '2 Task', '3 Task', '4 Task']
plt.xticks(tasks,legendX)
plt.plot(tasks,times)
plt.show()

