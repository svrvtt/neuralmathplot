import matplotlib.pyplot as plt
import math
import time
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet 
from pybrain.supervised.trainers import BackpropTrainer

ds = SupervisedDataSet(2,1)

#4 exemplos pra IA, 
ds.addSample((0.8, 0.4), (0.7))
ds.addSample((0.2, 0.4), (0.45))
ds.addSample((1.0, 0.6), (0.75))
ds.addSample((0.6, 0.8), (0.90))

nn = buildNetwork(2, 4, 1, bias=True)

trainer = BackpropTrainer(nn, ds)


tempos = []
tarefas = []
tarefa = 1
legendaY = []
for i in range(2000):
    while tarefa < 5:
        inicio = time.perf_counter()
        print(trainer.train())
        fim = time.perf_counter()
        tempo = round(fim - inicio,4)
        tempos.append(tempo)
        tarefas.append(tarefa)
        tarefa += 1
    
    print(trainer.train())


vez = 1

while vez <= 5:
    valid_dormiu = False
    valid_estudou = False
    while valid_dormiu == False:
        dormiu = input('Dormiu (Ex.: 0.4 = 4 horas)\n')
        try:
            dormiu = float(dormiu)
            if dormiu > 1 or dormiu < 0:
                print("Informe numeros no formato '0.1' (para 1 hora)")
                print("\n")
            else:
                valid_dormiu = True
        except:
            print("Informe numeros no formato '0.1 (para 1 hora)")

    while valid_estudou == False:
        estudou = input('Estudou (Ex.: 0.6 = 6 horas)\n')
        try:
            estudou = float(estudou)
            if estudou > 1 or estudou < 0:
               print("Informe numeros maiores que 0 e menores que 1.0, utilize o formato '0.2' para 2 horas, i.e.")
            else:
                valid_estudou = True
        except:
            print("Informe numeros no formato '0.1 (para 1 hora)")
            
    z = nn.activate((dormiu, estudou))[0] * 10.0
    print('precisao de nota: ' + str(z))
    vez += 1
print(tarefas,tempos)
legendaX = ['1a Tarefa', '2a Tarefa', '3a Tarefa', '4a Tarefa']
plt.xticks(tarefas,legendaX)
plt.plot(tarefas,tempos)
plt.show()

