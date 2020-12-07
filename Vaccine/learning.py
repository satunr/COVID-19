import matplotlib.pyplot as plt

actual = 0.2
learn = 0.1

initial = [0.2, 0.3, 0.4, 0.5]
T = 50

for init in initial:
    ins = init
    print (init)
    V = []
    for t in range(T):
        ins = ins + (actual - ins) * learn
        V.append(ins)

    plt.plot([t for t in range(T)], V, label = 'Learning rate ' + str(round(init, 2)))

plt.plot([t for t in range(T)], [actual for t in range(T)],
         label = 'Actual learning rate', linestyle = 'dotted', color = 'black', linewidth = 3)

plt.xlabel('Duration', fontsize = 15)
plt.ylabel('    Learning rate', fontsize = 15)
plt.legend()
plt.tight_layout()

plt.savefig('Learning.png', dpi = 300)
plt.show()