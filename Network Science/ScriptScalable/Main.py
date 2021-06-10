import matplotlib.pyplot as plt

# 11.6 100.0 100.0
# 83.8 100.0 100.0
# 8.2 100.0 100.0
# 15.4 100.0 100.0
# 4.8 65.2 14.6
# 9.0 100.0 100.0
# 4.6 100.0 7.0
# 8.4 100.0 100.0
# 4.0 100.0 6.0
# 7.4 100.0 24.2


def alternate(L):
    return [L[i * 2] for i in range(5)] + [L[i * 2 + 1] for i in range(5)]


width = 0.2
ax = plt.subplot(111)
ran = [11.6, 83.8, 8.2, 15.4, 4.8, 9.0, 4.6, 8.4, 4.0, 7.4]
op1 = [100.0, 100.0, 100.0, 100.0, 65.2, 100.0, 100.0, 100.0, 100.0, 100.0]
op2 = [100.0, 100.0, 100.0, 100.0, 14.6, 100.0, 7.0, 100.0, 6.0, 24.2]

ran = alternate(ran)
op1 = alternate(op1)
op2 = alternate(op2)

bar1 = ax.bar([i - width - 0.1 for i in range(5)], ran[5:], width, edgecolor = 'black',
       color = ['b', 'b', 'b', 'b', 'b'], alpha = 0.5, label = 'Random', hatch = '*')

bar2 = ax.bar([i for i in range(5)], op1[5:], width, edgecolor = 'black',
       color =['b', 'b', 'b', 'b', 'b'], alpha = 0.5, label = 'Opt1', hatch = '\\\\')

bar3 = ax.bar([i + width + 0.1 for i in range(5)], op2[5:], width, edgecolor = 'black',
       color =['b', 'b', 'b', 'b', 'b'], alpha = 0.5, label = 'Opt2', hatch = 'O')


ax.legend(['', ''])
plt.xticks([i for i in range(5)], [50, 100, 150, 200, 250], fontsize = 12)
plt.xlabel('Population size', fontsize = 12)
plt.ylabel('Timepoint', fontsize = 12)
plt.ylim([0, 150])

plt.legend()
plt.tight_layout()
plt.savefig('Scalabilty-75.png', dpi = 300)
plt.show()