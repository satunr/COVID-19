import numpy as np
import random, pickle
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation

from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D


'''
#-----position and color and path for counties-----------------
pos = {'Westchester': (50, 2500), 'Kings': (-100, 2000), 'New York': (200, 2000), 'Nassau': (-50, 1500), 'Queens': (150, 1500), 
        'Rockland': (50, 1000), 'Bronx': (0, 500), 'Suffolk': (50, 500), 'Ulster': (100, 500), 'Saratoga': (50, 0), 'Dutchess': (-100, -500), 
        'Orange': (200, -500), 'Albany': (-50, -1000), 'Broome': (0, -1000), 'Delaware': (50, -1000), 'Herkimer': (100, -1000), 
        'Monroe': (150, -1000), 'Schenectady': (0, -1500), 'Tioga': (50, -1500), 'Tompkins': (100, -1500), 'Erie': (0, -2000), 
        'Greene': (50, -2000), 'Montgomery': (100, -2000), 'Allegany': (-50, -2500), 'Onondaga': (0, -2500), 'Ontario': (50, -2500), 
        'Putnam': (100, -2500), 'Wyoming': (150, -2500), 'Clinton': (-25, -3000), 'Rensselaer': (25, -3000), 'Richmond': (75, -3000), 
        'Sullivan': (125, -3000), 'Chenango': (-50, -3500), 'Essex': (0, -3500), 'Hamilton': (50, -3500), 'Warren': (100, -3500), 
        'Washington': (150, -3500), 'Fulton': (-100, -4000), 'Genesee': (-50, -4000), 'Jefferson': (0, -4000), 'Niagara': (50, -4000), 
        'Oneida': (100, -4000), 'Schoharie': (150, -4000), 'Wayne': (200, -4000), 'Columbia': (0, -4500), 'Livingston': (100, -4500), 
        'Steuben': (50, -5000), 'Cortland': (0, -5500), 'Madison': (50, -5500), 'St. Lawrence': (100, -5500), 'Cayuga': (0, -6000), 
        'Oswego': (50, -6000), 'Otsego': (100, -6000), 'Chemung': (0, -6500), 'Orleans': (100, -6500), 'Chautauqua': (0, -7000), 
        'Franklin': (100, -7000), 'Cattaraugus': (0, -7500), 'Schuyler': (100, -7500), 'Lewis': (50, -8000), 'Seneca': (50, -8500), 'Yates': (50, -9000)}

color = {'Westchester': 'blue', 'Kings': 'lime', 'New York': 'lime', 'Nassau': 'yellow', 'Queens': 'yellow', 'Rockland': 'chocolate', 
        'Bronx': 'dodgerblue', 'Suffolk': 'dodgerblue', 'Ulster': 'dodgerblue', 'Saratoga': 'plum', 'Dutchess': 'greenyellow', 
        'Orange': 'greenyellow', 'Albany': 'lightcoral', 'Broome': 'lightcoral', 'Delaware': 'lightcoral', 'Herkimer': 'lightcoral', 
        'Monroe': 'lightcoral', 'Schenectady': 'gold', 'Tioga': 'gold', 'Tompkins': 'gold', 'Erie': 'springgreen', 'Greene': 'springgreen', 
        'Montgomery': 'springgreen', 'Allegany': 'skyblue', 'Onondaga': 'skyblue', 'Ontario': 'skyblue', 'Putnam': 'skyblue', 'Wyoming': 'skyblue', 
        'Clinton': 'cyan', 'Rensselaer': 'cyan', 'Richmond': 'cyan', 'Sullivan': 'cyan', 'Chenango': 'lawngreen', 'Essex': 'lawngreen', 
        'Hamilton': 'lawngreen', 'Warren': 'lawngreen', 'Washington': 'lawngreen', 'Fulton': 'mediumslateblue', 'Genesee': 'mediumslateblue', 
        'Jefferson': 'mediumslateblue', 'Niagara': 'mediumslateblue', 'Oneida': 'mediumslateblue', 'Schoharie': 'mediumslateblue', 
        'Wayne': 'mediumslateblue', 'Columbia': 'brown', 'Livingston': 'brown', 'Steuben': 'teal', 'Cortland': 'aqua', 'Madison': 'aqua', 
        'St. Lawrence': 'aqua', 'Cayuga': 'pink', 'Oswego': 'pink', 'Otsego': 'pink', 'Chemung': 'darkkhaki', 'Orleans': 'darkkhaki', 
        'Chautauqua': 'peru', 'Franklin': 'peru', 'Cattaraugus': 'limegreen', 'Schuyler': 'limegreen', 'Lewis': 'lightsteelblue', 
        'Seneca': 'sandybrown', 'Yates': 'cornflowerblue'}

path = "counties_all_0-189.p"
T = 190
threshold = 0.05
'''

#--------position and color for states--------------------
pos = {'Washington': (50, 2500), 'Illinois': (-50, 2000), 'California': (150, 1500), 'Arizona': (-100, 1000), 'Massachusetts': (200, 500), 
        'Wisonsin': (-50, 0), 'Texas': (150, -500), 'Nebraska': (-100, -1000), 'Utah': (200, -1500), 'Oregon': (50, -1500), 'Florida': (-50, -2500), 
        'New York': (50, -2500), 'Rhode Island': (150, -2500), 'Georgia': (0, -3000), 'New Hampshire': (100, -3000), 'North Carolina': (50, -3500), 
        'New Jersey': (50, -4000), 'Colorado': (-25, -4500), 'Maryland': (25, -4500), 'Nevada': (75, -4500), 'Tennessee': (125, -4500), 
        'Hawaii': (-100, -5000), 'Indiana': (-50, -5000), 'Kentucky': (0, -5000), 'Minnesota': (50, -5000), 'Oklahoma': (100, -5000), 
        'Pennsylvania': (150, -5000), 'South Carolina': (200, -5000), 'Columbia': (-50, -5500), 'Kansas': (0, -5500), 'Missouri': (50, -5500), 
        'Vermont': (100, -5500), 'Virginia': (150, -5500), 'Connecticut': (0, -6000), 'Iowa': (100, -6000), 'Louisiana': (0, -6500), 
        'Ohio': (100, -6500), 'Michigan': (0, -7000), 'South Dakota': (100, -7000), 'Arkansas': (-75, -7500), 'Delaware': (-25, -7500), 
        'Mississippi': (25, -7500), 'New Mexico': (75, -7500), 'North Dakota': (125, -7500), 'Wyoming': (175, -7500), 'Alaska': (0, -8000), 
        'Maine': (100, -8000), 'Alabama': (-25, -8500), 'Idaho': (25, -8500), 'Montana': (75, -8500), 'Puerto Rico': (125, -8500), 
        'Virgin Islands': (50, -9000), 'Guam': (50, -9500), 'West Virginia': (50, -10000), 'Northern Mariana Islands': (50, -10500)}

color = {'Washington': 'blue', 'Illinois': 'lime', 'California': 'yellow', 'Arizona': 'chocolate', 'Massachusetts': 'dodgerblue', 
        'Wisonsin': 'plum', 'Texas': 'greenyellow', 'Nebraska': 'lightcoral', 'Utah': 'gold', 'Oregon': 'springgreen', 'Florida': 'skyblue', 
        'New York': 'skyblue', 'Rhode Island': 'skyblue', 'Georgia': 'cyan', 'New Hampshire': 'cyan', 'North Carolina': 'cornflowerblue', 
        'New Jersey': 'mediumslateblue', 'Colorado': 'thistle', 'Maryland': 'thistle', 'Nevada': 'thistle', 'Tennessee': 'thistle', 
        'Hawaii': 'navajowhite', 'Indiana': 'navajowhite', 'Kentucky': 'navajowhite', 'Minnesota': 'navajowhite', 'Oklahoma': 'navajowhite', 
        'Pennsylvania': 'navajowhite', 'South Carolina': 'navajowhite', 'Columbia': 'aqua', 'Kansas': 'aqua', 'Missouri': 'aqua', 
        'Vermont': 'aqua', 'Virginia': 'aqua', 'Connecticut': 'pink', 'Iowa': 'pink', 'Louisiana': 'darkkhaki', 'Ohio': 'darkkhaki', 
        'Michigan': 'peru', 'South Dakota': 'peru', 'Arkansas': 'limegreen', 'Delaware': 'limegreen', 'Mississippi': 'limegreen', 
        'New Mexico': 'limegreen', 'North Dakota': 'limegreen', 'Wyoming': 'limegreen', 'Alaska': 'lightsteelblue', 'Maine': 'lightsteelblue', 
        'Alabama': 'sandybrown', 'Idaho': 'sandybrown', 'Montana': 'sandybrown', 'Puerto Rico': 'sandybrown', 'Virgin Islands': 'lawngreen', 
        'Guam': 'paleturquoise', 'West Virginia': 'darkseagreen', 'Northern Mariana Islands': 'royalblue'}

path = "states_all_0-229.p"
T = 230
threshold = 0.07

names = list(sorted(pos.keys()))
node_names = {k: k for k in names}
node_colors = [color[k] for k in sorted(color.keys())]
# print(node_colors)
# exit()
# print(len(names))
# exit()
node_size = 50
node_numbers = len(pos)


G=nx.DiGraph()

for n in names:
    G.add_node(n)
print(G.nodes())

fig = plt.gcf()

# tracker = []
# for i in range(T):
#     genie3 = pickle.load(open("genie_K_division/states_pickle/states_"+str(i)+"-W+"+str(i)+".p", "rb"))
#     tracker.append(genie3)
# pickle.dump(tracker, open("genie_K_division/states_all_0-229.p", "wb"))
# exit()

tracker = pickle.load(open(path, "rb"))
def animate(k):
    plt.clf()
    
    G=nx.DiGraph()
    for n in names:
        G.add_node(n)

    for i in names:
        for j in names:
            G.add_edge(i, j, weight=0, color="white")
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors)
    labels = nx.draw_networkx_labels(G, pos, labels=node_names, font_size=7, font_color='k')
    edge_list = tracker[k]
    
    count = 0
    for i in range(node_numbers):
        for j in range(node_numbers):
            if i != j and edge_list[i][j] > threshold:
                G.add_edge(names[i], names[j], weight=edge_list[i][j]*10, color="red")
                count +=1
    
    edges = G.edges()
    edges_weight_list = [G[u][v]['weight'] for u,v in edges]
    edges_color_list = [G[u][v]['color'] for u,v in edges]
    print("edge:", count)
    # print(len(edges_color_list))

    nx.draw_networkx_edges(G, pos, width = edges_weight_list, arrowstyle="->", edge_color=edges_color_list)

    # limits = plt.axis('off')  # turn of axis
    plt.xlabel("Day: "+ str(k+1) + " to "+str(k+61))

    return G

anim = FuncAnimation(fig, animate, frames = T, repeat = False, interval = 500)
plt.tight_layout()
anim.save('ani1.gif', dpi=300)
plt.show()

