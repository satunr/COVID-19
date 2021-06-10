def boundary_of_each_grid(B, size_of_each_grid):

    bG = {}
    grid_Id = 0

    # Proceed row-wise
    number_of_cols = int(B[1][0] / size_of_each_grid[1][0])
    number_of_rows = int(B[1][1] / size_of_each_grid[1][1])

    # print (number_of_rows, number_of_cols)
    for r in range(number_of_rows):

        for c in range(number_of_cols):

            bR = (r * size_of_each_grid[1][0], c * size_of_each_grid[1][0])
            bC = ((r + 1) * size_of_each_grid[1][1], (c + 1) * size_of_each_grid[1][1])

            bG[grid_Id] = [bR, bC]
            grid_Id += 1

    return bG


def locate(P, bG):
    p_dict = {}
    for p in P:
        for k in bG.keys():
            x0 = bG[k][0][0]
            y0 = bG[k][0][1]
            x1 = bG[k][1][0]
            y1 = bG[k][1][1]

            if x0 <= p[0] < x1 and y0 <= p[1] < y1:
                p_dict[P.index(p)] = k
                break
    return p_dict


# B = [(0, 0), (200, 200)]
# size_of_each_grid = [(0, 0), (100, 100)]

# (x0, y0) and (x1, y1) for deployment region
B = [(0, 0), (2, 2)]

# (x0, y0) and (x1, y1) for each grid size
size_of_each_grid = [(0, 0), (1, 1)]

# Create grids (dictionary of grid ID and boundary (x0, y0) and (x1, y1)
bG = boundary_of_each_grid(B, size_of_each_grid)
print (bG)

# Locate points on grid (dictionary of point ID and grid ID)
P = [(1.4, 1.5), (0.5, 1.4)]
p_dict = locate(P, bG)
print (p_dict)
