import heapq
import numpy as np

# returns the Manhattan distance summed for each tile
def heuristics(state):
    # Manhattan distance = abs(x_1-x_2) + abs(y_1-y_2)
    # loop through the 8 numbers (1-8)
    #print(state)
    points = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)] # coordinates of board positions
    goal = [1, 2, 3, 4, 5, 6, 7, 8, 0] # goal state to reach
    i = 0
    d = 0 # singular distance
    distance = 0 # summed distances
    temp = state.copy()
    for i in range(9):
        # check the state at each position, retrieve it's points
        val = temp[i] # store the number that's being checked
        #if val != 0:
        goal_point = points[val-1] # -1 for indexing, save location of point
        point = points[i] # 
        #print(val, goal_point, point)
        if val != 0:
            d = abs(goal_point[0] - point[0]) + abs(goal_point[1] - point[1])
            #print(d)
            distance += d

    return distance
# Actual function to return the successors
def successors(state):
    # state given as a single list of ints
    # Rough outline:

    # variable instantiation
    s_list = []
    corner = False
    moab = False
    center = False

    # locate where the 0 is (corner, middle of a boundary, or center)
    # corner is at positions i = 0, 2, 6, 8
    # moab is at i = 1, 3, 5, 7
    # center is at i = 4
    i = 0
    for i in range(9): # 9 positions on the board to check for empty slot
        if state[i] == 0:
            if i in (0, 2, 6, 8):
                corner = True
                #print("corner")
                break
            elif i in (1, 3, 5, 7):
                moab = True
                #print("moab")
                break
            else: 
                center = True
                #print(center)
                break
    # depending on location, make successor options
    
    temp = []
    if corner == True: # 2 successors
        if i in (0, 2):
            temp = state.copy() # temp will be added to s_list once edited
            temp[i] = temp[1] # swap with top center tile
            temp[1] = 0
            s_list.append(temp)
            temp = state.copy() # reset temp
            temp[i] = temp[i+3] # swap with tile below
            temp[i+3] = 0
            s_list.append(temp)
        if i in (6, 8):
            temp = state.copy() # temp will be added to s_list once edited
            temp[i] = temp[7] # swap with tile to the left
            temp[7] = 0
            s_list.append(temp)
            temp = state.copy() # reset temp
            temp[i] = temp[i-3] # swap with tile above
            temp[i-3] = 0
            s_list.append(temp)
    elif moab == True: # 3 successors
        if i in (1, 7): # swap left and right, and above/below
            temp = state.copy() # temp will be added to s_list once edited
            temp[i] = temp[i-1] # swap with tile to left
            temp[i-1] = 0
            s_list.append(temp)
            temp = state.copy() # reset temp
            temp[i] = temp[i+1] # swap with tile to right
            temp[i+1] = 0
            s_list.append(temp)
            temp = state.copy()
            if i == 1: # swap with tile below
                temp[i] = temp[i+3]
                temp[i+3] = 0
                s_list.append(temp)
            else: # i == 7, swap with tile above
                temp[i] = temp[i-3]
                temp[i-3] = 0
                s_list.append(temp)
        if i in (3, 5): # swap above and below, and right/left
            temp = state.copy()
            temp[i] = temp[i-3] # swap above
            temp[i-3] = 0
            s_list.append(temp)
            temp = state.copy()
            temp[i] = temp[i+3]
            temp[i+3] = 0
            s_list.append(temp)
            if i == 3: # swap w/ right
                temp = state.copy()
                temp[i] = temp[i+1]
                temp[i+1] = 0
                s_list.append(temp)
            else: # i == 5, swap w/ left
                temp = state.copy()
                temp[i] = temp[i-1]
                temp[i-1] = 0
                s_list.append(temp)
    elif center == True: # 4 successors
        j = 0
        for j in (-1, 1, 3, -3):
            temp = state.copy()
            temp[i] = temp[i+j]
            temp[i+j] = 0
            s_list.append(temp)

    # return the list of successors
    return sorted(s_list)

# given a state of the puzzle, represented as a single list of integers with a 0 in the empty space, 
# print to the console all of the possible successor states
# 3 possible successor combos: 2 (empty @ corner), 3 (empty at middle of a boundary), or 4 (empty @ center of a boundary)
def print_succ(state):
    # state given as a single list of ints

    #Rough outline for procedure:
    # use helper function to get the successors
    # use this as a wrapper to print the successors
    s_list = successors(state)
    h = 0
    for succ in s_list:
        h = heuristics(succ)
        print(succ, 'h={:d}'.format(h))
    #print(sorted(s_list))
    return

def print_path(pq, closed):
    p = []
    p.append(closed[-1]) # backwards traversal
    n = closed[-1][2][2]
    # begin traversal
    for i in range(len(closed)):
        if(closed[len(closed)-i-1][2][3] == n):
            p.append(closed[len(closed) - i - 1])
            n = closed[len(closed)-i-1][2][2]
    # print out result
    for i in range(len(p)): #string concatenation
        print(str(p[len(p)-i-1][1])+" h="+str(p[len(p)-i-1][2][1])+" moves: "+str(p[len(p)-i-1][2][0]))

    # A* algorithm
    # 1. Put the start node S on the pq, called OPEN
    # 2. If OPEN is empty, exit w/ failure
    # 3. Remove from OPEN and place on CLOSED a node n for which f(n) is the minimum (f(n) = g(n) + h(n))
    # 4. If n is a goal node, exit (trace back from n to S)
    # 5. Expand n, generate all successors and attach pointers back to n. for each successor n' of n
    #       5.1 If n' is not already on OPEN or CLOSED estimate h(n'), g(n') = g(n) + c(n,n'), f(n')=g(n')+h(n')
    # 6. goto 2
    # format to push onto heap (g+h, state, (g, h, parent_index))
    # parent_index is used for indexing into closed
    # g is moves, 
    # #########  g != parent_index ###########
    # step 1
def solve(state):
    goal = [1, 2, 3, 4, 5, 6, 7, 8, 0] # goal state to reach
    temp = state[:]
    g = 0 # g metric
    h = 0 # heuristic measure
    pq = [] # priority queue
    closed = [] # set of visited states + state info
    visited = [] # set of visited states
    max_len = 0 # used to track diff. b/w closed and pq
    h = heuristics(temp)
    heapq.heapify(pq) # make the heap into a list
    heapq.heappush(pq, (g+h, temp, (g, h, -1, max_len))) # push the root into the heap
    max_len += 1 # init max_len
    while len(pq) > 0:
        # step 3, pop the lowest priority
        curr = heapq.heappop(pq)
        closed.append(curr)
        if heuristics(curr[1]) == 0:
            return print_path(pq, closed)
        #print(successors(curr[1])) 
        s_list = successors(curr[1]) 
        for s in s_list:# get successors from state of curr
            if not(s in visited):
                h_s = heuristics(s) # calculate new h and g values
                g_s = curr[2][0] + 1
                heapq.heappush(pq, (g_s+h_s, s, (g_s, h_s, curr[2][3], max_len)))
                max_len += 1  
                visited.append(s)                
    return
 

    
    



    
     