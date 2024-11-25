import time
from search.algorithms import State
from search.map import Map
import getopt
import sys
import heapq

def main():
    """
    Function for testing your A* and Dijkstra's implementation. 
    Run it with a -help option to see the options available. 
    """
    optlist, _ = getopt.getopt(sys.argv[1:], 'h:m:r:', ['testinstances', 'plots', 'help'])

    plots = False
    for o, a in optlist:
        if o in ("-help"):
            print("Examples of Usage:")
            print("Solve set of test instances and generate plots: main.py --plots")
            exit()
        elif o in ("--plots"):
            plots = True

    test_instances = "test-instances/testinstances.txt"
    
    # Dijkstra's algorithm and A* should receive the following map object as input
    gridded_map = Map("dao-map/brc000d.map")
    # gridded_map = Map("dao-map/arena.map")

    
    nodes_expanded_dijkstra = []  
    nodes_expanded_astar = []

    time_dijkstra = []  
    time_astar = []

    start_states = []
    goal_states = []
    solution_costs = []
       
    file = open(test_instances, "r")
    for instance_string in file:
        list_instance = instance_string.split(",")
        start_states.append(State(int(list_instance[0]), int(list_instance[1])))
        goal_states.append(State(int(list_instance[2]), int(list_instance[3])))
        
        solution_costs.append(float(list_instance[4]))
    file.close()
        
    for i in range(0, len(start_states)):    
        start = start_states[i]
        goal = goal_states[i]
    
        time_start = time.time()
        # cost, expanded_diskstra = None, None # replace None, None with the call to your Dijkstra's implementation
        cost,  expanded_diskstra = dijkastra(start, goal, gridded_map)
        # cost,  expanded_diskstra, closed = dijkastra(start, goal, gridded_map)
        # Map.plot_map(gridded_map, closed, start, goal, "dmap{}".format(i))
        time_end = time.time()
        nodes_expanded_dijkstra.append(expanded_diskstra)
        time_dijkstra.append(time_end - time_start)

        if cost != solution_costs[i]:
            print("There is a mismatch in the solution cost found by Dijkstra and what was expected for the problem:")
            print("Start state: ", start)
            print("Goal state: ", goal)
            print("Solution cost encountered: ", cost)
            print("Solution cost expected: ", solution_costs[i])
            print()    

        start = start_states[i]
        goal = goal_states[i]
    
        time_start = time.time()
        cost, expanded_astar = astar(start, goal, gridded_map) # replace None, None with the call to your A* implementation
        # cost, expanded_astar, closed = astar(start, goal, gridded_map) # replace None, None with the call to your A* implementation
        # Map.plot_map(gridded_map, closed, start, goal, "amap{}".format(i))
        time_end = time.time()

        nodes_expanded_astar.append(expanded_astar)
        time_astar.append(time_end - time_start)

        if cost != solution_costs[i]:
            print("There is a mismatch in the solution cost found by A* and what was expected for the problem:")
            print("Start state: ", start)
            print("Goal state: ", goal)
            print("Solution cost encountered: ", cost)
            print("Solution cost expected: ", solution_costs[i])
            print()
            # print(solution_costs[i] - cost)

    if plots:
        from search.plot_results import PlotResults
        plotter = PlotResults()
        plotter.plot_results(nodes_expanded_astar, nodes_expanded_dijkstra, "Nodes Expanded (A*)", "Nodes Expanded (Dijkstra)", "nodes_expanded")
        plotter.plot_results(time_astar, time_dijkstra, "Running Time (A*)", "Running Time (Dijkstra)", "running_time")

def dijkastra(start, goal, gridded_map):
    open_list = []
    closed_list = {}
    expanded_diskstra = 0

    heapq.heappush(open_list, start)
    closed_list[start.state_hash()] = start

    while len(open_list) != 0:
        n = heapq.heappop(open_list)
        expanded_diskstra += 1

        if State.__eq__(n,goal):
            goal = n
            return (State.get_g(goal), expanded_diskstra)
        
        children = gridded_map.successors(n)

        for child in children:
            hash_value = child.state_hash()
            # x = max(State.get_x(n),State.get_x(child)) - min(State.get_x(n),State.get_x(child))
            # y = max(State.get_y(n),State.get_y(child)) - min(State.get_y(n),State.get_y(child))
            
            State.set_cost(child,State.get_g(child))
            # State.set_cost(child, Map.cost(gridded_map, child.get_x(), child.get_y()))
        
            if hash_value not in closed_list:
                # new_cost = State.get_g(n) + State.get_cost(child)
                # State.set_cost(child, new_cost)
                heapq.heappush(open_list, child)
                closed_list[hash_value] = child
                # parent[child.state_hash()] = n
            
            old_cost = State.get_g(closed_list[hash_value])
            newcost = State.get_cost(child)
            # newcost = State.get_g(n) + State.get_cost(child)

            if hash_value in closed_list and newcost < old_cost:
                # State.set_cost(child, newcost)
                heapq.heappush(open_list, child)
                closed_list[hash_value] = child
                # parent[child.state_hash()] = n

        heapq.heapify(open_list)

    # expanded_diskstra = 0
    return (-1, expanded_diskstra)

def astar(start, goal, gridded_map):
    open_list = []
    closed_list = {}
    expanded_astar = 0

    heapq.heappush(open_list, start)
    closed_list[start.state_hash()] = start

    h = heuristic(start, goal)
    # State.set_cost(start, h)

    while len(open_list) != 0:
        n = heapq.heappop(open_list)
        expanded_astar += 1
        if State.__eq__(n,goal):
            goal = n
            return (State.get_cost(goal), expanded_astar)
        
        children = gridded_map.successors(n)

        for child in children:
            # last_h = h
            hash_value = child.state_hash()
            # x = max(State.get_x(n),State.get_x(child)) - min(State.get_x(n),State.get_x(child))
            # y = max(State.get_y(n),State.get_y(child)) - min(State.get_y(n),State.get_y(child))
            h = heuristic(child, goal)
            
            # State.set_cost(child, Map.cost(gridded_map, child.get_x(), child.get_y()))
        
            new_cost = State.get_g(child) + h
            # new_cost = State.get_g(n) + h
            # new_cost = State.get_g(n) + State.get_cost(child) + h - last_h
            State.set_cost(child, new_cost)
            
            if hash_value not in closed_list:

                heapq.heappush(open_list, child)
                closed_list[hash_value] = child
            
            old_cost = State.get_cost(closed_list[hash_value])
            newcost = State.get_cost(child)
            # newcost = State.get_g(n) + State.get_cost(child) + h - last_h

            if hash_value in closed_list and newcost < old_cost:
                # State.set_cost(child, newcost)
                heapq.heappush(open_list, child)
                closed_list[hash_value] = child

        heapq.heapify(open_list)

    # expanded_astar = 0
    return (-1, expanded_astar)

def heuristic(evaluated_node, goal):
    deltax = abs(State.get_x(evaluated_node) - State.get_x(goal))
    deltay = abs(State.get_y(evaluated_node) - State.get_y(goal))

    h = 1.5 * min(deltax, deltay) + abs(deltax-deltay)

    return h
       
    # heapq.heappush(my_heap, h)
    # h = heapq.heappop(my_heap)
    # heapq.heapify(my_heap)


if __name__ == "__main__":
    main()