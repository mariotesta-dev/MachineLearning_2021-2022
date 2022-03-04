#Lab 1 - Exercise 2
from math import sqrt
import sys
from typing import Reversible

f = open(sys.argv[1], "r")  #must be "transports.txt"
flag = sys.argv[2]          #flag "-b" (next parameter is a busID) or "-l" (next parameter is a lineID)
param = sys.argv[3]         #can be a busID or lineID

if(flag == '-b'):
    #print total distance traveled by the busID given in param

    distances = []

    for line in f:
        line = line.rstrip()
        entry = line.split()

        x = float(entry[2])
        y = float(entry[3])

        if(entry[0] == param):
            distances.append({'x': x, 'y': y})
    
    variations = []

    for index, el in enumerate(distances):
        if(index > 0):
            variations.append({'variation_x': distances[index]['x'] - distances[index-1]['x'], 'variation_y': distances[index]['y'] - distances[index-1]['y']})
    
    tot_var_x = 0
    tot_var_y = 0

    for i in variations:
        tot_var_x += i['variation_x']
        tot_var_y += i['variation_y']
    
    distance = sqrt(tot_var_x**2 + tot_var_y**2)
    print("%s - Total Distance: %.1f" % (param, distance))

    
elif(flag == '-l'):
    #print average speed of buses traveling on the lineID given in param
    lines = {}
    
    for line in f:
        line = line.rstrip()
        entry = line.split()

        if(entry[1] == param):
            bus_data = {'x': float(entry[2]), 'y': float(entry[3]), 'time': int(entry[4])}
            if(entry[0] in lines.keys()):
                lines[entry[0]].append(bus_data)
            else:
                lines[entry[0]] = [bus_data]
    
    speeds = []

    for i in lines.values():
        variations = []

        for index, el in enumerate(i):
            if(index > 0):
                variations.append({'variation_x': i[index]['x'] - i[index-1]['x'], 'variation_y': i[index]['y'] - i[index-1]['y'], 'variation_time': i[index]['time'] - i[index-1]['time']})
        
                tot_var_x = 0
                tot_var_y = 0
                tot_var_time = 0
                for k in variations:
                    tot_var_x += k['variation_x']
                    tot_var_y += k['variation_y']
                    tot_var_time += k['variation_time']
                
                print(variations)
        
                speed = sqrt(tot_var_x**2 + tot_var_y**2)/tot_var_time
                speeds.append(speed)

    print("%s - Avg Speed: %f" % (param, sum(speeds)/len(speeds)))


else:   
    print("Error: flag must be -b or -l")
    exit(0)

