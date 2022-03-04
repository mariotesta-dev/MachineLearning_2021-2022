#Lab. 1 - Exercise 3
import sys

f = open(sys.argv[1],'r')   #must be "people.txt"

months = {'01': 'January', '02': 'February', '03': 'March', '04': 'April', '05': 'May', '06': 'June', '07': 'July', '08':'August', '09':'September', '10': 'October', '11': 'November', '12':'December'}

cities_birth = {}
month_birth = {}

for line in f:
    line = line.rstrip()
    entry = line.split()

    city = entry[2]
    month = entry[3].split("/")[1]

    if(city in cities_birth.keys()):
        cities_birth[city] += 1
    else:
        cities_birth[city] = 1

    if(month in month_birth.keys()):
        month_birth[month] += 1
    else:
        month_birth[month] = 1   

print("Births per city:")
for i in cities_birth.items():
    print("%s: %d" % (i[0],i[1]))  

print("Births per month:")
for i in month_birth.items():
    print("%s: %d" % (months[i[0]],i[1]))       

print("Average number of births: %.2f" % (sum(cities_birth.values())/len(cities_birth)))
