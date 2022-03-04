#Lab. 1 - Exercise 1
import sys

f = open(sys.argv[1], "r")  #must be "scores.txt"

athletes = []
countries = {}

for line in f:
    line = line.rstrip()        #removes \n at the end
    entry = line.split(" ")     #split by space character

    #final_score = sum(sorted([float(i) for i in entry[3:]])[1:-1])

    #step by step
    scores = [float(i) for i in entry[3:]]  #create a list with the 5 scores as floating numbers
    scores_sorted = sorted(scores)[1:-1]     #sort the list and remove first and last element
    final_score = sum(scores_sorted)        #sum all remaining scores of that athlete
    
    #create object with all useful data
    obj = {'name': entry[0]+" "+entry[1], 'nationality': entry[2], 'score': final_score} 

    #add it to athletes list
    athletes.append(obj)

    #add it to a dictionary of (nationality, score)
    if(obj['nationality'] in countries.keys()):
        countries[obj['nationality']] += obj['score']
    else:
        countries[obj['nationality']] = obj['score']

#get only the first 3 athletes based on their score
best3_athletes = sorted(athletes, key=lambda tup: tup['score'], reverse=True)[:3]

#get only the first 3 countries based on the sum of the scores
best_country = sorted(countries.items(), key=lambda tup: tup[1], reverse=True)[0]

print("final ranking:")
for index,athlete in enumerate(best3_athletes):
    print("%d: %s -- Score: %.1f" % (index+1, athlete['name'], athlete['score']))

print("\nBest Country:")
print("%s  Total score: %.1f" % (best_country[0], best_country[1]))


                

        
    
    


