import csv
import math
import itertools
import random

def load_data(filepath):
    poke_list = []
    int_keys = ['#', 'Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
    with open(filepath, "r") as csvfile: # open and iterate through file
        for row in itertools.islice(csv.DictReader(csvfile), 20):
            row.pop('Generation') # remove unwanted data
            row.pop('Legendary')
            for key in row: # convert number strings to ints
                if key in int_keys:
                    row[key] = int(row.get(key))
            poke_list.append(row)
    return poke_list


def calculate_x_y(stats):
    # x calc: offensive strength x = Attack + Sp. Atk + Speed
    # y cal: defensive strength y = Defense + Sp. Def + HP
    off_keys = ['Attack', 'Sp. Atk', 'Speed']
    def_keys = ['Defense', 'Sp. Def', 'HP']
    x = 0
    y = 0
    for key in stats:
        if key in off_keys:
            x = x + stats.get(key)
        elif key in def_keys:
            y = y + stats.get(key)
    return (x, y)

def hac(dataset):
    for row in dataset: # check dataset that data is valid
        for col in row: # check individual index
            if not math.isfinite(dataset[row][col]): # remove values not NaN or inf
                dataset.remove(row) 
        # NOTE: did not finish,  

        #if not isinstance(dataset[0], (int)):
        #    dataset.remove(stats)
        #elif not isinstance(dataset[1], (int)):
        #    dataset.remove(stats)
    print("Not implemented")

    return

def random_x_y(m):
    rand_data = []
    for i in range(m): # make m random tuples
        x = random.randint(1, 359)
        y = random.randint(1, 359)
        rand_data.append((x, y))
    return rand_data

def imshow_hac(dataset):
    print("Not implemented")
    return