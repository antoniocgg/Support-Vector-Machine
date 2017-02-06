import re
import numpy as np

# parameter is the path to the text file i.e. ("a7a.train")
# encode classes: '-1' as 0 and '+1' as 1
def parse(textfile):
    tf = open(textfile, "r")
    content = tf.readlines()
    new_content = []
    bias = 0
    for index, line in enumerate(content):
        line = line.split(" ")
        truth = -1 if line[0] == '-1' else 1     # convert true class
        line = line[1:-1]
        # ['2:1', '6:1', '18:1', '20:1', '37:1', '42:1', '53:1',...]
        # 123 is the longest vector in the data
        new_line = [0]*124
        for group in line:
            group = re.sub('[^0-9]','', group.rpartition(':')[0])
            new_line[int(group)] = 1
        new_line.insert(0, truth)
        new_content.append(np.array(new_line))
    return new_content

# this is how it would be run
#a = parse("a7a.test")
