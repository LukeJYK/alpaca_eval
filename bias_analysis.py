import json 
#read your files
with open('./test2.json', 'r') as file1:
    data1 = json.load(file1)
    
with open('./test3.json', 'r') as file2:
    data2 = json.load(file2)
    
    
#info for test1.json: 1-normal, 2-brief+, 3-brief
#info for test2.json: 1-brief+, 2-brief, 3-normal
verify1 = []
verify2 =[]
for sample1 in data1:
    verify1.append(sample1['raw_completion'])
for sample2 in data2:
    verify2.append(sample2['raw_completion'])
count = 0 
for i in range(len(verify1)):
    if (verify1[i] == 'Output (1)' and verify2[i] == 'Output (3)') or (verify1[i] == 'Output (2)' and verify2[i] == 'Output (1)') or (verify1[i] == 'Output (3)' and verify2[i] == 'Output (2)'):
        count = count + 1
print("consistency:",count/len(verify1))
        