## read sentences and extract only line which contain the keywords
import pandas as pd
import re
# open file
prison_keyword = open('prison keyword.txt', 'r', encoding = 'utf-8').readlines()
texts = open('sent_token.txt', 'r', encoding = 'utf-8').readlines()
# define function to read file and remove next line symbol
def read_file(file):
    texts = []
    for word in file:
        text = word.rstrip('\n')
        texts.append(text)

    return texts


# save to variable        
key = read_file(prison_keyword)
corpus = read_file(texts)

# open file to write line which contain keywords
file = open('keyline_prison.txt', 'w', encoding = 'utf-8') 
def write_file(file, keyword, corpus):    
    #keyline = []      
    for line in corpus:
        line = line.lower()
        for key in keyword:
            result = re.search(r"(\s)+" + key + r"\s", line)
            if result != None:
                #keypair = [key, line]
                #keyline.append(keypair) 
                file.write(line)                                
                #file.write("\n") 
                #file.write(":".join(map(str,keypair)))
                break    # to prevent copy the same line from diferent keyword to lower file size                
            else:
                pass                                            
          
    #return(keyline)

#print(write_file(file,key,corpus))
#output = write_file(file,key,corpus)
# create DataFrame using data 
#df = pd.DataFrame(output, columns =['Key', 'Line']) 
# save to csv file    
#df.to_csv("df_keyline_output.csv")
write_file(file,key,corpus)