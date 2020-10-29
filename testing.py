import re

#txt = "The rain in Spain"
#y=[]
#x = re.search(r"\brain\$", txt)
#print(x)

# this will find line which not totally match 
def write_file2(file, keyword, corpus):    
    keyline = []    
    for line in corpus:
        line = line.lower()
        for key in keyword:
            if key in line:
                keypair = [key, line]
                keyline.append(keypair)                
                file.write("\n") 
                file.write(":".join(map(str,keypair)))                
          
            else:
                pass

    return(keyline)


from gensim.test.utils import common_corpus, common_dictionary
from gensim.models import HdpModel

hdp = HdpModel(common_corpus, common_dictionary)
topic_info = hdp.print_topics(num_topics=20, num_words=10)
#print(topic_info)
print(common_dictionary)