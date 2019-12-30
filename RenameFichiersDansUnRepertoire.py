#-*- coding: utf-8 -*-

import os
import time
import sys

def main(repertoire,index): 
    i = int(index)
    
    for filename in os.listdir(repertoire):
        nomImage ="Image__" + str(i) + ".jpg"
        print('On renomme l\'image : ' + filename)
        #On va dans le répertoire
        old_file = os.path.join(repertoire, filename)
        new_file = os.path.join(repertoire, nomImage)
        #On renomme le fichier
        os.rename(old_file, new_file)
        print(filename +  ': Opération Réussie !')
        i += 1

# Driver Code 
if __name__ == '__main__':
    if len(sys.argv) <= 2:
        print('USAGE: repertoire index')
    else:
        t1 = time.time()
        main(sys.argv[1],sys.argv[2])
        print('Temps de Traitement : %d ms'%((time.time()-t1)*1000))
