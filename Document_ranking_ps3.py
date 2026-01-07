import pandas as pd
import numpy as np

class Documents:
    def __init__(self):
        self.p = np.random.uniform(low=0, high = 1, size =1)[0]

    def didClick(self):
        return np.random.binomial(n=1, p = self.p, size=1)[0] #For bernoulli
    
def computeRank(documents, n, k, eps, delta, x = -1):
    document_chosen = []
    ranking = []

    for rank in range(k):
        max_count = -1
        max_doc = -1
        if x == -1:
            num_draws = round(2 * (k - rank) ** 2 / eps ** 2 * np.log(2 * (k - rank) / delta))
        else:
            num_draws = x
        for document in range(n):
            count = 0
            if document in document_chosen:
                continue
            for draw in range(num_draws):
                count += documents[document].didClick()
            if count > max_count:
                max_count = count
                max_doc = document
        ranking.append(max_doc + 1)
        document_chosen.append(max_doc)

    return ranking


    
def main(): 
    n = 100 # Number of documents
    k = 10 #No of ranks
    eps = 0.2 #epsilon
    delta = 0.7

    documents = [Documents() for i in range(n)]
    ranking = computeRank(documents, n, k, eps, delta, x = 1000)
    print("Ranking (x = 1000)=> ", ranking)
    ranking = computeRank(documents, n, k, eps, delta)
    print("Ranking (x = (2k^2 /ϵ^2 )log(2k/δ))=> ", ranking)

main()