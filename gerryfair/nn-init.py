# import the data
dataset = "./dataset/communities.csv"
attributes = "./dataset/communities_protected.csv"
centered = True
X, X_prime, y = gerryfair.clean.clean_dataset(dataset, attributes, centered)