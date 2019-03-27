import gerryfair
import matplotlib.pyplot as plt
'''
Experiments to run:
- Pareto curve for several datasets (linear thresholds)
- FP vs FN
- Neural nets
'''

def multiple_pareto():
    communities_dataset = "./dataset/communities.csv"
    communities_attributes = "./dataset/communities_protected.csv"
    lawschool_dataset = "./dataset/lawschool.csv"
    lawschool_attributes = "./dataset/lawschool_protected.csv"
    adult_dataset = "./dataset/student-mat.csv"
    adult_attributes = "./dataset/student_protected.csv"

    C = 15
    printflag = True
    gamma = .01
    fair_model = gerryfair.model.Model(C=C, printflag=printflag, gamma=gamma, fairness_def='FP')
    max_iters = 200
    gamma_list = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
    fair_model.set_options(max_iters=max_iters)


    centered = True
    train_size = 1000


    """
    X_train = X.iloc[:train_size]
    X_prime_train = X_prime.iloc[:train_size]
    y_train = y.iloc[:train_size]
    """

    # Train Set (Communities - size=1000, iters=200)
    X, X_prime, y = gerryfair.clean.clean_dataset(communities_dataset, communities_attributes, centered)

    # Train the model
    X_train = X.iloc[:train_size]
    X_prime_train = X_prime.iloc[:train_size]
    y_train = y.iloc[:train_size]
    communities_all_errors, communities_all_violations = fair_model.pareto(X_train, X_prime_train, y_train, gamma_list)


    train_size = 100

    # Train Set (Communities)
    X, X_prime, y = gerryfair.clean.clean_dataset(lawschool_dataset, lawschool_attributes, centered)
    
    # Train the model
    X_train = X.iloc[:train_size]
    X_prime_train = X_prime.iloc[:train_size]
    y_train = y.iloc[:train_size]
    fair_model.set_options(max_iters=500)
    lawschool_all_errors, lawschool_all_violations = fair_model.pareto(X_train, X_prime_train, y_train, gamma_list)


    # Train Set (Communities)
    #X, X_prime, y = gerryfair.clean.clean_dataset(adult_dataset, adult_attributes, centered)

    # Train the model
    #adult_all_errors, adult_all_violations = fair_model.pareto(X_train, X_prime_train, y_train, gamma_list)


    plt.plot(communities_all_errors, communities_all_violations, label='communities')
    plt.plot(lawschool_all_errors, lawschool_all_violations, label='law school')
    #plt.plot(adult_all_errors, adult_all_violations, label='adult')
    plt.xlabel('error')
    plt.ylabel('unfairness')
    plt.legend()
    plt.title('error vs. unfairness (iterations = {})'.format(max_iters))
    plt.show()


    print("done")

multiple_pareto()



