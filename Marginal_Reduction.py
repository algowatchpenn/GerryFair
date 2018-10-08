from Reg_Oracle_Fict import *
from Audit import *
import sys

# Helper Functions
# -----------------------

#Parse arguments for user input
def setup():
    parser = argparse.ArgumentParser(description='Marginal_Reduction.py input parser')
    parser.add_argument('-d', '--dataset', type=str, help='name of the dataset (communities, lawschool, adult, student), (Required)')
    parser.add_argument('-i', '--iters', type=int, default=10, required=False, help='number of iterations to terminate after, (Default = 10)')

    args = parser.parse_args()
    return [args.dataset, args.iters]

# calculate the next lambda via gradient descent
# for a group a compute P(preds = 1|A = a, Y = 0) = 1/n # y = 0, A = a,
# preds = 1 / p0a
def compute_fp(a, probs_0, preds, X_prime, X, y):
    p_0a = probs_0[a]
    y0_a = [y[i] == 0 and preds[i] == 1 and X_prime.iloc[i, a[0]] == a[1]
            for i in range(len(y))]
    p_y0_a = np.mean(y0_a)
    if p_0a == 0:
        return 'flag'
    return p_y0_a / p_0a


def next_lambda(lambda_0, iteration, probs_0, h, X_prime, X, y):
    A = lambda_0.keys()
    pred = h.predict(X)
    fp = {a: compute_fp(a, probs_0, pred, X_prime, X, y) for a in A}
    fp_av = np.mean({a: fp[a] for a in A if fp[a] != 'flag'}.values())
    for a in A:
        if fp[a] == 'flag':
            fp[a] = fp_av
    # update average vector with gradient. note gradient step is + eta*(grad)
    # since we are maximizing over lambda
    lambda_0 = {a: lambda_0[a] *
                (iteration -
                 1) /
                iteration +
                1.0 /
                iteration *
                (lambda_0[a] +
                 1.0 /
                 np.sqrt(iteration) *
                 (fp[a] -
                  fp_av)) for a in A}
    return lambda_0, [np.abs(fp_av - t) for t in fp.values()]


# lambda_0, lambda_1 : dictionaries whose keys are a in A, and values are lambda_ia
# probs_0, probs_1: dictionaries whose keys are a in A and values are P[A=a,y= y]
# return: cost arrays c_0 c_1
def update_costs(lambda_0, probs_0, X_prime, y):
    num_sens = X_prime.shape[1]
    cost_0 = [float(y[i] != 0) for i in range(len(y))]
    cost_1_ind = [y[i] == 0 for i in range(len(y))]
    mu_0 = np.mean(lambda_0.values())
    weights_1 = [0.0] * len(y)
    for k in range(len(y)):
        for j in range(num_sens):
            a_kj = (j, X_prime.iloc[k, j])
            if a_kj in lambda_0.keys():
                y_k = y[k]
                if y_k == 0:
                    weights_1[k] += (lambda_0[a_kj] - mu_0) / probs_0[a_kj]
                else:
                    weights_1[k] = 0
    cost_1 = np.add(cost_1_ind, weights_1)
    return cost_0, cost_1


def fit_weighted(cost_0, cost_1, x):
    reg0 = linear_model.LinearRegression()
    reg0.fit(x, cost_0)
    reg1 = linear_model.LinearRegression()
    reg1.fit(x, cost_1)
    primal_model = Reg_Oracle_Class.RegOracle(reg0, reg1)
    return primal_model


def evaluate_classifier(h, X, y):
    n = len(y)
    preds = h.predict(X)
    acc = np.mean([preds[i] == y[i] for i in range(n)])
    return acc


def marginal_preds(X, X_prime, X_prime_cts, y, max_iters, printflag=False):
    # initialize parameters
    iteration = 1
    hypothesis = []
    max_disp = []
    num_sens = X_prime.shape[1]
    n = X.shape[0]

    # set up lambda, p dictionaries
    immutable_keys = [(s, t) for s in range(num_sens) for t in [0, 1]]
    # get unique keys
    immutable_keys = set(immutable_keys)
    # convert back to list (iterable)
    immutable_keys = list(immutable_keys)

    # set values of prob_0
    prob_0 = {key: 1.0 / n * np.sum([X_prime.iloc[i, key[0]] == key[1]
                                     and y[i] == 0 for i in range(len(y))]) for key in immutable_keys}

    # only focus on groups with non-trivial mass (speed up convergence)
    # alpha = 0 includes all groups
    alpha = 0
    key_alpha = [key for key in prob_0.keys() if prob_0[key] > alpha]

    lambda_0 = {el: 0 for el in key_alpha}
    prob_0 = {key: prob_0[key] for key in key_alpha}

    q = [1.0 / n] * n
    cost_0 = [0 if s == 0 else q[r] for r, s in enumerate(y)]
    cost_1 = [0 if s == 1 else q[r] for r, s in enumerate(y)]

    h = fit_weighted(cost_0, cost_1, X)

    # print out the best classifier error
    h_dec = h.predict(X)
    best_error = np.mean([h_dec[i] != y[i] for i in range(n)])
    if printflag:
        print('best classiifer error: {}'.format(best_error))

    while iteration < max_iters:
        # primal player best responds via cost-sensitive learning oracle
        print(iteration)
        cost_0, cost_1 = update_costs(lambda_0, prob_0, X_prime, y)
        h = fit_weighted(cost_0, cost_1, X)
        # update lambda via gradient descent
        lambda_0, unfairness = next_lambda(
            lambda_0, iteration, prob_0, h, X_prime, X, y)
        hypothesis.append(h)
        max_disp.append(np.max(unfairness))
        acc = evaluate_classifier(h, X, y)
        if printflag:
            new_predictions = h.predict(X)
            audit(new_predictions, X, X_prime_cts, y)
            print('Marginal FP disparity in each group: {}'.format(unfairness))
            print('Max Marginal FP disparity: {}'.format(np.max(unfairness)))
            sys.stdout.flush()
        iteration += 1
    # print the decisions of the final classifier
    if printflag:
        print('decisions on the dataset of the final fair classifier: {}'.format(h.predict(X)))
    return h.predict(X)


if __name__ == "__main__":
    random.seed(1)
    # get command line arguments
    dataset, max_iters = setup() #sys.argv[1:]
    # dataset = str(dataset)
    # max_iters = float(max_iters)
    # Data Cleaning and Import
    f_name = 'clean_{}'.format(dataset)
    clean_the_dataset = getattr(clean_data, f_name)
    X, X_prime, y = clean_the_dataset()
    X, X_prime_cts, y = clean_the_dataset()
    n = X.shape[0]
    # threshold sensitive features by average value
    sens_means = np.mean(X_prime)
    for col in X_prime.columns:
        X.loc[(X[col] > sens_means[col]), col] = 1
        X_prime.loc[(X_prime[col] > sens_means[col]), col] = 1
        X.loc[(X[col] <= sens_means[col]), col] = 0
        X_prime.loc[(X_prime[col] <= sens_means[col]), col] = 0
    marginal_preds(X, X_prime, X_prime_cts, y, max_iters=max_iters, printflag=True)


