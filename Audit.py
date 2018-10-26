# TODO: Rewrite using marginal_fair
from Reg_Oracle_Fict import *
from Marginal_Reduction import *
from sklearn import svm
from sklearn import neighbors
#import sys



# For usage information, see README.md or use: python Audit.py -h

# Helper Functions

#Parse arguments for user input
def setup():
    parser = argparse.ArgumentParser(description='Audit.py input parser')
    parser.add_argument('-d', '--dataset', type=str, help='name of the dataset (communities, lawschool, adult, student, all), (Required)')
    parser.add_argument('-a', '--attributes', type=str,
                        help='name of the file representing which attributes are protected (unprotected = 0, protected = 1, label = 2) (Required)')
    parser.add_argument('-i', '--iters', type=int, default=10, required=False, help='number of iterations to terminate after, (Default = 10)')

    args = parser.parse_args()
    return [args.dataset, args.attributes, args.iters]

def get_fp(preds, y):
    """Return the fp rate of preds wrt to true labels y."""
    # print('my FP ', np.mean(preds[y == 0]))

    # return np.mean(preds[y == 0])
    return np.mean([p for i,p in enumerate(preds) if y[i] == 0])


def audit(predictions, X, X_prime, y):
    """Takes in predictions on dataset (X, X',y) and prints gamma-unfairness,
    fp disparity, group size, group coefficients, and sensitive column names.
    """
    FP = get_fp(predictions, y)
    aud_group, gamma_unfair, fp_in_group, err_group, pos_neg = get_group(predictions, X_sens=X_prime, X=X, y_g=y, FP=FP)
    group_size = gamma_unfair/fp_in_group
    group_coefs = aud_group.b0.coef_ - aud_group.b1.coef_
    # get indices of maximum k coefficients
    k = np.min([len(group_coefs), 3])
    top_indices = np.abs(group_coefs).argsort()[-k:][::-1]
    # print('accuracy: {}'.format(1-np.mean(np.abs(np.subtract(predictions, y)))))
    # print('group size: {}, gamma-unfairness: {}, FP-disparity: {}'.format(group_size, gamma_unfair, fp_in_group))
    # print('subgroup_coefficients: {}'.format(group_coefs),)
    # print('sensitive attributes: {}'.format([c for c in X_prime.columns],))
    # print('sensitive attributes with the largest group coefficients: {}'.format(X_prime.columns[top_indices]))
    # print('coefficients of top sensitive attributes: {}'.format(group_coefs[top_indices]))
    return gamma_unfair





def get_group(A, X, X_sens, y_g, FP):
    """Given decisions on X, sensitive attributes, labels, and FP rate audit wrt
        to gamma unfairness. Return the group found, the gamma unfairness, fp disparity, and sign(fp disparity).
    """

    A_0 = [a for u, a in enumerate(A) if y_g[u] == 0]
    X_0 = pd.DataFrame([X_sens.iloc[u, :]
                        for u, s in enumerate(y_g) if s == 0])
    m = len(A_0)
    n = float(len(y_g))
    cost_0 = [0.0] * m
    cost_1 = -1.0 / n * (FP - A_0)
    reg0 = linear_model.LinearRegression()
    reg0.fit(X_0, cost_0)
    reg1 = linear_model.LinearRegression()
    reg1.fit(X_0, cost_1)
    func = Reg_Oracle_Class.RegOracle(reg0, reg1)
    group_members_0 = func.predict(X_0)
    err_group = np.mean([np.abs(group_members_0[i] - A_0[i])
                         for i in range(len(A_0))])
    # get the false positive rate in group
    if sum(group_members_0) == 0:
        fp_group_rate = 0
    else:
        fp_group_rate = np.mean([r for t, r in enumerate(A_0) if group_members_0[t] == 1])
    g_size_0 = np.sum(group_members_0) * 1.0 / n
    fp_disp = np.abs(fp_group_rate - FP)
    fp_disp_w = fp_disp * g_size_0

    # negation
    cost_0_neg = [0.0] * m
    cost_1_neg = -1.0 / n * (A_0-FP)
    reg0_neg = linear_model.LinearRegression()
    reg0_neg.fit(X_0, cost_0_neg)
    reg1_neg = linear_model.LinearRegression()
    reg1_neg.fit(X_0, cost_1_neg)
    func_neg = Reg_Oracle_Class.RegOracle(reg0_neg, reg1_neg)
    group_members_0_neg = func_neg.predict(X_0)
    err_group_neg = np.mean(
        [np.abs(group_members_0_neg[i] - A_0[i]) for i in range(len(A_0))])
    if sum(group_members_0_neg) == 0:
        fp_group_rate_neg = 0
    else:
        fp_group_rate_neg = np.mean([r for t, r in enumerate(A_0) if group_members_0[t] == 0])
    g_size_0_neg = np.sum(group_members_0_neg) * 1.0 / n
    fp_disp_neg = np.abs(fp_group_rate_neg - FP)
    fp_disp_w_neg = fp_disp_neg*g_size_0_neg

    # return group
    if fp_disp_w_neg > fp_disp_w:
        return [func_neg, fp_disp_w_neg, fp_disp_neg, err_group_neg, -1]
    else:
        return [func, fp_disp_w, fp_disp, err_group, 1]

if __name__ == "__main__":
    random.seed(1)
    ds = ['communities', 'lawschool', 'adult', 'student']
    dataset, attributes, max_iters = setup() #sys.argv[1:]
    # dataset = str(dataset)
    # max_iters = int(max_iters)

    if dataset == 'all':
        for dataset in ds:
            # Data Cleaning and Import
            # f_name = 'clean_{}'.format(dataset)
            # clean_the_dataset = getattr(clean_data, f_name)
            X, X_prime, y = clean_generic_data.clean_dataset(dataset, attributes)

            # print out the invoked parameters
            num_sens = X_prime.shape[1]
            print('Invoked Parameters: number of sensitive attributes = {}, dataset = {}'.format(num_sens, dataset))

            # logistic regression
            model = linear_model.LogisticRegression()
            model.fit(X, y)
            yhat = list(model.predict(X))
            print('logistic regression audit:')
            audit(predictions=yhat, X=X, X_prime=X_prime, y=y)

            # shallow neural network
            # model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 2), random_state=1)
            # model.fit(X,y)
            # yhat = list(model.predict(X))
            # print('multilayer perceptron (3, 2) audit:')
            # audit(predictions=yhat, X=X, X_prime=X_prime, y=y)

            # support vector machine
            model = svm.SVC()
            model.fit(X, y)
            yhat = list(model.predict(X))
            print('SVM audit:')
            audit(predictions=yhat, X=X, X_prime=X_prime, y=y)

            # nearest neighbor
            model = neighbors.KNeighborsClassifier(3)
            model.fit(X, y)
            yhat = list(model.predict(X))
            print('nearest neighbors audit:')
            audit(predictions=yhat, X=X, X_prime=X_prime, y=y)

            # Marginal reduction with Reg Oracle
            X, X_prime_cts, y = clean_the_dataset()
            n = X.shape[0]
            # threshold sensitive features by average value
            sens_means = np.mean(X_prime)
            for col in X_prime.columns:
                X.loc[(X[col] > sens_means[col]), col] = 1
                X_prime.loc[(X_prime[col] > sens_means[col]), col] = 1
                X.loc[(X[col] <= sens_means[col]), col] = 0
                X_prime.loc[(X_prime[col] <= sens_means[col]), col] = 0
            yhat = MSR_preds(X, X_prime, X_prime_cts, y, max_iters, False)
            audit(yhat, X, X_prime, y)
    else:
        # Data Cleaning and Import
        # f_name = 'clean_{}'.format(dataset)
        # clean_the_dataset = getattr(clean_data, f_name)
        X, X_prime, y = clean_generic_data.clean_dataset(dataset, attributes)

        # print out the invoked parameters
        num_sens = X_prime.shape[1]
        print('Invoked Parameters: number of sensitive attributes = {}, dataset = {}'.format(num_sens, dataset))

        # logistic regression
        model = linear_model.LogisticRegression()
        model.fit(X, y)
        yhat = list(model.predict(X))
        print('logistic regression audit:')
        audit(predictions=yhat, X=X, X_prime=X_prime, y=y)

        # shallow neural network
        # model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(3, 2), random_state=1)
        # model.fit(X,y)
        # yhat = list(model.predict(X))
        # print('multilayer perceptron (3, 2) audit:')
        # audit(predictions=yhat, X=X, X_prime=X_prime, y=y)

        # support vector machine
        model = svm.SVC()
        model.fit(X, y)
        yhat = list(model.predict(X))
        print('SVM audit:')
        audit(predictions=yhat, X=X, X_prime=X_prime, y=y)

        # nearest neighbor
        model = neighbors.KNeighborsClassifier(3)
        model.fit(X, y)
        yhat = list(model.predict(X))
        print('nearest neighbors audit:')
        audit(predictions=yhat, X=X, X_prime=X_prime, y=y)

        # Marginal reduction with Reg Oracle
        X, X_prime_cts, y = clean_generic_data.clean_dataset(dataset, attributes)
        X_prime = X_prime_cts.iloc[:,:]
        n = X.shape[0]
        # threshold sensitive features by average value
        sens_means = np.mean(X_prime)
        for col in X_prime.columns:
            X.loc[(X[col] > sens_means[col]), col] = 1
            X_prime.loc[(X_prime[col] > sens_means[col]), col] = 1
            X.loc[(X[col] <= sens_means[col]), col] = 0
            X_prime.loc[(X_prime[col] <= sens_means[col]), col] = 0
        yhat = marginal_preds(X, X_prime, X_prime_cts, y, max_iters, True)

