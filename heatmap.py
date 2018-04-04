import matplotlib
matplotlib.use('TkAgg')
from Reg_Oracle_Fict import *
from Reg_Oracle_Class import *
import seaborn as sns

def calc_disp(A_p, X, y_g, X_sens, g):
    """Return the fp disparity in a group g."""
    X_0 = pd.DataFrame([X_sens.iloc[u, :] for u, s in enumerate(y_g) if s == 0])
    group_0 = g.predict(X_0)
    n = len(y_g)
    g_size_0 = np.sum(group_0) * 1.0 / n
    FP = [A_p[i] for i, c in enumerate(y_g) if c == 0]
    FP = np.mean(FP)
    group_members = g.predict(X_sens)
    fp_g = [A_p[i] for i,c in enumerate(y_g) if group_members[i] == 1 and c == 0]
    if len(fp_g) == 0:
        return 0
    fp_g = np.mean(fp_g)
    return (FP - fp_g) * g_size_0


def heat_map(X, X_prime, y, A, eta, plot_name, mini=None, maxi=None):
    columns = [str(c) for c in X_prime.columns]
    columns.append('gamma-disparity')
    q = int(1/eta*1/eta)
    mat = pd.DataFrame(columns=columns, index=range(q))
    # calculate initial heatmap
    ind = 0.0
    for i in range(int(1/eta)):
        for j in range(int(1/eta)):
            beta = [-1 + 2*eta*i, -1 + 2*eta*j]
            group = LinearThresh(beta)
            mat.iloc[int(ind),:] = [beta[0], beta[1], calc_disp(A_p=A, X=X, y_g=y, X_sens=X_prime, g=group)]
            print(ind/q)
            ind += 1.0
    mat_list = pd.DataFrame({c: list(mat[c]) for c in mat.columns})
    mat_piv = mat_list.pivot(index=mat.columns[0], columns=mat.columns[1], values=mat.columns[2])
    if mini is None or maxi is None:
        figure = sns.heatmap(mat_piv, fmt='g')
    else:
        figure = sns.heatmap(mat_piv, fmt='g', vmin=mini, vmax=maxi)
    fig = figure.get_figure()
    fig.savefig('{}'.format(plot_name))
    fig.clf()
    mat_list.to_csv('{}.csv'.format(plot_name))
    return [np.min(mat.loc[:, 'gamma-disparity']), np.max(mat.loc[:, 'gamma-disparity']), mat_list]


if __name__ == "__main__":

    # experiments
    C, num_sens, printflag, dataset, oracle, max_iters, gamma, fairness_def, plot_name = 100, 2, True, 'communities', 'reg_oracle', 1000, .0001, 'gamma', 'test'

    # Data Cleaning and Import
    f_name = 'clean_{}'.format(dataset)
    clean_the_dataset = getattr(clean_data, f_name)
    X, X_prime, y = clean_the_dataset(num_sens)
    # subsample
    num = 100
    col = 14
    X = X.iloc[0:num,0:col]
    y = y[0:num]
    X_prime = X_prime.iloc[0:num]

    # get base classifier
    n = X.shape[0]
    m = len([s for s in y if s == 0])
    p = learner_br([1.0 / n] * m, X, y)
    A = p.predict(X)
    heat_map(X, X_prime, y, A, eta, plot_name)