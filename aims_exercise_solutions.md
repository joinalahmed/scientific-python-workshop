# Numpy

Generate a NumPy array of 1000 random numbers sampled from a Poisson distribution, with parameter `lam=5`. What is the modal value in the sample?

```python
x = np.random.poisson(5, size=100)
[sum(x==i) for i in np.unique(x)]
```

Generate the following structure as a numpy array, without typing the values by hand. Then, create another array containing just the 2nd and 4th rows.

        [[1,  6, 11],
         [2,  7, 12],
         [3,  8, 13],
         [4,  9, 14],
         [5, 10, 15]]
         
```python
np.reshape(np.arange(15)+1, (3,5)).T
```

Create a **tridiagonal** matrix with 5 rows and columns, with 1's on the diagonal and 2's on the off-diagonal.

```python
np.eye(5) + np.eye(5, k=1)*2 + np.eye(5, k=-1)*2 
```

Divide each column of the array:

        np.array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19],
           [20, 21, 22, 23, 24]])

elementwise with the array `np.array([1., 5, 10, 15, 20])`.

```python
a = np.arange(25).reshape(5, 5)
b = np.array([1., 5, 10, 15, 20])
(a.T/b).T
```

Generate a 10 x 3 array of random numbers (in range [0,1]). For each row, pick the number closest to 0.5.

```
x = np.random.random((10,3))
y = np.abs(x - 0.5)
y[y.argsort()==0]
```

# Matplotlib

Plot the two series on the same axes, and use a legend to label the series. (**Hint**: you must first give the original plot a `label`)

```python
plt.figure(figsize=(14,6))
plt.plot(x, y, 'r--', label='Jan')
plt.plot(x, rain['Feb'], 'g*-', label='Feb')
plt.legend()
```

Create a 2x2 grid of plots, rather than a single column. Think about how you would iterate over the axes in this case.

```python
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10,10))

months = rain.dtype.names[1:]

for i,ax in enumerate(axes.flatten()):
    ax.plot(rain['Year'], rain[months[i]], 'r')
    if i>1:
        ax.set_xlabel('Year')
    if not i%2:
        ax.set_ylabel('Rainfall')
    ax.set_title(months[i])
    
fig.tight_layout()
```

Give the second set of axes above a grid with a **dashed** line with a line width of 0.5.

```
axes[1].grid(ls='--', lw=0.5)
```

# Intro to Pandas

From the `data` table above, create an index to return all rows for which the phylum name ends in "bacteria" and the value is greater than 1000.

```
data[(data.phylum.str.endswith('bacteria')) & (data.value>1000)]
```

You can use the isin method query a DataFrame based upon a list of values as follows:

    microbiome['Taxon'].isin(['Firmacutes', 'Bacteroidetes'])
    
Use isin to find all players that played for the Los Angeles Dodgers (LAN) or the San Francisco Giants (SFN). How many records contain these values?

```
baseball[baseball['team'].isin(['LAN', 'SFN'])].shape
```

Calculate on base percentage for each player, and return the ordered series of estimates.

```
((baseball.h + baseball.bb + baseball.hbp) / 
 (baseball.ab + baseball.bb + baseball.hbp + baseball.sf)).sort_values(ascending=False)
```

Ebola import

```python
ebola_dirs = !ls data/ebola/

import glob

filenames = {data_dir[:data_dir.find('_')]: 
             glob.glob('data/ebola/{0}/*.csv'.format(data_dir)) 
             for data_dir in ebola_dirs[1:]}

datasets = []
for country in filenames:
    
    country_files = filenames[country]
    for f in country_files:
        
        data = pd.read_csv(f)
        
        
        # Convert to lower case to avoid capitalization issues
        data.columns = data.columns.str.lower()
        # Column naming is inconsistent. These procedures deal with that.
        keep_columns = ['date']
        if 'description' in data.columns:
            keep_columns.append('description')
        else:
            keep_columns.append('variable')
            
        if 'totals' in data.columns:
            keep_columns.append('totals')
        else:
            keep_columns.append('national')
            
        # Index out the columns we need, and rename them
        keep_data = data[keep_columns]
        keep_data.columns = 'date', 'variable', 'totals'
        
        # Extract the rows we might want
        lower_vars = keep_data.variable.str.lower()
        # Of course we can also use regex to do this
        case_mask = (lower_vars.str.contains('new') 
                     & (lower_vars.str.contains('case') | lower_vars.str.contains('suspect') 
                                                        | lower_vars.str.contains('confirm')) 
                     & ~lower_vars.str.contains('non')
                     & ~lower_vars.str.contains('total'))
        
        keep_data = keep_data[case_mask].dropna()
        
        # Convert data types
        keep_data['date'] = pd.to_datetime(keep_data.date)
        keep_data['totals'] = keep_data.totals.astype(int)
        
        # Assign country label and append to datasets list
        datasets.append(keep_data.assign(country=country))

all_data = pd.concat(datasets)
all_data.head(10)     
```

# Data Wrangling

AIS Transformation

```
segments.seg_length.apply(np.log).hist(bins=500)
```

Generate counts of vessel types in the vessels DataFrame for only this subset of vessels.

```
vessels[vessels.type.apply(lambda x: x in subset)].type.value_counts()
```

Label the lone column in each DataFrame as "Count" and the respective indices as "Taxon"

```
mb1.columns = mb2.columns = ['Count']
mb1.index.name = mb2.index.name = 'Taxon'
```

In the data/microbiome subdirectory, there are 9 spreadsheets of microbiome data that was acquired from high-throughput RNA sequencing procedures, along with a 10th file that describes the content of each. Write code that imports each of the data spreadsheets and combines them into a single DataFrame, adding the identifying information from the metadata spreadsheet as columns in the combined DataFrame.

```python
metadata = pd.read_excel('../data/microbiome/metadata.xls', sheetname='Sheet1')

chunks = []
for i in range(9):
    this_file = pd.read_excel('../data/microbiome/MID{0}.xls'.format(i+1), 'Sheet 1', index_col=0, header=None, names=['Taxon', 'Count'])
    this_file.columns = ['Count']
    this_file.index.name = 'Taxon'
    for m in metadata.columns:
        this_file[m] = metadata.ix[i][m]
    chunks.append(this_file)

pd.concat(chunks)
```
    
Which columns uniquely define a row? Create a DataFrame called cdystonia2 with a hierarchical index based on these columns.

```python
cdystonia2 = cdystonia.set_index(['patient','obs'])
cdystonia2.index.is_unique
```


Create a subset of the vessels DataFrame called vessels5 that only contains the 5 most common types of vessels, based on their prevalence in the dataset.

```
top5 = vessels.type.isin(vessels.type.value_counts().index[:5])
vessels5 = vessels[top5]
```

Use the discretized segment lengths as the input for get_dummies to create 5 indicator variables for segment length:

```
pd.get_dummies(quantiles).head(10)
```

Use the quantile method to generate the median values of the twstrs variable for each patient.

```
cdystonia.groupby('patient')['twstrs'].quantile(.50)
```

Women and children first?
- Use the groupby method to calculate the proportion of passengers that survived by sex.
- Calculate the same proportion, but by class and sex.
- Create age categories: children (under 14 years), adolescents (14-20), adult (21-64), and senior(65+), and calculate survival proportions by age category, class and sex.

```
titanic = pd.read_excel('../data/titanic.xls')
titanic.groupby('sex')['survived'].mean()
titanic.groupby(['sex','pclass'])['survived'].mean()
titanic['age_cat'] = pd.cut(titanic.age, [0,14,21,65,100], right=False)
titanic.groupby(['sex','pclass','age_cat'])['survived'].mean()
```

# Statistical data analysis

Cervical dystonia estimation:

```python
norm_like = lambda theta, x: -np.log(norm.pdf(x, theta[0], theta[1])).sum()

fmin(norm_like, np.array([1,2]), args=(cdystonia.twstrs[(cdystonia.obs==6) & (cdystonia.treat=='Placebo')],))
fmin(norm_like, np.array([1,2]), args=(cdystonia.twstrs[(cdystonia.obs==6) & (cdystonia.treat=='5000U')],))
```

# Regression modeling

General polynomial function:

```
def calc_poly(params, data):
    x = np.c_[[data**i for i in range(len(params))]]
    return np.dot(params, x)
```

Which other variables might be relevant for predicting the probability of surviving the Titanic? Generalize the model likelihood to include 2 or 3 other covariates from the dataset.

```python
def logistic_like(theta, x, y):
    
    p = invlogit(theta[0] + np.dot(theta[1:], x))
    
    # Return negative of log-likelihood
    return -np.sum(y * np.log(p) + (1-y) * np.log(1 - p))
    
x = titanic[titanic.fare.notnull()].assign(male=titanic.sex.replace({'male':1, 'female':0}))[['fare', 'male']].values.T

b0, b1, b2 = fmin(logistic_like, [0.5,0,0], args=(x,y))
b0, b1, b2
```

# Resampling

Extend the subsetting operation from above to create a function to generate five random partitions of the dataset. Call the function `gen_k_folds`.

```python
def gen_k_folds(data, k=5):
    n = int(data.shape[0]/k)
    remaining_data = data.copy()
    training, testing = [], []
    while remaining_data.shape[0]>n:
        sample = remaining_data.sample(n=n)
        testing.append(sample)
        training.append(data.drop(sample.index))
        remaining_data = remaining_data.drop(sample.index)
    testing.append(remaining_data)
    training.append(data.drop(remaining_data.index))
    return training, testing
    
train_subsets, test_subsets = gen_k_folds(salmon)

k = 5
fig, axes = plt.subplots(1, k, figsize=(14,4))
for i in range(k):
    axes[i].plot(train_subsets[i].spawners, train_subsets[i].recruits, 'ro')
    axes[i].plot(test_subsets[i].spawners, test_subsets[i].recruits, 'bo')
    
plt.tight_layout()

```


```python
x = cdystonia.twstrs[(cdystonia.obs==6) & (cdystonia.treat=='Placebo') & (cdystonia.twstrs.notnull())].values
n = len(x)
s = [x[np.random.randint(0,n,n)].mean() for i in range(R)]
placebo_mean = np.sum(s)/R

s_sorted = np.sort(s)
alpha = 0.05
s_sorted[[(R+1)*alpha/2, (R+1)*(1-alpha/2)]]
```

# High-level Plotting

Survivor KDE plots:

```python
surv = dict(list(titanic.groupby('survived')))
for s in surv:
    surv[s]['age'].dropna().plot(kind='kde', label=bool(s)*'survived' or 'died', grid=False)
plt.legend()
plt.xlim(0,100)
```

# Model selection

```python
y = vlbw.pop('ivh')

def brier(mod, X, y): 
    return ((mod.fit(X,y).predict_proba(X) - y)**2).mean()

C = np.logspace(-3, 2, 20)

scores = np.empty(len(C))
scores_std = np.empty(len(C))


for i,c in enumerate(C):
    lr = linear_model.LogisticRegression(C=c)
    s = cross_validation.cross_val_score(lr, vlbw, y, scoring=brier, n_jobs=-1)
    scores[i] = s.mean()
    scores_std[i] = s.std()

plt.semilogx(alphas, scores)
plt.semilogx(alphas, np.array(scores) + np.array(scores_std)/20, 'b--')
plt.semilogx(alphas, np.array(scores) - np.array(scores_std)/20, 'b--')
plt.yticks(())
plt.ylabel('Brier score')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.text(5e-2, np.max(scores)+1e-4, str(np.max(scores).round(3)))



from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

y = vlbw.pop('ivh').values
X = vlbw

from sklearn.grid_search import GridSearchCV
cvalues = [0.1, 1, 10,100]
grid = GridSearchCV(LogisticRegression(), 
    param_grid={'C': cvalues}, scoring='average_precision')
gf = grid.fit(X, y)
gf.grid_scores_

grid = GridSearchCV(LogisticRegression(), 
            param_grid={'C': cvalues}, scoring='roc_auc')
gf = grid.fit(X, y)
gf.grid_scores_
```

# Clustering

Use residual sums of squares to select competitive threshold values for the predictive model defined above

```python
# Predictors: "age" "sex" "bmi" "map" "tc"  "ldl" "hdl" "tch" "ltg" "glu"
diabetes = load_diabetes()
y = diabetes['target']
bmi, ltg = diabetes['data'][:,[2,8]].T

def rss_cost(data, mask):
    this_cost = ((data[mask] - np.mean(data[mask]))**2).sum()
    that_cost = ((data[~mask] - np.mean(data[~mask]))**2).sum()
    return this_cost + that_cost
    
ltg_space = np.linspace(ltg.min(), ltg.max())
ltg_cost = [rss_cost(y, ltg<c) for c in ltg_space]
ltg_opt = ltg_space[np.argmin(ltg_cost)]
print(ltg_opt)

bmi_space = np.linspace(bmi[ltg>ltg_opt].min(), bmi[ltg>ltg_opt].max())
bmi_cost = [rss_cost(y[ltg>ltg_opt], bmi[ltg>ltg_opt]<c) for c in bmi_space]
bmi_opt = bmi_space[np.argmin(bmi_cost)]
print(bmi_opt)

plt.scatter(ltg, bmi,  c=y)
plt.vlines(ltg_opt, *plt.gca().get_ylim(), linestyles='dashed')
plt.hlines(bmi_opt, ltg_opt, plt.gca().get_xlim()[1], linestyles='dashed')
plt.colorbar()
plt.xlabel('ltg'); plt.ylabel('bmi')
```

Select the optimal random forest regression model for the nashville daily temperature data via cross-validation in scikit-learn. Use the number of estimators and the maximim leaf nodes as tuning parameters.

```python
daily_temps = pd.read_table("../data/TNNASHVI.txt", sep='\s+', 
                            names=['month','day','year','temp'], na_values=-99)
                            
# Transmogrify data
y = daily_temps.temp[daily_temps.year>2010]
X = np.atleast_2d(np.arange(len(y))).T

rf = RandomForestRegressor()
parameters = {'n_estimators':[10, 50, 100, 200, 300],
              'max_leaf_nodes':[3, 5, 7, 9, 11, 13]}

### Warning: be sure your data is shuffled before using GridSearch!
clf_grid = grid_search.GridSearchCV(rf, parameters)
clf_grid.fit(*shuffle(X, y))

rf_best = clf_grid.best_estimator_
X_fit = np.linspace(0, len(X), 1000).reshape((-1, 1))
y_fit_best = rf_best.predict(X_fit)

print((rf_best.n_estimators, rf_best.max_depth))

plt.plot(X.ravel(), y, '.k', alpha=0.3)
plt.plot(X_fit.ravel(), y_fit_best, color='red')

```

## Bayesian Computation

Perform a Bayesian sensitivity analysis by performing SIR on the stomach cancer dataset $N$ times, with one observation (a city) removed from the dataset each time. Calculate and plot posterior medians and 95% posterior intervals for each $f(\theta|y_{(-i)})$ to visually analyze the influence of each observation.

```python
datasets = [cancer.drop(i).values.T for i in range(len(cancer))]

def sir(data, var=var, mode=mode, samples=1000):
    
    y, n = data
    
    # Sample from q
    theta = rmvt(5, var, mode, size=samples)
    
    f_theta = np.array([betabin_trans(t, n, y) for t in theta])
    
    q_theta = mvt(theta, 4, var, mode)
    
    w = np.exp(f_theta - q_theta - max(f_theta - q_theta))
    
    p_sir = w/w.sum()
    
    theta_sir = theta[np.random.choice(range(len(theta)), size=samples, p=p_sir)]
    
    logK_sample = theta_sir[:,1]
    
    return logK_sample
    
samples = [sir(d) for d in datasets]

_ = plt.boxplot(samples)
```

## MCMC

Use Metropolis-Hastings sampling to fit a Bayesian model to analyze this bioassay data, and to estimate LD50.

```python
ld50 = lambda alpha, beta: -alpha/beta

invlogit = lambda x: 1/(1. + np.exp(-x))

dbinom = distributions.binom.logpmf
dnorm = distributions.norm.logpdf

def bioassay_post(alpha, beta):

    logp = dnorm(alpha, 0, 10000) + dnorm(beta, 0, 10000)

    p = invlogit(alpha + beta*np.array(log_dose))

    logp += dbinom(deaths, n, p).sum()

    return logp

def metropolis_bioassay(n_iterations, initial_values, prop_var=1,
                     tune_for=None, tune_interval=100):

    n_params = len(initial_values)

    # Initial proposal standard deviations
    prop_sd = [prop_var] * n_params

    # Initialize trace for parameters
    trace = np.empty((n_iterations+1, n_params))

    # Set initial values
    trace[0] = initial_values
    # Initialize acceptance counts
    accepted = [0]*n_params

    # Calculate joint posterior for initial values
    current_log_prob = bioassay_post(*trace[0])

    if tune_for is None:
        tune_for = n_iterations/2

    for i in range(n_iterations):

        if not i%1000: print('Iteration', i)

        # Grab current parameter values
        current_params = trace[i]

        for j in range(n_params):

            # Get current value for parameter j
            p = trace[i].copy()

            # Propose new value
            theta = rnorm(current_params[j], prop_sd[j])

            # Insert new value
            p[j] = theta

            # Calculate log posterior with proposed value
            proposed_log_prob = bioassay_post(*p)

            # Log-acceptance rate
            alpha = proposed_log_prob - current_log_prob

            # Sample a uniform random variate
            u = runif()

            # Test proposed value
            if np.log(u) < alpha:
                # Accept
                trace[i+1,j] = theta
                current_log_prob = proposed_log_prob
                accepted[j] += 1
            else:
                # Reject
                trace[i+1,j] = trace[i,j]

            # Tune every 100 iterations
            if (not (i+1) % tune_interval) and (i < tune_for):

                # Calculate aceptance rate
                acceptance_rate = (1.*accepted[j])/tune_interval
                if acceptance_rate<0.2:
                    prop_sd[j] *= 0.9
                elif acceptance_rate>0.5:
                    prop_sd[j] *= 1.1

                accepted[j] = 0

    return trace[tune_for:], accepted

tr, acc = metropolis_bioassay(10000, (0,0))

for param, samples in zip(['intercept', 'slope'], tr.T):
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(samples)
    axes[0].set_ylabel(param)
    axes[1].hist(samples[len(samples)//2:])

a, b = tr.T
print('LD50 mean is {}'.format(ld50(a,b).mean()))
```