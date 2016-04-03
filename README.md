# AIMS Scientific Python Workshop

## Schedule

### Day 1

| Time | Topic | Packages |
|-------|-------|----------|
|9:00-10:30|**Introduction to Python**|          |
|10:30-10:45|*Break*|          |
|10:45-12:15|**IPython and Jupyter**|IPython, Jupyter|
|12:15-13:15|*Lunch*|          |
|13:15-14:45|**Scientific Python Programming**|NumPy, SciPy|
|14:45-15:00|*Break*|          |
|15:00-16:30|**Plotting in Python**       |Matplotlib, Bokeh|

### Day 2

| Time | Topic | Packages |
|-------|-------|----------|
|9:00-10:30|**Introduction to pandas**|pandas|
|10:30-10:45|*Break*|          |
|10:45-12:15|**Data Wrangling with pandas (I)**|pandas|
|12:15-13:15|*Lunch*|          |
|13:15-14:45|**Data Wrangling with pandas (II)**|pandas|
|14:45-15:00|*Break*|          |
|15:00-16:30|**Python for Data Analysis**|pandas, NumPy, SciPy|

### Day 3

| Time | Machine Learning Option | Bayesian Modeling Option |
|-------|-------|----------|
|9:00-10:30|**Supervised and Unsupervised Learning**|Bayesian Inference|
|10:30-10:45|*Break*|          |
|10:45-12:15|**Introduction to `scikit-learn`**|Markov chain Monte Carlo|


   
## Software Installation

This workshop is taught using Python 3 and the "Scientific Stack", a set of core scientific computing packages written and maintained by various third parties.

### Python

The first step is to install Python on your computer. I will be teaching this course based on **Python 3.5**. If Python 3 is not on your system, you can either download an installer from [Python.org](http://python.org) or install a third-party distribution (see below). I recommend the latter, since these distributions are enhanced, containing most or all of the packages required for the course.

In addition to Python itself, we will be making use of several packages in the scientific stack. These include the following:

* [NumPy](http://www.numpy.org/ "NumPy &mdash; Numpy")
* [SciPy](http://www.scipy.org/ "SciPy.org &mdash; SciPy.org")
* [IPython](http://ipython.org/ "Announcements &mdash; IPython")
* [Pandas](http://pandas.pydata.org/ "Python Data Analysis Library &mdash; pandas: Python Data Analysis Library")
* [Matplotlib](http://matplotlib.org/ "matplotlib: python plotting &mdash; Matplotlib 1.2.1 documentation")
* [PyMC](https://github.com/pymc-devs/pymc "pymc-devs/pymc · GitHub")
* [scikit-learn](http://scikit-learn.org/ "scikit-learn: machine learning in Python &mdash; scikit-learn 0.13.1 documentation")

### All-in-one Scientific Python

Perhaps the easiest way to get a feature-complete version of Python on your system is to install the [Anaconda](http://continuum.io/downloads.html) distribution by Continuum Analytics. Anaconda is a completely free Python environment that includes includes almost 200 of the best Python packages for science and data analysis. Its simply a matter of downloading the installer (either graphical or command line), and running it on your system.

Be sure to download the Python 3.5 installer, by following the **Python 3.5 link** for your computing platform (Mac OS X example shown below).

![get Python 3](http://fonnesbeck-dropshare.s3.amazonaws.com/687474703a2f2f666f6e6e65736265636b2d64726f7073686172652e73332e616d617a6f6e6177732e636f6d2f53637265656e2d53686f742d323031362d30332d31382d61742d332e32342e32362d504d2e706e67.png)

To install the packages required for this course, the easiest and safest way is to create a suitable environment by typing the following in your terminal:

    conda create -n pyaims python=3 sympy numpy scipy jupyter ipyparallel pandas matplotlib scikit-learn seaborn patsy rpy2

This creates a self-contained Python environment in your home directory (called `pyaims`) that includes all the packages you will need, along with their dependencies. To use this environment at any time, type:

    source activate pyaims

To exit the `pyaims` environment, you can switch it off via:

    source deactivate
    
## Downloading Course Materials

The final step is accessing the course materials. **If you are familiar with Git**, you can simply clone this repository:

    git clone https://github.com/fonnesbeck/scientific-python-workshop.git
    
Otherwise, you may download a zip archive containing the course content. Near the top right-hand part of the repository main page, you should see a **Download ZIP** button.

![download zip](http://fonnesbeck-dropshare.s3.amazonaws.com/Screen-Shot-2016-03-31-07-46-51.png)

Clicking this will initiate the download. Unzipping the file (or cloning the repo) will generate a directory called `scientific-python-workshop`, within which will be the same directory structure that you see at the top of the repository main page, which includes two subdirectories:

* `data`
* `notebooks`

We will be accessing the Jupyter notebook files (suffix `.ipynb`) in the `notebooks` subdirectory.