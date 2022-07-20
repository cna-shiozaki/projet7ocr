class Mock():
    def fit(self, X,y):
        print("Mock Fit is called")
        pass

    def score(self, X,y):
        print("Mock Score is called")
        return 0.0

    def transform(self, X):
        print("Mock Transform is called")
        return X


def in_ipython():
    try:
        return __IPYTHON__
    except NameError:
        return False