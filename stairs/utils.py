import matplotlib.pyplot as plt
import numpy as np


def barplot(data, title):
    assert (data is not None)
    plt.title(title)
    plt.bar(np.array(range(data.shape[0])), data, align='center', color="blue")
    plt.grid(True)
    plt.show()


def lineplot(data, title, **kwargs):
    assert (data is not None), "data is empty"
    plt.xlabel('Periods')
    plt.title(title)
    plt.grid(True)
    plt.ylim([0,1.2])
    plt.plot(data)
    if kwargs.get("save",None) is not None:
        save=kwargs["save"]
        plt.savefig(save,dpi=600,format='png')
        return
    plt.show()


def multiple_lineplots(data, title, legend_loc="upper right"):
    xaxis = np.arange(data.shape[1])
    funds = []
    for i in range(data.shape[0]):
        handle, = plt.plot(data[i, :])
        funds.append((handle, "Fund {0}".format(i)))
    plt.xticks(xaxis, ["Vintage {0}".format(k) for k in range(data.shape[1])])
    plt.legend(*zip(*funds), loc=legend_loc)
    plt.title(title)
    plt.grid(True)
    # plt.autoscale()
    plt.show()


def multiple_barplots(data, title, legend_loc="upper right"):
    xaxis = np.arange(data.shape[1])
    width = 0.25
    funds = []
    for i in range(data.shape[0]):
        funds.append((plt.bar(xaxis + i * width, data[i, :], width), "Fund {0}".format(i)))
    plt.xticks(xaxis + width / 2, ["Vintage {0}".format(k) for k in range(data.shape[1])])
    plt.legend(*zip(*funds), loc=legend_loc)
    plt.title(title)
    plt.grid(True)
    # plt.autoscale()
    plt.show()


def read_file(path, sep=";", cast=None):
    print(path)
    with open(path, "r") as fd:
        for line in fd:
            row = line.split(sep)
            if cast:
                row = list(map(cast, row))
            yield row

def read_parameters_file(path, sep=";"):
    cast=[str,str,int,int,int,float,float,int,int,str,float,int,int,str,float,float,float]
    with open(path, "r") as fd:
        cnt = 0
        params={}
        for line in fd:
            row = line.replace("\n","").split(sep)
            params[row[1]]=cast[cnt](row[2])
            cnt += 1
            if cnt >= len(cast):
               yield params
               cnt=0
               params={}


class listSet(list):
    """

    Just a custom class implementing a simple key and index concept

    """

    def __init__(self):
        """
        Constuctor

        Create a list object with dictionnary to map key
        """
        super().__init__()
        self.map={}

    def append(self,o):
        """

        Parameters
        ----------
        o: object to be added

        Returns
        -------

        """
        # Get a value of the hash called the key
        key=hash(o)
        # if thet key does not exists, add the (key,o) to the dictionnary
        if self.map.get(key,None) is None:
            self.map[key]=o
            super().append(o)

    def extend(self, it):
        """

        Parameters
        ----------
        it: Another iterable

        Returns
        -------

        """
        for element in it:
            self.append(element)

    def remove(self,x):
        """

        Parameters
        ----------
        x: Remove object

        Returns
        -------

        """
        key=hex(hash(x))
        del self.map[key]
        super().remove(key)

    def pop(self, i):
        """


        Parameters
        ----------
        i: Index of object

        Returns
        -------
        The object

        """
        x = super().pop(i)
        key = hex(hash(x))
        del self.map[key]
        return x

    def get(self,key,default=None):
        """

        Parameters
        ----------
        key: Object key
        default: Default value (object) to return if the key is not present

        Returns
        -------
        The Object or the default value (object)
        """
        if self.map.get(key,None) is None:
            return default
        return self.map[key]



def bisect_left(a, x, lo=0, hi=None,key=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(x) will
    insert just before the leftmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        # Use __lt__ to match the logic in list.sort() and in heapq
        if key is None:
            value = a[mid]
        else:
            value = key(a[mid])
        # Use __lt__ to match the logic in list.sort() and in heapq
        if value < x: lo = mid+1
        else: hi = mid
    return lo



def bisect_right(a, x, lo=0, hi=None,key=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if key is None:
            value = a[mid]
        else:
            value = key(a[mid])
        # Use __lt__ to match the logic in list.sort() and in heapq
        if x < value: hi = mid
        else: lo = mid+1
    return lo


def limits(value, mini=-1e+8, maxi=1e+8):
    return min(max(value, mini), maxi)


def skew_random(alpha,size):
    if  abs(alpha) < 1e-4:
        return  np.random.normal(0,1,size)
    U = np.random.normal(0,1,size)
    V = np.random.normal(0,1,size)
    X=None
    if abs(alpha + 1) < 1e-4:
       X = np.fmin(U,V)
    elif abs(alpha - 1) < 1e-4:
       X = np.fmax(U,V)
    else:
        lambda1 = (1+alpha)/(np.sqrt(2*(1+alpha**2)))
        lambda2 = (1-alpha)/(np.sqrt(2*(1+alpha**2)))
        assert abs((lambda1**2+lambda2**2)-1) < 1e-4, "Error sum != 1"
        X = lambda1 * np.fmax(U,V) + lambda2 * np.fmin(U,V)
    return X


def simcorr(values,alpha=0,corr=0.5):
    # Get angle from correlation coefficient
    theta = np.arccos(corr)
    x = np.array(values)
    # Center data
    x_center = np.transpose(np.atleast_2d(x - np.mean(x)))
    # Generate a second vector
    y = skew_random(alpha,len(x))
    # Center it
    y_center = np.transpose(np.atleast_2d(y - np.mean(y)))
    X = np.concatenate((x_center,y_center),axis=1)
    ID = np.identity(len(x))
    # QR decomposition to get Q for the projection
    Q,_ = np.linalg.qr(np.atleast_2d(X[:,0]).transpose())
    P = np.dot(Q,Q.transpose())
    # Get an orthogonal vector
    y_ortho = np.dot((ID - P), X[:,1])
    X_new = np.concatenate((np.atleast_2d(X[:,0]).transpose(),np.atleast_2d(y_ortho).transpose()),axis=1)
    X_new_scale = X_new / np.std(X_new,axis=0)
    # New vector with specific theta as angle with values
    return list(X_new_scale[:,1] + (1/np.tan(theta))*X_new_scale[:,0])

def to_proba_scores(values):
    return np.exp(values)/sum(np.exp(values))

def binary_tournament(values):
    indexes = np.random.choice(len(values),2)
    return indexes[0] if values[indexes[0]] >= values[indexes[1]] else  indexes[1] 




if __name__ == "__main__":
    a=[4,1,5,0,3]
    b = simcorr(a,-0.50)
    b1 = b + np.abs(np.min(b))
    b2 = b1/np.sum(b1)
    print(b2)
    c =  to_proba_scores(b)
    print(c)
    print(np.corrcoef(a,b))

    fig, axs = plt.subplots(5, 2)
    st = fig.suptitle("Different skewness", fontsize="x-large")
    axes = axs.flatten()
    for k,alpha in enumerate(np.linspace(-10,10,10)):
        skew = skew_random(alpha,100000)
        axes[k].hist(skew,bins=200)
        axes[k].set_title("Skew = {0}".format(int(alpha)))
    fig.tight_layout()
    plt.show()






