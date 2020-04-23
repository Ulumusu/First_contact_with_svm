import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

style.use("bmh")
from sklearn import svm


def download_currency_values(d1, d2):
    df = pd.DataFrame(columns=['Date', 'Code', 'Buy', 'Sell'])
    datelist = pd.date_range(start=d1, end=d2)
    codes = ["USD", "EUR", "CZK"]
    for code in codes:
        for date in datelist:
            dt = str(date)
            try:
                read_json = pd.read_json("http://api.nbp.pl/api/exchangerates/rates/C/" +
                                         str(code) + "/" + str(dt.split()[0])
                                         + "/?format=json&fbclid=IwAR0r2IoMvQi_vka8-n3sQPRukcOSyA1xckibzZhgKXSGJ_a5uc94LMuU-0c")
                df = df.append({'Date': date, 'Code': code,
                                'Buy': read_json["rates"][0]["bid"],
                                'Sell': read_json["rates"][0]["ask"]},
                               ignore_index=True)
            except:
                pass
    df.to_csv("Currency_rates.csv")


def Build_Data_Set(features=["Buy", "Sell"]):
    data_df = pd.read_csv("Currency_rates.csv")
    data_df = data_df[:3090]
    data_df = data_df.reindex(np.random.permutation(data_df.index))
    X = np.array(data_df[features].values)  # .tolist())
    y = (data_df["Code"]
         .replace("USD", 0)
         .replace("EUR", 1)
         .values.tolist())

    return X, y


def Analysis():
    X, y = Build_Data_Set()

    clf = svm.SVC(kernel="linear", C=1)
    clf.fit(X, y)

    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min(X[:, 0]), max(X[:, 0]))
    yy = a * xx - clf.intercept_[0] / w[1]

    h0 = plt.plot(xx, yy, "g")

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, )
    plt.ylabel("Sell")
    plt.xlabel("Buy")
    plt.legend()

    plt.show()


d1 = '1/1/2013'
d2 = '2/21/2019'

# download_currency_values(d1, d2)
Analysis()
