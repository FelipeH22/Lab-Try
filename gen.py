import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from uncertainties import ufloat

def gen_plot(i):
    df=pd.read_csv(f'r{i}.csv')
    df.rename(columns = {'L-X':'d'}, inplace = True)
    df['ref']=list(map(lambda x: ufloat(x,0.5),df.ref.values))
    df['X']=list(map(lambda x: ufloat(x,0.05)/100, df.X.values))
    df['d']=list(map(lambda x: ufloat(x,0.05)/100, df.d.values))
    df['R_x']=[(df['ref'][i]*df['d'][i])/df['X'][i] for i in range(len(df.ref))]
    """
    #Exporta tabla a latex
    print(f"R_3={sum(df['R_x'])/10}")
    print(df.to_latex(index=False))
    """
    df['ref'] = list(map(lambda x: x.nominal_value, df.ref.values))
    df['R_x'] = list(map(lambda x: x.nominal_value, df.R_x.values))
    x=df.ref.values.reshape(10, 1)
    y=df.R_x.values.reshape(10, 1)
    regr=LinearRegression()
    regr.fit(x,y)
    plt.scatter(x,y,color='black')
    plt.xlabel('R_2')
    plt.ylabel('R_x')
    plt.plot(x,regr.predict(x),color='blue',linewidth=3,)
    plt.title(f'r_{i}\n avg(r_{i})={round(sum(df["R_x"]) / 10, 3)} intercepto:{round(regr.intercept_[0], 3)} Coeficiente:{round(regr.coef_[0][0], 3)}')
    plt.savefig(f'p{i}.png')
    plt.clf()

for i in range(1,5):
    gen_plot(i)