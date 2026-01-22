```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
```


```python
filename="Salary_dataset.csv"
```


```python
df=pd.read_csv(filename,usecols=['YearsExperience','Salary'])
```


```python
df['sqr_sal'] = df['Salary'] ** 2
df['sqr_yrs'] = df['YearsExperience'] ** 2
df['xy'] = df['YearsExperience'] * df['Salary']

n = len(df)

m = (n * df['xy'].sum() - df['YearsExperience'].sum() * df['Salary'].sum()) / \
    (n * df['sqr_yrs'].sum() - (df['YearsExperience'].sum() ** 2))

b = (df['Salary'].sum() - m * df['YearsExperience'].sum()) / n

equation_label = f'y = {m:.2f}x + {b:.2f}'

x = np.linspace(df['YearsExperience'].min(), df['YearsExperience'].max(), 100)
y = m * x + b

```


```python
plt.plot(x, y, label=equation_label, color='red' ,linestyle=":")

plt.xlabel("Years of Experience")
plt.ylabel("Salary(in Dollars)")
plt.scatter(df['YearsExperience'],df['Salary'])
plt.legend() 
plt.grid(True) 


plt.show()
```


    
![png](output_4_0.png)
    



```python
x_trained = 5
```


```python
y_trained = m * x_trained + b
```


```python
print(int(y_trained), " is the expected pay")
```

    72098  is the expected pay
    


```python
from sklearn.linear_model import LinearRegression

# Features and target
X = df[['YearsExperience']]  # 2D
y = df['Salary']              # 1D

# Train model
model = LinearRegression()
model.fit(X, y)

print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)

x_trained = 5
prediction = model.predict([[x_trained]])
print(f"Predicted value for {x_trained} years:", int(prediction[0]))

```

    Coefficient: [9449.96232146]
    Intercept: 24848.203966523208
    Predicted value for 5 years: 72098
    

    C:\Users\neeks\AppData\Local\Programs\Python\Python311\Lib\site-packages\sklearn\utils\validation.py:2691: UserWarning: X does not have valid feature names, but LinearRegression was fitted with feature names
      warnings.warn(
    
