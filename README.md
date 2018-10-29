
## Applying gradient descent

### Introduction

In the last lesson, we derived the functions that we help us descend along our cost functions efficiently.  Remember that this technique is not so different from what we saw with using the derivative to tell us our next step size and direction in two dimensions.  

![](./slopes.png)

When descending along our cost curve in two dimensions, we used the slope of the tangent line at each point, to tell us how large of a step to take next.  And with the our cost curve being a function of $m$ and $b$, we had to use the gradient to determine each step.  

![](./gradientdescent.png)

But really it's an analogous approach.  Just like we can calculate the use derivative of a function $f(x)$ to calculate the slope at a given value of $x$ on the graph, and thus our next step.  Here, we calculated the partial derivative with respect to both variables, our slope and y-intercept, to calculate the amount to move next in either direction, and thus to steer us towards our minimum.   

## Reviewing our gradient descent formulas

Luckily for us, we already did the hard work of deriving these formulas.  Now we get to see the fruit of our labor.  The following formulas tell us how to update regression variables of $m$ and $b$ to approach a "best fit" line.   

* $ \frac{dJ}{dm}J(m,b) = -2\sum_{i = 1}^n x(y_i - (mx_i + b)) = -2\sum_{i = 1}^n x_i*\epsilon_i$ 
* $ \frac{dJ}{db}J(m,b) = -2\sum_{i = 1}^n(y_i - (mx_i + b)) = -2\sum_{i = 1}^n \epsilon_i $

Now the formulas above tell us to take some dataset, with values of $x$ and $y$, and then given a regression formula with values $m$ and $b$, iterate through our dataset, and use the formulas to calculate an update to $m$ and $b$.  So ultimately, to descend along the cost function, we will use the calculations:

`current_m` = `old_m` $ -  (-2*\sum_{i=1}^n x_i*\epsilon_i )$

`current_b` =  `old_b` $ - ( -2*\sum_{i=1}^n \epsilon_i )$

Ok let's turn this into code.  First, let's initialize some data.


```python
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(112)

X1 = np.random.rand(100,1).reshape(100)
X2 = np.random.rand(100,1).reshape(100)
y_randterm = np.random.normal(0,0.4,100)
y = 2+ 3* X1+ 7*X2 + y_randterm

plt.plot(X2,y, '.b')
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14);

data = np.array([X1,X2, y])
data = np.transpose(data)
data.shape;
```


![png](index_files/index_12_0.png)


Now we set our initial regression line by initializing $m$ and $b$ variables as zero.  Then to calculate our next step to update our regression line, we iterate through each of the points in the dataset, and at each iteration change our `update_to_b` by $2*\epsilon$ and change our `update_to_m` by $2*x*\epsilon$.


```python
# initial variables of our regression line
b_current = 0
m1_current = 0
m2_current= 0

#amount to update our variables for our next step
update_to_b = 0
update_to_m1 = 0 
update_to_m2 = 0

def error_at(point, b, m1, m2):
    return (m1 * point[0] + m2 * point[1] + b - point[2])

for i in range(0, len(data)):
    update_to_b += -2*(error_at(data[i], b_current, m1_current, m2_current))
    update_to_m1 += -2*(error_at(data[i], b_current, m1_current, m2_current)*data[i][0])
    update_to_m2 += -2*(error_at(data[i], b_current, m1_current, m2_current)*data[i][1])

new_b = b_current - update_to_b
new_m1 = m1_current - update_to_m1
new_m2 = m2_current - update_to_m2
```

In the last two lines of the code above, we calculate our `new_b` and `new_m` values by updating our taking our current values and adding our respective updates.  We define a function called `error_at`, which we can use in the error component of our partial derivatives above.

The code above represents **just one** update to our regression line, and therefore just one step towards our best fit line.  We'll just repeat the process to take multiple steps.  But first we have to make a couple other changes. 

## Tweaking our approach 

Ok, the above code is very close to what we want, but we just need to make tweaks to our code before it's perfect.

The first one is obvious if we think about what these formulas are really telling us to do.  Look at the graph below, and think about what it means to change each of our $m$ and $b$ variables by at least the sum of all of the errors, of the $y$ values that our regression line predicts and our actual data.  That would be an enormous change.  To ensure that we drastically updating our regression line with each step, we multiply each of these partial derivatives by a learning rate.  As we have seen before, the learning rate is just a small number, like $.0001$ which controls for how large our updates to the regression line will be.  The learning rate is  represented by the Greek letter eta, $\eta$, or alpha $\alpha$.  We'll use eta, so $\eta = .0001$ means the learning rate is $.0001$.

Multiplying our step size by our learning rate works fine, so long as we multiply both of the partial derivatives by the same amount.  This is because with out gradient,  $ \nabla J(m,b)$, we think of as steering us in the correct direction.  In other words, our derivatives ensure we are make the correct **proportional** changes to $m$ and $b$.  So scaling down these changes to make sure we don't update our regression line too quickly works fine, so long as we keep me moving in the correct direction.  While were at it, we can also get rid of multiplying our partials by 2.  As mentioned, so long as our changes are proportional we're in good shape. 

For our second tweak, note that in general the larger the dataset, the larger the sum of our errors would be.  But that doesn't mean our formulas are less accurate, and there deserve larger changes.  It just means that the total error is larger.  But we should really think accuracy as being proportional to the size of our dataset.  We can correct for this effect by dividing the effect of our update by the size of our dataset, $n$.

Making these changes, our formula looks like the following:


```python
#amount to update our variables for our next step
update_to_b = 0
update_to_m = 0 

learning_rate = .01
n = len(data)
for i in range(0, n):
    update_to_b += -2*(error_at(data[i], b_current, m1_current, m2_current))
    update_to_m1 += -2*(error_at(data[i], b_current, m1_current, m2_current)*data[i][0])
    update_to_m2 += -2*(error_at(data[i], b_current, m1_current, m2_current)*data[i][1])

new_b = b_current - update_to_b
new_m1 = m1_current - update_to_m1
new_m2 = m2_current - update_to_m2

```

So our code now reflects what we know about our gradient descent process.  Start with an initial regression line with values of $m$ and $b$.  Then for each point, calculate how the regression line fares against the actual point (that is, find the error).  Update what our next step to the respective variable should be using by using the partial derivative.  And after iterating through all of the points, update the value of $b$ and $m$ appropriately, scaled down by a learning rate.

### Seeing our gradient descent formulas in action

As mentioned earlier, the code above represents just one update to our regression line, and therefore just one step towards our best fit line.  To take multiple steps we wrap the process we want to duplicate in a function called `step_gradient` and then can call that function as much as we want. 


```python
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(112)

X1 = np.random.rand(100,1).reshape(100)
X2 = np.random.rand(100,1).reshape(100)
y_randterm = np.random.normal(0,0.4,100)
y = 2+ 3* X1+ 7*X2 + y_randterm

data = np.array([X1,X2, y])
data = np.transpose(data)

def step_gradient(b_current, m1_current, m2_current, points):
    b_gradient = 0
    m1_gradient = 0
    m2_gradient = 0
    learning_rate = .1
    N = float(len(points))
    for i in range(0, len(points)):
        x1 = points[i][0]
        x2 = points[i][1] 
        y = points[i][2]
        b_gradient += -(1/N) * (y - (m1_current * x1 + m2_current * x2 + b_current))
        m1_gradient += -(1/N) * x1 * (y -  (m1_current * x1 + m2_current * x2 + b_current))
        m2_gradient += -(1/N) * x2 * (y -  (m1_current * x1 + m2_current * x2 + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m1 = m1_current - (learning_rate * m1_gradient)
    new_m2 = m2_current - (learning_rate * m2_gradient)
    return {'b': new_b, 'm1': new_m1, 'm2': new_m2}
```


```python
b = 0
m1 = 0
m2 = 0

step_gradient(b, m1,m2, data) # {'b': 0.0085, 'm': 0.6249999999999999}
```




    {'b': 0.6852021393247482, 'm1': 0.3168360951144547, 'm2': 0.4125183248650799}



So just looking at input and output, we begin by setting $b$ and $m$ to 0, 0.  Then from our step_gradient function, we receive new values of $b$ and $m$ of .0085 and .6245.  Now what we need to do, is take another step in the correct direction by calling our step gradient function with our updated values of $b$ and $m$.


```python
updated_b = 0.6852021393247482
updated_m1 = 0.3168360951144547
updated_m2 = 0.4125183248650799
step_gradient(updated_b, updated_m1, updated_m2, data) # {'b': 0.01345805, 'm': 0.9894768333333332}
```




    {'b': 1.2674288943972738, 'm1': 0.5869077567342338, 'm2': 0.7690436910569831}



Let's do this, say, 10 times.


```python
# set our initial step with m and b values, and the corresponding error.
b = 0
m1 = 0
m2= 0
iterations = []
for i in range(700):
    iteration = step_gradient(b, m1,m2, data)
    # {'b': value, 'm': value}
    b = iteration['b']
    m1 = iteration['m1']
    m2 = iteration['m2'] 
    # update values of b and m
    iterations.append(iteration)
```

Let's take a look at these iterations.


```python
iterations[699]
```




    {'b': 2.037347658310951, 'm1': 2.7320590732571497, 'm2': 7.188143864955476}



As you can see, our  mm  and  bb  values both update with each step. Not only that, but with each step, the size of the changes to  mm  and  bb  decrease. This is because they are approaching a best fit line.

# MORE XES


```python
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(112)

X1 = np.random.rand(100,1).reshape(100)
X2 = np.random.rand(100,1).reshape(100)
y_randterm = np.random.normal(0,0.4,100)
y = 2+ 3* X1+ -4*X2 + y_randterm

plt.plot(X2,y, '.b')
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14);

data = np.array([X1,X2, y])
data = np.transpose(data)
data.shape;
```


![png](index_files/index_33_0.png)


Now we set our initial regression line by initializing $m$ and $b$ variables as zero.  Then to calculate our next step to update our regression line, we iterate through each of the points in the dataset, and at each iteration change our `update_to_b` by $2*\epsilon$ and change our `update_to_m` by $2*x*\epsilon$.


```python
# initial variables of our regression line
b_current = 0
m1_current = 0
m2_current= 0

#amount to update our variables for our next step
update_to_b = 0
update_to_m1 = 0 
update_to_m2 = 0

def error_at(point, b, m1, m2):
    return (m1 * point[0] + m2 * point[1] + b - point[2])

for i in range(0, len(data)):
    update_to_b += -2*(error_at(data[i], b_current, m1_current, m2_current))
    update_to_m1 += -2*(error_at(data[i], b_current, m1_current, m2_current)*data[i][0])
    update_to_m2 += -2*(error_at(data[i], b_current, m1_current, m2_current)*data[i][1])

new_b = b_current - update_to_b
new_m1 = m1_current - update_to_m1
new_m2 = m2_current - update_to_m2
```

In the last two lines of the code above, we calculate our `new_b` and `new_m` values by updating our taking our current values and adding our respective updates.  We define a function called `error_at`, which we can use in the error component of our partial derivatives above.

The code above represents **just one** update to our regression line, and therefore just one step towards our best fit line.  We'll just repeat the process to take multiple steps.  But first we have to make a couple other changes. 

### Tweaking our approach 

Ok, the above code is very close to what we want, but we just need to make tweaks to our code before it's perfect.

The first one is obvious if we think about what these formulas are really telling us to do.  Look at the graph below, and think about what it means to change each of our $m$ and $b$ variables by at least the sum of all of the errors, of the $y$ values that our regression line predicts and our actual data.  That would be an enormous change.  To ensure that we drastically updating our regression line with each step, we multiply each of these partial derivatives by a learning rate.  As we have seen before, the learning rate is just a small number, like $.0001$ which controls for how large our updates to the regression line will be.  The learning rate is  represented by the Greek letter eta, $\eta$, or alpha $\alpha$.  We'll use eta, so $\eta = .0001$ means the learning rate is $.0001$.

Multiplying our step size by our learning rate works fine, so long as we multiply both of the partial derivatives by the same amount.  This is because with out gradient,  $ \nabla J(m,b)$, we think of as steering us in the correct direction.  In other words, our derivatives ensure we are make the correct **proportional** changes to $m$ and $b$.  So scaling down these changes to make sure we don't update our regression line too quickly works fine, so long as we keep me moving in the correct direction.  While were at it, we can also get rid of multiplying our partials by 2.  As mentioned, so long as our changes are proportional we're in good shape. 

![](./regression-scatter.png)

For our second tweak, note that in general the larger the dataset, the larger the sum of our errors would be.  But that doesn't mean our formulas are less accurate, and there deserve larger changes.  It just means that the total error is larger.  But we should really think accuracy as being proportional to the size of our dataset.  We can correct for this effect by dividing the effect of our update by the size of our dataset, $n$.

Making these changes, our formula looks like the following:


```python
#amount to update our variables for our next step
update_to_b = 0
update_to_m = 0 

learning_rate = .01
n = len(data)
for i in range(0, n):
    update_to_b += -2*(error_at(data[i], b_current, m1_current, m2_current))
    update_to_m1 += -2*(error_at(data[i], b_current, m1_current, m2_current)*data[i][0])
    update_to_m2 += -2*(error_at(data[i], b_current, m1_current, m2_current)*data[i][1])

new_b = b_current - update_to_b
new_m1 = m1_current - update_to_m1
new_m2 = m2_current - update_to_m2
```

So our code now reflects what we know about our gradient descent process.  Start with an initial regression line with values of $m$ and $b$.  Then for each point, calculate how the regression line fares against the actual point (that is, find the error).  Update what our next step to the respective variable should be using by using the partial derivative.  And after iterating through all of the points, update the value of $b$ and $m$ appropriately, scaled down by a learning rate.

### Seeing our gradient descent formulas in action

As mentioned earlier, the code above represents just one update to our regression line, and therefore just one step towards our best fit line.  To take multiple steps we wrap the process we want to duplicate in a function called `step_gradient` and then can call that function as much as we want. 


```python
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(112)

X1 = np.random.rand(100,1).reshape(100)
X2 = np.random.rand(100,1).reshape(100)
y_randterm = np.random.normal(0,0.4,100)
y = 2+ 3* X1+ 7*X2 + y_randterm

data = np.array([X1,X2, y])
data = np.transpose(data)

def step_gradient(b_current, m1_current, m2_current, points):
    b_gradient = 0
    m1_gradient = 0
    m2_gradient = 0
    learning_rate = .1
    N = float(len(points))
    for i in range(0, len(points)):
        x1 = points[i][0]
        x2 = points[i][1] 
        y = points[i][2]
        b_gradient += -(1/N) * (y - (m1_current * x1 + m2_current * x2 + b_current))
        m1_gradient += -(1/N) * x1 * (y -  (m1_current * x1 + m2_current * x2 + b_current))
        m2_gradient += -(1/N) * x2 * (y -  (m1_current * x1 + m2_current * x2 + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m1 = m1_current - (learning_rate * m1_gradient)
    new_m2 = m2_current - (learning_rate * m2_gradient)
    return {'b': new_b, 'm1': new_m1, 'm2': new_m2}
```


```python
b = 0
m1 = 0
m2 = 0

step_gradient(b, m1,m2, data) # {'b': 0.0085, 'm': 0.6249999999999999}
```




    {'b': 0.6852021393247482, 'm1': 0.3168360951144547, 'm2': 0.4125183248650799}



So just looking at input and output, we begin by setting $b$ and $m$ to 0, 0.  Then from our step_gradient function, we receive new values of $b$ and $m$ of .0085 and .6245.  Now what we need to do, is take another step in the correct direction by calling our step gradient function with our updated values of $b$ and $m$.


```python
updated_b = 0.6852021393247482
updated_m1 = 0.3168360951144547
updated_m2 = 0.4125183248650799
step_gradient(updated_b, updated_m1, updated_m2, data) # {'b': 0.01345805, 'm': 0.9894768333333332}
```




    {'b': 1.2674288943972738, 'm1': 0.5869077567342338, 'm2': 0.7690436910569831}



Let's do this, say, 10 times.


```python
# set our initial step with m and b values, and the corresponding error.
b = 0
m1 = 0
m2= 0
iterations = []
for i in range(700):
    iteration = step_gradient(b, m1,m2, data)
    # {'b': value, 'm': value}
    b = iteration['b']
    m1 = iteration['m1']
    m2 = iteration['m2'] 
    # update values of b and m
    iterations.append(iteration)
```

Let's take a look at these iterations.


```python
iterations[599]
```




    {'b': 2.0700302247625215, 'm1': 2.7087343553944305, 'm2': 7.147311248751663}



As you can see, our $m$ and $b$ values both update with each step.  Not only that, but with each step, the size of the changes to $m$ and $b$ decrease.  This is because they are approaching a best fit line.

###  Animating these changes

We can use Plotly to see these changes to our regression line visually.  We'll write a method called `to_line` that takes a dictionary of $m$ and $b$ variables and changes it to produce a line object.  We can then see our line changes over time. 


```python
def to_line(m, b):
    initial_x = 0
    ending_x = 100
    initial_y = m*initial_x + b
    ending_y = m*ending_x + b
    return {'data': [{'x': [initial_x, ending_x], 'y': [initial_y, ending_y]}]}

frames = list(map(lambda iteration: to_line(iteration['m'], iteration['b']),iterations))
frames[0]
```




    {'data': [{'x': [0, 100], 'y': [0.2568596255859691, 15.809849090395078]}]}



Now we can see how our regression line changes, and approaches our data, with each iteration.


```python
from plotly.offline import init_notebook_mode, iplot
from IPython.display import display, HTML

init_notebook_mode(connected=True)

x_values_of_shows = list(map(lambda show: show['x'], shows))
y_values_of_shows = list(map(lambda show: show['y'], shows))
figure = {'data': [{'x': [0], 'y': [0]}, {'x': x_values_of_shows, 'y': y_values_of_shows, 'mode': 'markers'}],
          'layout': {'xaxis': {'range': [0, 110], 'autorange': False},
                     'yaxis': {'range': [0,160], 'autorange': False},
                     'title': 'Regression Line',
                     'updatemenus': [{'type': 'buttons',
                                      'buttons': [{'label': 'Play',
                                                   'method': 'animate',
                                                   'args': [None]}]}]
                    },
          'frames': frames}
iplot(figure)
```


<script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window.Plotly) {{require(['plotly'],function(plotly) {window.Plotly=plotly;});}}</script>



<div id="99d7c2d8-f079-49b6-9442-b977c6c73ba3" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";
        Plotly.plot(
            '99d7c2d8-f079-49b6-9442-b977c6c73ba3',
            [{"x": [0], "y": [0], "type": "scatter", "uid": "7542d1c6-d671-11e8-a6ab-acde48001122"}, {"mode": "markers", "x": [30, 40, 100], "y": [45, 60, 150], "type": "scatter", "uid": "7542d31a-d671-11e8-90e7-acde48001122"}],
            {"title": "Regression Line", "updatemenus": [{"buttons": [{"args": [null], "label": "Play", "method": "animate"}], "type": "buttons"}], "xaxis": {"autorange": false, "range": [0, 110]}, "yaxis": {"autorange": false, "range": [0, 160]}},
            {"showLink": true, "linkText": "Export to plot.ly"}
        ).then(function () {return Plotly.addFrames('99d7c2d8-f079-49b6-9442-b977c6c73ba3',[{"data": [{"x": [0, 100], "y": [0.2568596255859691, 15.809849090395078], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [0.48016109086238873, 29.763004641648585], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [0.6741829979433267, 42.094911262949346], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [0.8426590832820985, 53.01103852382516], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [0.9888476004185627, 62.690697900568466], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1155918676097685, 71.29037367755522], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2253731053970576, 78.94662969585256], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3203565458998157, 85.7786459602084], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4024316706060926, 91.89043223780824], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.473247324330222, 97.37275978027655], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5342423578001607, 102.30484706276663], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.586672368253451, 106.75583086330504], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.6316330349166877, 110.78605001690205], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.670080482971925, 114.44816569819531], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.7028490543989145, 117.78813904888551], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.7306668158984906, 120.84608431550375], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.7541690920544608, 123.65701334987395], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.7739102751977598, 126.25148530601177], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.7903741314155266, 128.65617360563013], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.8039827942042044, 130.894360707166], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.815104612880496, 132.98636987173862], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.824061001583936, 134.94994194877407], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.8311324161345466, 136.80056418242216], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.8365635698034608, 138.55175714838055], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.8405679849124312, 140.21532515274652], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.843331964836975, 141.80157474559775], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.8450180602182666, 143.31950540953164], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.8457680937906318, 144.7769759663694], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.8457058000300088, 146.18084979404347], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.8449391286716155, 147.53712155195188], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.8435622548993198, 148.85102776946863], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.8416573335587925, 150.12714335245417], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.8392960299901517, 151.36946580095028], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.83654085592511, 152.58148870290037], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.8334463352714834, 153.76626586947225], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.8300600214470057, 154.92646730366926], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.8264233851659897, 156.06442804216715], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.8225725891752171, 157.18219077789098], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.8185391643348163, 158.2815430552831], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.8143505996067417, 159.36404972936862], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.8100308569137662, 160.43108129171898], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.8056008204358904, 161.48383858961682], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.8010786886928352, 162.52337439770596], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.796480316698176, 163.5506122429268], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.7918195145429427, 164.56636283249824], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.78710830795691, 165.57133839017078], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.782357165689299, 166.56616516710767], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.7775751979340766, 167.55139435983358], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.7727703294869996, 168.52751163809126], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.7679494508520424, 169.49494545961875], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.7631185501051134, 170.45407432631671], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.7582828279654095, 171.40523311660775], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.7534467982127329, 172.34871861162281], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.748614375316806, 173.2847943178709], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.7437889509069986, 174.21369467597535], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.7389734605035203, 175.1356287336536], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.7341704417501806, 176.05078335116187], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.7293820852308999, 176.95932599873862], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.724610278814355, 177.86140719800068], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.7198566463508853, 178.75716265262867], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.715122581440841, 179.6467151079073], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.7104092769019756, 180.53017597364567], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.7057177504835672, 181.40764674060813], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.7010488673052129, 182.27922021674837], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.6964033594373762, 183.14498160619175], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.691781842987661, 184.00500945098932], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.6871848330104329, 184.8593764531166], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.682612756516969, 185.70815019196567], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.6780659638280164, 186.55139375063766], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.6735447384798396, 187.38916626264708], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.6690493058679636, 188.22152338917232], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.6645798407893535, 189.04851773569533], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.6601364740233104, 189.87019921574722], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.6557192980734994, 190.68661536849442], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.6513283721779308, 191.49781163604268], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.646963726680124, 192.30383160558682], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.6426253668428, 193.10471722088187], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.6383132761750987, 193.90050896694154], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.6340274193352744, 194.69124603137183], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.6297677446629293, 195.4769664453143], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.6255341863879673, 196.25770720659432], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.621326666557439, 197.03350438733966], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.6171450967162055, 197.8043932280456], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.612989379372775, 198.57040821981184], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.6088594092776733, 199.33158317625606], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.6047550745382257, 200.0879512964184], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.6006762575905844, 200.83954521980223], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.596622836047187, 201.58639707455262], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5925946834355125, 202.3285385196447], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5885916698419806, 203.06600078184385], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5846136624730813, 203.79881468810333], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5806605261442757, 204.52701069397818], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5767321237058736, 205.25061890856279], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5728283164139159, 205.96966911639353], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5689489642530707, 206.6841907967021], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5650939262176584, 207.39421314035584], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5612630605561413, 208.09976506477923], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5574562249837363, 208.80087522711193], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.553673276867213, 209.49757203582814], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5499140733854258, 210.18988366101115], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5461784716686717, 210.87783804345455], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5424663289195781, 211.56146290273804], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5387775025178738, 212.24078574440773], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.535111850111103, 212.91583386637407], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5314692296930739, 213.5866343646264], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5278494996716114, 214.25321413835016], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.524252518926978, 214.91559989452185], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5206781468621557, 215.57381815204798], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5171262434460313, 216.2278952455044], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5135966692503917, 216.8778573285271], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5100892854815229, 217.52373037689713], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5066039540071023, 218.1655401913584], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.5031405373789908, 218.80331240020155], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.499698898852448, 219.43707246164234], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4962789024022325, 220.06684566602087], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.492880412735987, 220.69265713784253], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4895032953052583, 221.31453183768136], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4861474163144588, 221.9324945639613], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4828126427280326, 222.54656995463125], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.479498842276063, 223.1567824887457], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4762058834585206, 223.76315648796287], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4729336355483305, 224.36571611797012], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.469681968593414, 224.96448538984473], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4664507534178373, 225.5594881613577], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4632398616221862, 226.15074813822747], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4600491655832697, 226.73828887532855], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.45687853845324, 227.32213377786084], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.453727854158209, 227.90230610248298], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4505969873964297, 228.47882895841474], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4474858136360982, 229.05172530851064], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4443942091128332, 229.62101797030834], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4413220508268747, 230.18672961705428], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4382692165400421, 230.7488827787084], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.435235584772487, 231.30749984293047], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4322210347992692, 231.8626030560492], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4292254466467835, 232.41421452401605], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4262487010890588, 232.9623562133442], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4232906796439502, 233.50704995203543], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4203512645692404, 234.04831743049394], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4174303388586682, 234.58618020242952], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4145277862378924, 235.12065968575013], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4116434911604077, 235.65177716344456], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4087773388034186, 236.1795537844561], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.405929215063683, 236.70401056454742], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4030990065533304, 237.22516838715734], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.4002866005956631, 237.7430480042496], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3974918852209455, 238.25767003715453], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3947147491621863, 238.76905497740307], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.391955081850918, 239.27722318755454], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.389212773412978, 239.7821949020173], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3864877146642924, 240.28399022786334], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3837797971066697, 240.78262914563655], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.381088912923601, 241.27813151015482], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3784149549760742, 241.77051705130683], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3757578167984006, 242.2598053748423], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3731173925940559, 242.74601596315745], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.370493577231539, 243.2291681760744], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3678862662402478, 243.70928125161555], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3652953558063723, 244.18637430677242], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3627207427688086, 244.6604663382695], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.360162324615092, 245.13157622332298], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3576199994773512, 245.59972272039417], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.355093666128283, 246.06492446993832], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3525832239771494, 246.52719999514812], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.350088573065795, 246.98656770269278], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3476096140646894, 247.44304588345196], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3451462482689887, 247.89665271324498], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3426983775946215, 248.34740625355562], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3402659045743983, 248.7953244522518], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3378487323541408, 249.24042514430104], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3354467646888373, 249.68272605248112], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3330599059388188, 250.12224478808625], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.33068806106596, 250.55899885162876], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3283311356299004, 250.99300563353634], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3259890357842914, 251.42428241484478], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3236616682730646, 251.85284636788631], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.321348940426723, 252.27871455697374], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3190507601586545, 252.7019039390799], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3167670359614707, 253.12243136451335], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3144976769033645, 253.5403135775892], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3122425926244927, 253.95556721729608], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3100016933333811, 254.36820881795884], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3077748898033503, 254.7782548098969], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3055620933689656, 255.18572152007872], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3033632159225073, 255.5906251727718], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.3011781699104643, 255.99298189018887], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2990068683300493, 256.3928076931299], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2968492247257353, 256.79011850162004], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.294705153185814, 257.18493013554354], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2925745683389762, 257.5772583152738], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2904573853509127, 257.96711866229924], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2883535199209377, 258.35452669984534], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.286262888278632, 258.7394978534929], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2841854071805083, 259.122047451792], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.282120993906697, 259.5021907268727], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2800695662576531, 259.8799428150515], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2780310425508827, 260.25531875743354], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.276005341617692, 260.62833350051244], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2739923827999549, 260.9990018967645], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2719920859469018, 261.36733870524085], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2700043714119282, 261.7333585921545], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.268029160049424, 262.09707613146486], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2660663732116215, 262.45850580545766], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.264115932745465, 262.81766200532184], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.262177760989498, 263.17455903172237], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2602517807707716, 263.5292110953699], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.258337915401772, 263.8816323175863], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.256436088677366, 264.23183673086737], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2545462248717683, 264.5798382794411], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2526682487355247, 264.9256508198233], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2508020854925175, 265.2692881213691], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2489476608369867, 265.61076386682106], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2471049009305726, 265.9500916528546], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2452737323993748, 266.2872849906186], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2434540823310314, 266.622357306274], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2416458782718156, 266.95532194152804], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2398490482237505, 267.2861921541656], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2380635206417427, 267.61498111857713], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2362892244307335, 267.9417019262832], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2345260889428677, 268.2663675864558], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2327740439746817, 268.5889910264361], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.231033019764307, 268.9095850922496], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2293029469886925, 269.2281625491173], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2275837567608454, 269.5447360819642], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2258753806270875, 269.8593182959239], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2241777505643292, 270.1719217168413], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2224907989773617, 270.4825587917708], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2208144586961651, 270.79124188947185], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2191486629732342, 271.0979833009016], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2174933454809207, 271.40279523970463], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.215848440308792, 271.7056898426986], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2142138819610069, 272.006679170358], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.212589605353708, 272.3057752072942], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2109755458124298, 272.6029898627327], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2093716390695242, 272.8983349709872], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2077778212616004, 273.191822291931], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2061940289269832, 273.48346351146506], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2046201990031855, 273.77327024198337], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.2030562688243973, 274.0612540228355], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.201502176118991, 274.34742632078564], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1999578590070414, 274.63179853046967], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1984232559978618, 274.91438197484865], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1968983059875562, 275.19518790565957], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1953829482565854, 275.47422750386346], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1938771224673501, 275.7515118800907], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.192380768661788, 276.02705207508296], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1908938272589866, 276.30085906013363], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1894162390528107, 276.5729437375239], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.187947945209545, 276.84331694095715], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1864888872655515, 277.11198943599044], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.185039007124942, 277.378971920463], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.183598247057264, 277.64427502492225], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1821665496952027, 277.90790931304696], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.180743858032297, 278.16988528206826], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1793301154206692, 278.43021336318697], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1779252655687693, 278.68890392198955], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1765292525391338, 278.9459672588606], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1751420207461585, 279.2014136093928], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1737635149538852, 279.45525314479465], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1723936802738026, 279.70749597229576], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1710324621626602, 279.95815213554863], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.169679806420297, 280.2072316150291], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.168335659187483, 280.45474432843355], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1669999669437752, 280.7007001310737], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.165672676505386, 280.9451088162694], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1643537350230662, 281.18798011573796], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1630430899799997, 281.4293236999823], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1617406891897135, 281.6691491786755], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.160446480793999, 281.90746610104384], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1591604132608475, 282.1442839562465], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1578824353823984, 282.379612173754], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1566124962729, 282.6134601237232], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1553505453666835, 282.8458371173707], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.154096532416149, 283.07675240734324], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1528504074897663, 283.3062151880865], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.151612120970085, 283.53423459621104], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1503816235517599, 283.7608197108559], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1491588662395877, 283.98597955405046], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.147943800346556, 284.20972309107344], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1467363774919044, 284.43205923080984], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1455365495991985, 284.65299682610606], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1443442688944154, 284.8725446741218], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1431594879040419, 285.09071151668064], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1419821594531836, 285.3075060406184], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.140812236663687, 285.5229368781285], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1396496729522725, 285.737012607106], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1384944220286801, 285.949741751489], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1373464378938258, 286.1611327815982], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1362056748379705, 286.37119411447406], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1350720874388993, 286.5799341142119], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1339456305601134, 286.787361092295], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1328262593490328, 286.99348330792554], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1317139292352103, 287.1983089683536], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1306085959285568, 287.4018462292036], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.129510215417578, 287.60410319479934], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1284187439676219, 287.8050879184866], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1273341381191369, 288.00480840295376], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1262563546859419, 288.2032726005506], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.125185350753507, 288.4004884136046], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1241210836772442, 288.59646369473614], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1230635110808092, 288.79120624717046], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.122012590854415, 288.98472382504906], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1209682811531543, 289.17702413373814], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1199305403953337, 289.36811483013514], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.118899327260818, 289.5580035229741], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1178746006893856, 289.74669777312846], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.116856319879092, 289.9342050939119], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1158444442846467, 290.1205329513777], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.114838933615798, 290.3056887646161], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1138397478357291, 290.4896799060495], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.112846847159464, 290.672513701726], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1118601920522826, 290.8541974316114], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1108797432281474, 291.0347383298788], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1099054616481392, 291.2141435851966], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.108937308518902, 291.39242034101505], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.107975245291099, 291.56957569585046], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.107019233657877, 291.74561670356775], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1060692355533424, 291.9205503736614], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1051252131510438, 292.09438367153484], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1041871288624672, 292.2671235187773], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1032549453355385, 292.4387767934398], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1023286254531366, 292.60935033030876], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1014081323316158, 292.77885092117833], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.1004934293193367, 292.9472853151208], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.0995844799952068, 293.11466021875503], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.0986812481672306, 293.280982296514], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.0977836978710682, 293.4462581709099], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.096891793368604, 293.6104944227976], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.0960054991465227, 293.773697591637], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.095124779914896, 293.93587417575344], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.0942496006057771, 294.0970306325963], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.093379926371805, 294.25717337899596], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.0925157225848154, 294.41630879141974], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.091656954834464, 294.57444320622545], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.0908035889268546, 294.7315829199136], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.0899555908831788, 294.8877341893786], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.0891129269383624, 295.04290323215747], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.088275563539721, 295.1970962266774], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.0874434673456244, 295.35031931250205], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.0866166052241684, 295.50257859057564], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.0857949442518562, 295.6538801234663], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.0849784517122874, 295.80422993560705], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.0841670950948548, 295.9536340135361], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.0833608420934504, 296.1020983061346], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.0825596606051788, 296.24962872486435], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.0817635187290793, 296.39623114400246], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.080972384764855, 296.5419114008755], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.0801862272116114, 296.68667529609206], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.0794050147666017, 296.8305285937735], "type": "scatter"}]}, {"data": [{"x": [0, 100], "y": [1.07862871632398, 296.9734770217836], "type": "scatter"}]}]);}).then(function(){Plotly.animate('99d7c2d8-f079-49b6-9442-b977c6c73ba3');})
        });</script>


As you can see, our regression line starts off far away from our cost curve.  But it uses our `step_gradient` function to move closer to finding the line that produces the lowest error.

### Summary

In this section, we saw our gradient descent formulas in action.  The core of the gradient descent functions are understanding the two lines: 

$$ \frac{dJ}{dm}J(m,b) = -2\sum_{i = 1}^n x(y_i - (mx_i + b)) = -2\sum_{i = 1}^n x_i*\epsilon_i$$
$$ \frac{dJ}{db}J(m,b) = -2\sum_{i = 1}^n(y_i - (mx_i + b)) = -2\sum_{i = 1}^n \epsilon_i $$
    
Which both look to the errors of the current regression line for our dataset to determine how to update the regression line next.  These formulas came from our cost function, $J(m,b) = \sum_{i = 1}^n(y_i - (mx_i + b))^2 $, and using the gradient to find the direction of steepest descent.  Translating this into code, and seeing how the regression line continued to improve in alignment with the data, we saw the effectiveness of this technique in practice.  
