<p align="center"><img src = "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/ASHRAE_Logo.svg/1200px-ASHRAE_Logo.svg.png" width="300" /></p>
# Great Energy Predictor
### How much energy will a building consume? <br>
Founded in 1894, ASHRAE serves to advance the arts and sciences of heating, ventilation, air conditioning refrigeration and their allied fields. ASHRAE members represent building system design and industrial process professionals around the world. With over 54,000 members serving in 132 countries, ASHRAE supports research, standards writing, publishing and continuing education - shaping tomorrow’s built environment today.<br>
We aim to develop accurate models of metered building energy usage in the following areas: chilled water, electric, hot water, and steam meters. The data comes from over 1,000 buildings over a three-year timeframe. With better estimates of these energy-saving investments, large scale investors and financial institutions will be more inclined to invest in this area to enable progress in building efficiencies.

The dataset can be downloaded [from](https://www.kaggle.com/c/ashrae-energy-prediction)

# Understanding The Data:
## train.csv

- building_id - Foreign key for the building metadata.
- meter - The meter id code. Read as {0: electricity, 1: chilledwater, 2: steam, 3: hotwater}. Not every building has all meter types.
- timestamp - When the measurement was taken
- meter_reading - The target variable. Energy consumption in kWh (or equivalent). Note that this is real data with measurement error, which we expect will impose a baseline level of modeling error. UPDATE: as discussed here, the site 0 electric meter readings are in kBTU.

![No building pts per building id](https://user-images.githubusercontent.com/37273226/79433841-3e733080-7feb-11ea-84cf-1a5c521ef605.PNG)
<br>

On average each building has 13951.75983436853 datapoints
Building 403 has least no. of datapoints 479

![Meters count](https://user-images.githubusercontent.com/37273226/79433882-4b901f80-7feb-11ea-8978-5eb5558656a7.PNG)
<br>
We can see that maximum datapoints are for Meter 0. Meter 0 has more data points than 1,2,3 combined.

## building_meta.csv
- site_id - Foreign key for the weather files.
- building_id - Foreign key for training.csv
- primary_use - Indicator of the primary category of activities for the building based on EnergyStar property type definitions
- square_feet - Gross floor area of the building
- year_built - Year building was opened
- floor_count - Number of floors of the building

![primary use](https://user-images.githubusercontent.com/37273226/79433923-55b21e00-7feb-11ea-8130-e7db9f01c036.PNG)
<br>
We can see that most data points are for building related to Education, followed by Offices and Public Entertainment.

## weather_train.csv
Weather data from a meteorological station as close as possible to the site.

- site_id
- air_temperature - Degrees Celsius
- cloud_coverage - Portion of the sky covered in clouds, in oktas
- dew_temperature - Degrees Celsius
- precip_depth_1_hr - Millimeters
- sea_level_pressure - Millibar/hectopascals
- wind_direction - Compass direction (0-360)
- wind_speed - Meters per second

![wind speed](https://user-images.githubusercontent.com/37273226/79433949-619de000-7feb-11ea-944a-127c97a672d3.PNG)<br>
We can see that the wind_speed data is quite discrete. Later in Preprocessing we use this to our advantage and convert this data to [Beaufort Scale](https://en.wikipedia.org/wiki/Beaufort_scale). <br>

For more visualizations, correlation views etc visit the Project Notebook.

## Data PreProcessing
![Image for Data PreProcessing](https://miro.medium.com/max/8332/1*wK8k8Vo8_c6jdYIjUWL_Pw.png)
Need of Data Preprocessing<br>
- For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.<br>
- Another aspect is that data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithms are executed in one data set, and best out of them is chosen.
<br>
  For our data we've applied **Feature engineering** across timestamp data and wind speed data and **Dropped insignificant columns**. ALong with this we have impleented **Memory Reduction** to reduce our dataset size by **65%**. All can be seen and understood in our Notebook File.
<br>

## Building a Neural Network

<p align = "center"><img src = "https://media.giphy.com/media/9EvzNG9HAVc64/giphy.gif"/></p>
<br>
Here we will be using the Keras framework to build a Neural Network.<br>Keras is an open-source neural-network library written in Python. It is capable of running on top of TensorFlow, Microsoft Cognitive Toolkit, R, Theano, or PlaidML. Designed to enable fast experimentation with deep neural networks, it focuses on being user-friendly, modular, and extensible.

### Following is our model architecture:-
![Model](https://user-images.githubusercontent.com/37273226/79434006-6d89a200-7feb-11ea-92aa-578d0ceee485.PNG)<br>
We Built Separate Models and their Training loss trends for separate Meters which can be seen in our Notebook.

## Results

__ | Linear Regression       | Multivariate Polynomial Regression | Neural Network Non-Mean Imputed| Neural Network Mean Imputed 
------------ | ----------------- | ---------------- | ----------------- |----------------
Meter 0(Electric) | r2 = 0.3334<br>mse = 98387.13<br>mae = 134.24 | r2 = 0.3341<br>mse = 98300.27<br>mae = 133.74 |  r2 = 0.7225<br>mse = 40603.1<br>mae = 45.9878 | r2 = 0.7566<br>mse = 35621.7<br>mae = 38.6521
Meter 1 (Chilled Water)|  NA | NA | r2 = 0.012<br>mse = 6.369e7<br>mae = 379.935 | r2 = 0.0085<br>mse = 6.393e7<br>mae = 432.62
Meter 2 (Stream)|  NA | NA | r2 = 0.0031<br>mse = 1.86e11<br>mae = 13626 | r2 = 0.0028<br>mse = 1.86e11<br>mae = 13680.9
Meter 3 (Hot Water)| NA | NA |  r2 = 0.0273<br>mse = 6.258e6<br>mae = 294.626 | r2 = 0.0389<br>mse = 6.183e6<br>mae = 280.508

# Inferences

- We see that models trained on imputed data perform better. 
- Our model for meter 0 works well and gives good predictions.
- Remaining models do not perform that well. Probably using a different network architecture would result in better performance.
- It is also possible that data for meter 1,2 and 3 is insufficient. So a deeper network may fit the data better.
- The electric meter has a better model because of adequate amount of data.
- The neural network with imputed values i.e. NaN values filled with the mean performs better than the non-imputed neural network.
- Meter reading is better correlated with square feet, than other parameter.
- The graphs of each parameter with meter reading seems to fall in an area, and isn’t linearly related.

# References

https://keras.io/models/sequential/
https://en.wikipedia.org/wiki/Beaufort_scale
https://medium.com/@satnalikamayank12/on-learning-embeddings-for-categorical-data-using-keras-165ff2773fc9
https://machinelearningmastery.com/introduction-to-regularization-to-reduce-overfitting-and-improve-generalization-error/
https://scikit-learn.org/stable/supervised_learning.html#supervised-learning
https://towardsdatascience.com/deep-neural-networks-for-regression-problems-81321897ca33
