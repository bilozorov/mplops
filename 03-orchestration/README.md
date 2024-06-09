## Homework

The goal of this homework is to train a simple model for predicting the duration of a ride, but use Mage for it.

We'll use [the same NYC taxi dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page), the **Yellow** taxi data for 2023. 

---

## Question 1. Run Mage

First, let's run Mage with Docker Compose. Follow the quick start guideline. 

What's the version of Mage we run? 

(You can see it in the UI)

### Answer 1 = v0.9.71

![Question 1. Run Mage](https://raw.githubusercontent.com/bilozorov/mplops/main/03-orchestration/imgs/Question%201.%20Run%20Mage.png)

---

## Question 2. Creating a project

Now let's create a new project. We can call it "homework_03", for example.

How many lines are in the created `metadata.yaml` file? 

- 35
- 45
- 55
- 65

### Answer 2 = 55

![Question 2. Creating a project](https://raw.githubusercontent.com/bilozorov/mplops/main/03-orchestration/imgs/Question%202.%20Creating%20a%20project.png)

---

## Question 3. Creating a pipeline

Let's create an ingestion code block.

In this block, we will read the March 2023 Yellow taxi trips data.

How many records did we load? 

- 3,003,766
- 3,203,766
- 3,403,766
- 3,603,766

### Answer 3 = 3,403,766

![Question 3. Creating a pipeline](https://raw.githubusercontent.com/bilozorov/mplops/main/03-orchestration/imgs/Question%203.%20Creating%20a%20pipeline.png)

---

## Question 4. Data preparation

Let's use the same logic for preparing the data we used previously. We will need to create a transformer code block and put this code there.

This is what we used (adjusted for yellow dataset):

```python
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    
    return df
```

Let's adjust it and apply to the data we loaded in question 3. 

What's the size of the result? 


- 2,903,766
- 3,103,766
- 3,316,216 
- 3,503,766

### Answer 4 = 3,316,216

![Question 4. Data preparation](https://raw.githubusercontent.com/bilozorov/mplops/main/03-orchestration/imgs/Question%204.%20Data%20preparation.png)

---

## Question 5. Train a model

We will now train a linear regression model using the same code as in homework 1

* Fit a dict vectorizer
* Train a linear regression with default parameres 
* Use pick up and drop off locations separately, don't create a combination feature

Let's now use it in the pipeline. We will need to create another transformation block, and return both the dict vectorizer and the model

What's the intercept of the model? 

Hint: print the `intercept_` field in the code block

- 21.77
- 24.77
- 27.77
- 31.77

### Answer 1 = 24.77

![Question 5. Train a model](https://raw.githubusercontent.com/bilozorov/mplops/main/03-orchestration/imgs/Question%205.%20Train%20a%20model.png)

---

## Question 6. Register the model 

The model is trained, so let's save it with MLFlow.

Find the logged model, and find MLModel file. What's the size of the model? (`model_size_bytes` field):

* 14,534
* 9,534
* 4,534
* 1,534

> Note: typically we do two last steps in one code block 

### Answer 1 = 4,534

![Question 6. Register the model](https://raw.githubusercontent.com/bilozorov/mplops/main/03-orchestration/imgs/Question%206.%20Register%20the%20model.png)

---





