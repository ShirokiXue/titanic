# titanic
A deep learning homework.

https://www.kaggle.com/c/titanic

# Kaggle Titanic - Machine Learning from Disaster
###### tags: `Deep Learning`


## Preprocessing

### step 1
除去認為用不到的 features
- Name
- Cabin
- Ticket
- PassengerId
- Embarked

### step 2
除去 features 有缺漏的 instance

### step 3
整理留下來的 features
- Age:Normalization
- Fare:Normalization
- Sex:One hot encoding
- Pclass:One hot encoding

### step 4
隨機抽取兩成資料作為validation set

```python=
df = pd.read_csv("train.csv")

df = df.drop(['Name', 'Cabin', 'Ticket', 'PassengerId', 'Embarked'], axis=1)
df = df.dropna()
df['Age'] = (df['Age'] - df['Age'].min())/( df['Age'].max() - df['Age'].min())
df['Fare'] = (df['Fare'] - df['Fare'].min())/( df['Fare'].max() - df['Fare'].min())
df["Sex"] = df["Sex"].map({'male':0, 'female':1})
df['Sex'] = df['Sex'].astype('object')
df['Pclass'] = df['Pclass'].astype('object')
df = pd.get_dummies(df,prefix=["Sex", "Pclass"])
print(df)
train, vali = train_test_split(df, test_size=0.2, random_state=42)

X_train = train.drop(['Survived'], 1)
X_train = X_train.to_numpy()
X_vali = vali.drop(['Survived'], 1)
X_test = X_vali.to_numpy()
y_train = train['Survived']
y_vali = vali['Survived']
```

## Model
- loss: sparse_categorical_crossentropy
- optimizer: adam
- activation function: relu
- epoch: 15

```
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
batch_normalization_5 (Batch (None, 9)                 36        
_________________________________________________________________
dense_5 (Dense)              (None, 32)                320       
_________________________________________________________________
batch_normalization_6 (Batch (None, 32)                128       
_________________________________________________________________
dense_6 (Dense)              (None, 64)                2112      
_________________________________________________________________
batch_normalization_7 (Batch (None, 64)                256       
_________________________________________________________________
dense_7 (Dense)              (None, 128)               8320      
_________________________________________________________________
batch_normalization_8 (Batch (None, 128)               512       
_________________________________________________________________
dense_8 (Dense)              (None, 256)               33024     
_________________________________________________________________
batch_normalization_9 (Batch (None, 256)               1024      
_________________________________________________________________
dense_9 (Dense)              (None, 2)                 514       
=================================================================
Total params: 46,246
Trainable params: 45,268
Non-trainable params: 978
_________________________________________________________________

```

## Kaggle prediction
![](https://i.imgur.com/Ld83Eu6.png)
