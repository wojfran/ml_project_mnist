from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

mnist = fetch_openml('mnist_784', version=1)

data = pd.concat([pd.DataFrame(mnist.data), pd.DataFrame(mnist.target)], axis=1)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(data, data["class"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]

x_train = strat_train_set.drop(columns=["class"])
y_train = strat_train_set["class"]

x_test = strat_test_set.drop(columns=["class"])
y_test = strat_test_set["class"]



# x, y = mnist["data"], mnist["target"]

# x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

# digit = x_train.iloc[0].to_numpy()

# digit_image = digit.reshape(28, 28)

# plt.imshow(digit_image, cmap="binary")
# plt.axis("off")
# plt.show()

