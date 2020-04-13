import pandas as pd 


data = pd.read_csv("../data/banking_data/test.csv")

print(data.head())

data["text"].to_csv("../data/banking_data/only_messages_test.csv")