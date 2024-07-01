from model import Model
from time import time

model = Model()

start = time()
model.train("data/dataset.csv")
end = time()
model.save()
print("Model saved")

training_time = end - start

print("Training time:", training_time)