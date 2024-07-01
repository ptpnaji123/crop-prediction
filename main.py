from model import Model

model = Model()
model.load()

print(model.encoder.categories_)

area = input("Country: ")
area = area.strip()

item = input("Item: ")
item = item.strip()

rainfall = float(input("Average Rainfall Per Year (mm): "))
pesticides = float(input("Pesticides (tonn): "))
temperature = float(input("Average Temperature (C): "))

production = model.predict(area, item, rainfall, pesticides, temperature)

print("Production =", production)