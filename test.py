import yaml

file = open("yaml_test.yaml")
data = yaml.load(file, Loader=yaml.FullLoader)
print(data["extractor"]["data1"])
