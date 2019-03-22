from neupy import algorithms

def grnnet(target, x, y)
grnn = algorithms.GRNN(std=0.1)
grnn.train(x, y)

return grnn.predict(target)