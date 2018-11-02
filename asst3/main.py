import Data_Utils
from sklearn import svm

from evaluate import get_accuracy
from evolve import ElitistGA, SteadyStateGA, EstimationGA

rbfsvm = svm.SVC(gamma='auto')

CU_X, Y = Data_Utils.Get_Casis_CUDataset()
ga = SteadyStateGA(pop_size=100)
for i in range(8):
    ga.evolve(CU_X, Y, get_accuracy, rbfsvm, i)
# print(get_accuracy(rbfsvm, CU_X, Y))