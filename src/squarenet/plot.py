from squarenet import SquareNet
import shapely
IJ = [(400, 400), (100,100,100)]
method = [["test", "ball", "france", "germany"], ["test", "ball"]]

for IJ_, meth in zip(IJ, method):
    sqnet = SquareNet(IJplus = IJ_)
    sqnet.fit(meth)
    board = sqnet.checkerboard()