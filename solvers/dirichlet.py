from boundaryCondition import boundaryCondition

class dirichlet(boundaryCondition):
  def __init__(self,*args):
    super(dirichlet,self).__init__("dirichlet",args[0])
