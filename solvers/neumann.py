from boundaryCondition import boundaryCondition

class neumann(boundaryCondition):
  def __init__(self,*args):
    super(neumann,self).__init__("neumann",args[0])
