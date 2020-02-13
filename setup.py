from setuptools import setup

setup(
    name='Thesis',
    version='',
    packages=['RL.nets', 'RL.utils', 'RL.problems', 'RL.problems.op', 'RL.problems.op.opga', 'RL.problems.tsp',
              'RL.problems.vrp', 'RL.problems.pctsp', 'RL.problems.pctsp.salesman', 'RL.problems.pctsp.salesman.pctsp',
              'RL.problems.pctsp.salesman.pctsp.algo', 'RL.problems.pctsp.salesman.pctsp.model',
              'RL.problems.pctsp.salesman.pctsp.model.tests', 'Simulation.Anticipitory', 'RL_anticipatory',
              'RL_anticipatory.nets', 'RL_anticipatory.utils', 'RL_anticipatory.problems', 'MixedIntegerOptimization'],
    url='',
    license='',
    author='chanaross',
    author_email='chanaby@gmail.com',
    description='all python files needed for running RL_anticipatory and anticipatory simulation'
)
