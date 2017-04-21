from setuptools import setup, find_packages

setup(name='gym_ctf',
      version='0.0.1',
      install_requires=['gym'],
      py_modules=['gym_ctf.state.flag', 'gym_ctf.state.world',
                  'gym_ctf.state.agent', 'gym_ctf.state.team']
)
