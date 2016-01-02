from setuptools import setup

setup(name='stanmo',
      version='0.1',
      description='Execution engine for standard business models.',
      url='http://github.com/storborg/funniest',
      author='Duan Qiyang',
      author_email='qiyang.duan@qq.com',
      license='MIT',
      packages=['stanmo'
               ,'stanmo.app'
               ,'stanmo.config'
               ,'stanmo.data'
               ,'stanmo.log'
               ,'stanmo.model'
               ,'stanmo.spec'
               ,'stanmo.test'
                ],
      install_requires=[
          'simplejson',
          'flask',
          'docopt'
      ],
      test_suite='nose.collector',
      tests_require=['nose', 'nose-cover3', 'simplejson',
                    'flask',
                    'docopt',
                    'pandas',
                    'sklearn',
                    'scipy'],
      entry_points={
          'console_scripts': ['stanmo=stanmo.stanmoctl:main'],
      },
      zip_safe=False)