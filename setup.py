from setuptools import setup

setup(
    name='auto_LiRPA',
    version='0.1',
    description='Automatic Linear Relaxation based Perturbation Analysis (LiRPA) for general computational graphs',
    url='https://github.com/KaidiXu/auto_LiRPA',
    author='Kaidi Xu, Zhouxing Shi, Huan Zhang',
    author_email='xu.kaid@husky.neu.edu, zhouxingshichn@gmail.com, huan@huan-zhang.com',
    packages=['auto_LiRPA'],
    install_requires=[
        'torch>=1.8',
        'numpy>=1.16',
        'packaging>=20.0',
        'pytest>=5.0',
        'appdirs>=1.4',
        'oslo.concurrency>=4.2',
    ],
    platforms=['any'],
    license='BSD',
)
