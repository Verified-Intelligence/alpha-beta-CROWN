from setuptools import setup

setup(
    name='auto_LiRPA',
    version='0.1',
    description='A library for Automatic Linear Relaxation based Perturbation Analysis (LiRPA) on general computational graphs, with a focus on adversarial robustness verification and certification of deep neural networks.',
    url='https://github.com/KaidiXu/auto_LiRPA',
    author='Kaidi Xu, Zhouxing Shi, Huan Zhang, Yihan Wang, Shiqi Wang',
    author_email='xu.kaid@husky.neu.edu, zhouxingshichn@gmail.com, huan@huan-zhang.com, wangyihan617@gmail.com, sw3215@columbia.edu',
    packages=['auto_LiRPA'],
    install_requires=[
        'torch>=1.8,<1.9',
        'torchvision>=0.9,<0.10',
        'numpy>=1.16',
        'packaging>=20.0',
        'pytest>=5.0',
        'appdirs>=1.4',
    ],
    platforms=['any'],
    license='BSD',
)
