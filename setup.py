from setuptools import find_packages
from setuptools import setup

setup(
    name='tc_study',
    version='1.0',
    description=('Experiment on disentangled latent representations.'),
    author='TCStudy Authors',
    url='https://github.com/bonheml/tc_study',
    license='MIT',
    packages=find_packages(),
    include_package_data=True,
    scripts=[
        'bin/tcs_truncation_experiment',
        'bin/tcs_aggregate_results',
        'bin/tcs_visualize_results',
        'bin/tcs_download_models'
    ],
    install_requires=[
        'tensorboard==1.14.0',
        'tensorflow-estimator==1.14.0',
        'tensorflow-hub==0.9.0',
        'tensorflow-probability==0.7.0',
        'gin-config==0.3.0',
        'disentanglement-lib @ git+https://github.com/google-research/disentanglement_lib@v1.4',
        'seaborn==0.11.1',
        'requests==2.25.1',
        'absl-py==0.11.0'
    ],
    extras_require={
        'tf': ['tensorflow==1.14'],
        'tf_gpu': ['tensorflow-gpu==1.14'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='tensorflow machine learning disentanglement learning',
)
