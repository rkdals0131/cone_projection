from setuptools import find_packages, setup

package_name = 'cone_projection'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user1',
    maintainer_email='kikiws70@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'project_sorted_cones = cone_projection.project_sorted_cones:main',
            'visualize_sorted_cones = cone_projection.visualize_sorted_cones:main',
            'read_yaml = cone_projection.read_yaml:main',
        ],
    },
)
