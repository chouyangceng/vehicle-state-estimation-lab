from setuptools import find_packages, setup

package_name = "vehicle_state_estimation_ros"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", ["config/estimator.yaml"]),
        (f"share/{package_name}/launch", ["launch/estimator.launch.py"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    entry_points={"console_scripts": ["estimator_node = vehicle_state_estimation_ros.estimator_node:main"]},
)
