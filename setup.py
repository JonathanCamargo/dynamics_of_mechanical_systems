from setuptools import setup, find_packages

# Read the contents of your README file to include it in the long_description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="dynamics_of_mechanical_systems",  # Name of your package
    version="0.1.0",  # Initial version
    author="Jonathan Camargo",  # Author's name
    author_email="jon-cama@uniandes.edu.co",  # Replace with your email
    description="A package for dynamics of mechanical systems",  # Short description
    long_description=long_description,  # Long description from README
    long_description_content_type="text/markdown",  # Format of the README
    url="https://github.com/JonathanCamargo/dynamics_of_mechanical_systems",  # Link to your GitHub repo
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "sympy",
        "matplotlib",
    ],
    extras_require={  # Optional dependencies (for development, testing, etc.)
        "dev": [
            "pytest>=6.0",
            "sphinx>=4.0",
        ],
    },
    python_requires=">=3.8",
    include_package_data=True,  # Include non-Python files (e.g., data, docs)
    package_data={
        "dms": ["assets/mujoco_models/*.xml"],
    },
    entry_points={  # If you have command-line scripts, add them here
        "console_scripts": [
            # "mycli=my_module.cli:main",
        ],
    },
)
