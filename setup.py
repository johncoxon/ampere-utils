import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setuptools.setup(author="John Coxon",
                 author_email="work@johncoxon.co.uk",
                 classifiers=[
                    "Development Status :: 4 - Beta",
                    "Intended Audience :: Science/Research",
                    "Natural Language :: English",
                    "Programming Language :: Python :: 3",
                 ],
                 description="Simple utilities for working with AMPERE data.",
                 install_requires=[
                     "lmfit",
                     "matplotlib",
                     "numpy",
                     "pandas",
                     "xarray"
                 ],
                 long_description=long_description,
                 long_description_content_type="text/markdown",
                 name="ampere_utils",
                 package_dir={"": "src"},
                 packages=setuptools.find_packages(where="src"),
                 python_requires=">=3.9",
                 url="https://github.com/johncoxon/ampere-utils",
                 version="1.0",
                 )
