import setuptools
import re
import os

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

dependency_links = []
del_ls = []
for i_l in range(len(required)):
    l = required[i_l]
    if l.startswith("-e"):
        dependency_links.append(l.split("-e ")[-1])
        del_ls.append(i_l)
        required.append(l.split("=")[-1])

for i_l in del_ls[::-1]:
    del required[i_l]

setuptools.setup(
    name="guidebook",
    version="0.0.9",
    author="Casey Schneider-mizell",
    author_email="caseys@alleninstitute.org",
    description="First stab at endpoint for proofreading",
    install_requires=required,
    include_package_data=True,
    dependency_links=dependency_links,
    packages=["guidebook"]
)
