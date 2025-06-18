```shell
rm -rf dist build *.egg-info && python -m build
twine upload dist/*
```