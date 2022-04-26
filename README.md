## Clear BOW
### 1.0 Overview
A cheap model that takes a formatted dictionary as input, and pushes word occurence counts through either a softmax (multi-class) or sigmoid (multi-label) function, to produce label "probabilities". Not nuanced. Not complex. But if you're using a dictionary you probably don't care about those things. Useful for bootstrapping classifications with raw terminology lists.

### 2.0 Install
Via pip:
```sh
pip install clean-py
```

Or clone directly:
```sh
git clone https://github.com/samhardyhey/clear-bow
cd clear_bow
pip install .
```

### 3.0 Usage
Instantiate, call:
```python
```