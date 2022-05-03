## Clear BOW
### 1.0 Overview
A cheap model that takes a formatted dictionary as input, and pushes word occurence counts through either a softmax (multi-class) or sigmoid (multi-label) function, to produce label "probabilities". Not nuanced. Not complex. But if you're using a dictionary you probably don't care about those things. Useful for bootstrapping classifications with raw terminology lists.

### 2.0 Install
Via pip:
```sh
pip install clear_bow
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
from clear_bow.classifier import DictionaryClassifier

# define, instantiate, call
super_dict = {
    "regulation": ["asic", "government", "federal", "tax"],
    "contribution": ["contribution", "concession", "personal", "after tax", "10%", "10.5%"],
    "covid": ["covid", "lockdown", "downturn", "effect"],
    "retirement": ["retire", "house", "annuity", "age"],
    "fund": ["unisuper", "aus super", "australian super", "sun super", "qsuper", "rest", "cbus"],
}

# multi-class/label options available
dc = DictionaryClassifier(label_dictionary=super_dict)
dc.predict_single("A 10% contribution is not enough for a well balanced super fund!")

# {'regulation': 0.0878,
#  'contribution': 0.6488,
#  'covid': 0.0878,
#  'retirement': 0.0878,
#  'fund': 0.0878}
```

See tests for additinal usage.