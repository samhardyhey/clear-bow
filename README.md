# Clear BOW 📚

Lightweight dictionary-based classifier that converts word frequencies into label probabilities using softmax/sigmoid functions. Perfect for bootstrapping classifications with terminology lists.

## Features
- 🔍 Dictionary-based classification
- 📊 Multi-class (softmax) support
- 🏷️ Multi-label (sigmoid) support
- 📝 Simple terminology lists
- 🔢 Probability outputs

## Installation
```bash
# Via pip
pip install clear_bow

# Or from source
git clone https://github.com/samhardyhey/clear-bow
cd clear-bow
pip install .
```

## Usage
```python
from clear_bow.classifier import DictionaryClassifier

# Define your dictionary
super_dict = {
    "regulation": ["asic", "government", "federal", "tax"],
    "contribution": ["contribution", "concession", "personal", "after tax"],
    "fund": ["unisuper", "aus super", "sun super", "qsuper"],
}

# Create and use classifier
dc = DictionaryClassifier(label_dictionary=super_dict)
result = dc.predict_single("A 10% contribution to your super fund")

# Returns probability distribution across labels
```

## Development
```bash
# Run tests
pytest

# Multi-environment testing
tox

# Build distribution
python setup.py sdist bdist_wheel

# Upload to PyPI
twine upload dist/*
```

## Structure
- 🎯 `clear_bow/` # Core package
  - `classifier.py` # Main logic
  - `tests/` # Test suite
- 📝 `setup.py` # Package config

*Note: See tests for additional usage examples.*