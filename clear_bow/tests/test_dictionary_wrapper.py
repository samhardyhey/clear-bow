from clear_bow.model import DictionaryClassifier


def test_multi_class_dictionary_classifier(label_dictionaries, text):
    dc = DictionaryClassifier(label_dictionaries=label_dictionaries)
    res = dc.predict_single(text)
    key_set = set(label_dictionaries.keys())
    key_set.add("no_label")
    assert set(res.keys()).issubset(key_set)
    assert sum(list(res.values())) <= 1.0


def test_multi_label_dictionary_classifier(label_dictionaries, text):
    dc = DictionaryClassifier(classifier_type="multi_label", label_dictionaries=label_dictionaries)
    res = dc.predict_single(text)

    assert set(res.keys()) == set(label_dictionaries.keys())
    for v in res.values():
        assert type(v) == float
        assert v >= 0.5
        assert v <= 1.0


def test_multi_label_dictionary_classifier_no_label(wrong_label_dictionaries, text):
    dc = DictionaryClassifier(classifier_type="multi_label", label_dictionaries=wrong_label_dictionaries)
    res = dc.predict_single(text)
    key_set = set(wrong_label_dictionaries.keys())
    key_set.add("no_label")
    assert set(res.keys()).issubset(key_set)

    for k, v in res.items():
        if k == "no_label":
            assert type(v) == float
            assert v == 1.0
        else:
            assert v == 0.0
