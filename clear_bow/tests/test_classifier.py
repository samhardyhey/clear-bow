from clear_bow.classifier import DictionaryClassifier


def test_multi_class_single_simple(super_dictionary, example_doc):
    dc = DictionaryClassifier(label_dictionary=super_dictionary)
    pred = dc.predict_single(example_doc)
    key_set = set(super_dictionary.keys())
    key_set.add("no_label")  # null prediction
    assert set(pred.keys()).issubset(key_set)
    assert sum(list(pred.values())) <= 1.0  # softmax sum < 1


def test_multi_class_batch_simple(super_dictionary, example_docs):
    dc = DictionaryClassifier(label_dictionary=super_dictionary)
    preds = dc.predict_batch(example_docs)
    for pred in preds:
        key_set = set(super_dictionary.keys())
        key_set.add("no_label")
        assert set(pred.keys()).issubset(key_set)
        assert sum(list(pred.values())) <= 1.0


def test_multi_label_single_simple(super_dictionary, example_doc):
    dc = DictionaryClassifier(classifier_type="multi_label", label_dictionary=super_dictionary)
    pred = dc.predict_single(example_doc)

    assert set(pred.keys()) == set(super_dictionary.keys())
    for v in pred.values():
        assert type(v) == float
        assert v >= 0.5  # zero vector floor values
        assert v <= 1.0


def test_multi_label_batch_simple(super_dictionary, example_docs):
    dc = DictionaryClassifier(classifier_type="multi_label", label_dictionary=super_dictionary)
    preds = dc.predict_batch(example_docs)

    for pred in preds:
        key_set = set(super_dictionary.keys())
        key_set.add("no_label")
        assert set(pred.keys()).issubset(key_set)
        for k, v in pred.items():
            if k == "no_label":
                assert type(v) == float
                assert v == 1.0
            else:
                assert 0.5 <= v <= 1.0
