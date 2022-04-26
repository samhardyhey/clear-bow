import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)


def get_multi_class_dict_preds(
    data_split, artefact_dir, text_col, label_col, split_type="test", save=True
):
    assert split_type in ["dev", "test"], "Split type must be either dev/test"
    file_name = "dev_preds" if split_type == "dev" else "test_preds"

    # instantiate model, using wrapper (doubles as production pseudo-test)
    from clear_bow.model import DictionaryClassifier

    classifier = DictionaryClassifier(artefact_dir)

    # get x, y_true and y_pred, cast back to label dict format
    x = data_split[text_col]
    y_true = label_list_to_label_dictionary(data_split[label_col])
    y_pred = classifier.predict_batch(x)

    # format preds df
    preds = pd.DataFrame(np.column_stack((x, y_true, y_pred))).set_axis(
        labels=[text_col, f"{label_col}_true", f"{label_col}_pred"],
        axis="columns",
        inplace=False,
    )

    if save:
        preds.to_csv(artefact_dir / f"{file_name}.csv", index=False)
        logging.info(f"Successfully wrote {file_name}.csv to {artefact_dir}")

    return preds


def get_multi_label_dict_preds(
    data_split, artefact_dir, text_col, label_cols, split_type="test", save=True
):
    assert split_type in ["dev", "test"], "Split type must be either dev/test"
    file_name = "dev_preds" if split_type == "dev" else "test_preds"

    # instantiate model, using wrapper (doubles as production pseudo-test)
    from clear_bow.model import DictionaryClassifier

    classifier = DictionaryClassifier(artefact_dir)

    # get x, y_true and y_pred
    x = data_split[text_col]
    y_true = label_mat_to_label_dictionary(data_split[label_cols])
    y_pred = classifier.predict_batch(x)

    # format preds df
    preds = pd.DataFrame(
        {text_col: data_split[text_col], "y_true": y_true, "y_pred": y_pred}
    )

    if save:
        preds.to_csv(artefact_dir / f"{file_name}.csv", index=False)
        logging.info(f"Successfully wrote {file_name}.csv to {artefact_dir}")

    return preds
