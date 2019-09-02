import os

import math
import numpy as np
import pandas as pd

from default_config import DF_LOCATION


def get_filename(i):
    return os.path.basename(os.path.normpath(i))


def get_dataframe(ds_location, is_test=False):
    if is_test:
        folder_name = "test"
    else:
        folder_name = "train"
    filename = f"df_{folder_name}.pkl"
    if os.path.exists(filename):
        print("Loading existing df!")
        return pd.read_pickle(filename)

    df = get_merged_df(ds_location, folder_name, is_test).reset_index(drop=True)
    df = merge_by_channels_and_sites(df, is_test)
    if not is_test:
        df["sirna"] = df["sirna"].astype(int)
    else:
        df = df[df["well_type"] == "treatment"].reset_index(drop=True)
    df = df.replace(np.nan, '', regex=True)
    df.to_pickle(filename)
    return df


def merge_by_channels_and_sites(df, is_test):
    indexes = ["well_column", "well_row", "cell_line", "batch_number", "plate", "id_code", "well_type"]
    if not is_test:
        indexes.append("sirna")
    pivoted = df.pivot_table(
        index=indexes,
        columns=["site_num", "microscope_channel"],
        values="img_location",
        aggfunc='first'
    )

    pivoted.columns = ["img_" + '_'.join(str(s).strip() for s in col if s) for col in pivoted.columns]
    pivoted = pivoted.reset_index(drop=False)

    return pivoted


def get_merged_df(ds_location, dataset_type, is_test):
    sirna_df = pd.read_csv(os.path.join(ds_location, f"{dataset_type}.csv"))
    controls_df = pd.read_csv(os.path.join(ds_location, f"{dataset_type}_controls.csv"))
    data = []
    tests = [os.path.join(ds_location, dataset_type, t) for t in os.listdir(os.path.join(ds_location, dataset_type))]
    for t in tests:
        print(f"Loading: {str(t)}")
        plates = [os.path.join(t, p) for p in os.listdir(t)]
        for p in plates:
            imgs = [os.path.join(p, i) for i in os.listdir(p)]
            for i in imgs:
                f_name = get_filename(i)
                parts = f_name.split("_")

                well = str(parts[0])
                well_column = well[0]
                well_row = int(well.replace(well_column, ""))
                site = int(str(parts[1]).replace("s", ""))
                microscope_channel = int(str(parts[2]).replace(".png", "").replace("w", ""))
                test = get_filename(t)
                cell_line = test.split("-")[0]
                batch_number = int(test.split("-")[1])
                plate = int(str(get_filename(p)).replace("Plate", ""))

                data.append({
                    "well_column": well_column,
                    "well_row": well_row,
                    "site_num": site,
                    "microscope_channel": microscope_channel,
                    "cell_line": cell_line,
                    "batch_number": batch_number,
                    "plate": plate,
                    "img_location": i,
                    "id_code": f"{cell_line}-{batch_number:02d}_{plate}_{well_column}{well_row:02d}"
                })

    return add_sirna(sirna_df, pd.DataFrame(data), controls_df, is_test)


def add_sirna(sirna_df, metadata_df, controls_df, is_test):
    if is_test:
        metadata_with_sirna = pd.merge(metadata_df, sirna_df[["id_code"]], on="id_code", how="left")
    else:
        metadata_with_sirna = pd.merge(metadata_df, sirna_df[["id_code", "sirna"]], on="id_code", how="left")
        sirnas = metadata_with_sirna.pop("sirna")
    control_merged = pd.merge(
        metadata_with_sirna,
        controls_df[["id_code", "well_type", "sirna"]],
        on="id_code", how="left"
    )
    control_merged["well_type"].fillna("treatment", inplace=True)
    if not is_test:
        set_sirna(control_merged, sirnas)
    return control_merged


def set_sirna(control_merged, sirnas):
    sirnas2 = control_merged.pop("sirna")
    sirnas3 = []
    for s1, s2 in zip(sirnas, sirnas2):
        if not math.isnan(s1):
            sirnas3.append(s1)
        elif not math.isnan(s2):
            sirnas3.append(s2)
        else:
            raise
    control_merged["sirna"] = sirnas3


if __name__ == '__main__':
    cheat_dict = {}
    train_df = get_dataframe(DF_LOCATION, is_test=False)
    # for (cell_line, plate, batch_number), df_g \
    #         in train_df.groupby(["cell_line", "plate", "batch_number"]):
    #     without_controls = df_g[df_g["well_type"] == ""]
    #     unique_sirnas = without_controls["sirna"].unique()
    #     cheat_dict[f"{cell_line}_{batch_number}_{plate}"] = unique_sirnas.tolist()
    #     print("")
    test_df = get_dataframe(DF_LOCATION, is_test=True)
    print("")
