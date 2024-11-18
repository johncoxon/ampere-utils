# encoding: utf-8
from pandas import concat, read_csv, to_datetime
from xarray import Dataset, merge


def milan_interfaces(years, data_path):
    ds_list = []

    for h in ["north", "south"]:
        df_list = []

        for year in years:
            filename = f"AMPERE_R1R2_radii_v2_{year}_{h}.txt"
            df = read_csv(data_path / "Milan interfaces" / filename,
                          names=["day", "time", "R1", "R1_R2_interface", "R2", "HMB", "x0", "y0", "q"],
                          header=47, sep="\\s+")
            df.index = to_datetime(df.day.astype(str) + df.time, format="%Y%m%d%H:%S")
            df_list.append(df.drop(columns=["day", "time"]))

        df = concat(df_list)
        df["hemisphere"] = h
        df = df.set_index([df.index, "hemisphere"])
        ds_list.append(Dataset.from_dataframe(df).rename({"level_0": "time"}))

    boundaries = merge(ds_list)

    return boundaries
