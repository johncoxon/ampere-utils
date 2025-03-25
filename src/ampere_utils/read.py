# encoding: utf-8
import datetime as dt
import numpy as np
from scipy.io import netcdf_file
from pandas import concat, read_csv, to_datetime
from pathlib import Path
from xarray import Dataset, merge, open_mfdataset
from . import colat_and_mlt


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


def ncdf_file_from_path(filepath, read_db=False, read_geo_coordinates=False):
    """
    Return a dict which contains data from an AMPERE data file.

    Parameters
    ----------
    filepath : str
        The path to the data you want to read.
    read_db : bool, optional, default False
        Set this to read the magnetic perturbations.
    read_geo_coordinates : bool, optional, default False
        Set this to read GEO coordinates from the data files.

    Returns
    -------
    ncdf_dict : dictionary
        A dictionary containing "time", "colat", "mlt", "j".
    """
    expected_colat, expected_mlt = colat_and_mlt()

    with netcdf_file(filepath) as f:
        empty_file = (len(f.variables.keys()) == 0)

        if not empty_file:
            times = np.zeros(f.variables["year"][:].shape[0], dtype=object)

            for cnt, _ in enumerate(times):
                # Get the year from the file.
                times[cnt] = (dt.datetime(f.variables["year"][cnt], 1, 1) +
                              # Add the day of the year.
                              dt.timedelta(days=int(f.variables["doy"][cnt]) - 1) +
                              # Add the fractional hour.
                              dt.timedelta(seconds=int(np.round(60 * 60 * f.variables["time"][cnt]))) +
                              # Correct to the midpoint using the length of the window in seconds.
                              dt.timedelta(seconds=f.variables["avgint"][cnt] / 2))

            # Check that all the standard variables are what we think they should be.
            # (These are the variables that we're not going to read out of the file.)
            expected_values = {"kmax": np.array([60]),
                               "mmax": np.array([8]),
                               "res_deg": np.array([3]),
                               "nLatGrid": np.array([50]),
                               "nLonGrid": np.array([24])}

            for variable in expected_values:
                unique_variable = np.sort(np.unique(np.round(f.variables[variable][:]))).astype(int)

                if unique_variable.shape != expected_values[variable].shape:
                    raise ValueError(f"{variable} does not agree with expected unique shape.")
                elif (unique_variable != expected_values[variable]).any():
                    raise ValueError(f"{variable} does not agree with expected unique values.")

            # Check the residuals from the data file (we're also not reading these in).
            residuals = ["del_db_R", "del_db_T", "del_db_P", "del_db_geo", "del_jPar",
                         "del_db_Th_Th", "del_db_Ph_Th", "del_db_Th_Ph", "del_db_Ph_Ph"]

            for residual in residuals:
                unique_residuals = np.unique(f.variables[residual][:])
                if (unique_residuals.shape != (1,)) or (unique_residuals[0] != 0):
                    raise ValueError(f"residual of {residual[4:]} is non-zero.")

            # Read in the actual data.
            colat = np.round(f.variables["cLat_deg"][0, :]).astype(int)
            if (np.sort(np.unique(colat)) == expected_colat).all():
                hemisphere = "north"
            elif (np.sort(np.unique(colat)) == np.arange(130, 180)).all():
                hemisphere = "south"
            else:
                raise ValueError(f"colat does not agree with expected unique values.")

            mlt = np.round(f.variables["mlt_hr"][0, :]).astype(int)
            mlt[mlt == 24] = 0
            if (np.sort(np.unique(mlt)) != expected_mlt).all():
                raise ValueError(f"MLT does not agree with expected unique values.")

            j_dict = {
                "time": times,
                "colat": colat,
                "mlt": mlt,
                "j": f.variables["jPar"][:].copy(),
                "hemisphere": hemisphere
            }

            if read_db:
                db_dictionary = {
                    "db_geo_r": "db_R",
                    "db_geo_theta": "db_T",
                    "db_geo_phi": "db_P",
                    "db_geo_xyz": "db_geo",
                    "db_aacgm_north_par": "db_Th_Th",
                    "db_aacgm_north_perp": "db_Ph_Th",
                    "db_aacgm_east_par": "db_Ph_Ph",
                    "db_aacgm_east_perp": "db_Th_Ph"
                }

                for dict_key in db_dictionary:
                    j_dict[dict_key] = f.variables[db_dictionary[dict_key]][:].copy()

            if read_geo_coordinates:
                geo_dictionary = {
                    "geo_colat": "geo_cLat_deg",
                    "geo_lon": "geo_lon_deg",
                    "geo_r": "R",
                    "geo_vehicle_position": "pos_geo"
                }

                for dict_key in geo_dictionary:
                    j_dict[dict_key] = f.variables[geo_dictionary[dict_key]][:].copy()

    if empty_file:
        raise FileNotFoundError("File was empty.")
    else:
        ncdf_dict = j_dict

    return ncdf_dict


def xarray_dataset(paths, drop_variables=("npnt", "kmax", "mmax", "res_deg",
                                          "nLatGrid", "nLonGrid", "geo_cLat_deg", "geo_lon_deg", "R", "pos_geo",
                                          "db_R", "db_T", "db_P", "db_geo",
                                          "db_Th_Th", "db_Ph_Th", "db_Th_Ph", "db_Ph_Ph",
                                          "del_db_R", "del_db_T", "del_db_P", "del_db_geo", "del_jPar",
                                          "del_db_Th_Th", "del_db_Ph_Th", "del_db_Th_Ph", "del_db_Ph_Ph")):
    """
    Read AMPERE data to an xarray Dataset.

    Parameters
    ----------
    paths : str or listlike of str or Path
        Either a glob string that will identify the files, or a list of the files as strings or Path objects.
    drop_variables : tuple, optional
        A tuple of strings describing variables to drop from the dataset.
    """
    vars_to_drop = ("year", "doy", "avgint", "cLat_deg", "mlt_hr")
    vars_with_one_dimension = ("npnt", "kmax", "mmax", "res_deg", "nLatGrid", "nLonGrid")
    vars_geo = ("pos_geo", "db_geo", "del_db_geo")

    dataset = open_mfdataset(paths, combine='nested', concat_dim="nRec", drop_variables=drop_variables)

    years = dataset.variables["year"].values
    doys = dataset.variables["doy"].values
    time = dataset.variables["time"].values
    time_window_length = dataset.variables["avgint"].values

    time_length = time.shape[0]

    times = np.zeros(time_length, dtype=object)

    for cnt, _ in enumerate(times):
        # Get the year from the file.
        if np.isnan(years[cnt]):
            times[cnt] = np.nan
        else:
            times[cnt] = (dt.datetime(int(years[cnt]), 1, 1) +
                          # Add the day of the year.
                          dt.timedelta(days=int(doys[cnt]) - 1) +
                          # Add the fractional hour.
                          dt.timedelta(seconds=int(np.round(60 * 60 * time[cnt]))) +
                          # Correct to the midpoint using the length of the window in seconds.
                          dt.timedelta(seconds=int(time_window_length[cnt]) / 2))

    coordinates = {"time": times,
                   "mlt": dataset.mlt_hr.values.reshape(time_length, 24, 50)[0, :, 0],
                   "colat": dataset.cLat_deg.values.reshape(time_length, 24, 50)[0, 0, :]}

    for variable in vars_geo:
        if variable in dataset:
            coordinates["component"] = ["x", "y", "z"]
            break

    dataset = dataset.assign_coords(coordinates)

    for variable in dataset:
        if variable not in vars_to_drop and variable not in vars_with_one_dimension:
            if variable in vars_geo:
                dataset[variable] = (
                    ("time", "mlt", "colat", "component"), dataset[variable].values.reshape(time_length, 24, 50, 3))
            else:
                dataset[variable] = (("time", "mlt", "colat"), dataset[variable].values.reshape(time_length, 24, 50))
    dataset.close()

    dataset = dataset.rename(jPar="j").drop_vars(vars_to_drop)

    return dataset