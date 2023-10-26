import camb
from camb import model
import numpy as np

from cobaya.yaml import yaml_load_file
import arviz as az

# Export the results to GetDist
from getdist.mcsamples import loadMCSamples

# Notice loadMCSamples requires a *full path*
import os

import argparse


def compute_sigma8(
    vscale: float,
    omh2: float,  # total matter density
    n_p: float,  # this is spectral index at 0.78 h/mpc
    A_p: float,  # this is amplitude at 0.79 h/mpc
    ombh2: float = 0.0224,  # total baryon matter density. this is fixed
):
    omch2 = omh2 - ombh2  # this is total cdm density, total - baryon
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=vscale * 100, ombh2=ombh2, omch2=omch2)
    pars.InitPower.set_params(ns=n_p, As=A_p, pivot_scalar=0.78, pivot_tensor=0.78)

    pars.set_matter_power(redshifts=[0.0], kmax=30.0)

    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)

    s8 = np.array(results.get_sigma8())

    return s8


def main(path_yaml_load_file, folder_dir: str = "Chains/fps-meant/"):
    info_from_yaml = yaml_load_file(path_yaml_load_file)
    gd_sample = loadMCSamples(os.path.join(folder_dir, info_from_yaml["output"]))

    # necessary parameters to compute sigma_8
    # note ns is np. Ap and np are at pivot of 0.78 h/Mpc
    # omegamh2 is totoal matter density
    ns_samples = gd_sample["ns"]
    Ap_samples = gd_sample["Ap"]
    omh2_samples = gd_sample["omegamh2"]
    vscale_samples = gd_sample["hub"]

    n_samples = ns_samples.shape[0]

    sigma_8_samples = []

    for i in range(n_samples):
        sigma_8 = compute_sigma8(
            vscale=vscale_samples[i],
            omh2=omh2_samples[i],
            n_p=ns_samples[i],
            A_p=Ap_samples[i],
        )

        if i % 1000 == 0:
            print("[Info]", i)

        sigma_8_samples.append(sigma_8)

    return sigma_8_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_yaml_load_file",
        type=str,
        help="the yaml file for Cobaya",
    )

    parser.add_argument(
        "--folder_dir",
        type=str,
        help="the folder_dir of the sample files",
    )

    parser.add_argument(
        "--outfile_name",
        type=str,
        help="name of the sigma_8 samples",
    )

    args = parser.parse_args()

    sigma_8_samples = main(args.path_yaml_load_file, args.folder_dir)

    np.savetxt("sigma_8")
