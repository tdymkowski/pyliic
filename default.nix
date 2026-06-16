{ pkgs ? import <nixpkgs> {} }:

let
  nixpkgs = builtins.fetchTarball {
    url = "https://github.com/NixOS/nixpkgs/archive/ca77296380960cd497a765102eeb1356eb80fed0.tar.gz";
  };

  pkgs = import nixpkgs {};
  python = pkgs.python3;

  pyqdngSrc = pkgs.fetchgit {
    url = "https://gitlab.fysik.su.se/markus.kowalewski/pyqdng.git";
    rev = "cdda7a857a6b72e34939f9c172e9828d4b80e0a8";
    hash = "sha256-KLPn+Z8VyEgidgK32VVrf68n42IcmRHFnjfSMKKGBYk=";
  };

  pyqdng = import pyqdngSrc {};

in
{
  pyliic = python.pkgs.buildPythonPackage {
    pname = "pyliic";
    version = "0.2.0";

    src = ./.;

    pyproject = true;

    build-system = with python.pkgs; [
      setuptools
      wheel
    ];

    propagatedBuildInputs = with python.pkgs; [
      numpy
      pyyaml
      scipy
      matplotlib
      pyqdng
    ];

    doCheck = false;
  };
}

