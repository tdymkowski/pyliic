{ nixpkgs ? import <nixpkgs> {} }:

let
  pkgs = nixpkgs;

in
{
  pyliic = pkgs.python3.pkgs.buildPythonPackage {
    pname = "pyliic";
    version = "0.1.0";

    src = ./.;

    pyproject = true;

    build-system = with pkgs.python3.pkgs; [
      setuptools
      wheel
    ];

    dependencies = with pkgs.python3.pkgs; [
      numpy
      pyyaml
      scipy
      matplotlib
    ];

    doCheck = false;
  };
}
