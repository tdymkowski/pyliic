{ nixpkgs ? import <nixpkgs> {} }:

let
  pkgs = nixpkgs;
  pyqdngSrc = pkgs.fetchFromGitLab {
    url = "gitlab.fysik.su.se/markus.kowalewski/pyqdng.git";
    rev = "cdda7a85";
    hash = "sha256-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=";
  };
  pyqdng = import pyqdngSrc {
    pkgs = pkgs;
  };
in
{
  pyliic = pkgs.python3.pkgs.buildPythonPackage {
    pname = "pyliic";
    version = "0.2.0";

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
      pyqdng
    ];

    doCheck = false;
  };
}
