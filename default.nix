{ pkgs ? import <nixpkgs> {}
, python ? pkgs.python3
, pyqdng ? null
}:

let
  py = python.pkgs;
in
{
  pyliic = py.buildPythonPackage {
    pname = "pyliic";
    version = "0.4.0";

    src = ./.;

    pyproject = true;

    build-system = with py; [
      setuptools
      wheel
    ];

    propagatedBuildInputs = with py; [
      numpy
      pyyaml
      scipy
      matplotlib
    ] ++ pkgs.lib.optionals (pyqdng != null) [
      pyqdng
    ];

    doCheck = false;
  };
}
