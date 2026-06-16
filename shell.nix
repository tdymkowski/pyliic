{ nixpkgs ? import <nixpkgs> {} }:

let
  pkgs = nixpkgs;
  packages = import ./default.nix { inherit pkgs; };

in
pkgs.mkShell {
  packages = [
    packages.pyliic
    pkgs.python3
  ];
}
