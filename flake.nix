{
  description = "my steno project in python";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable-small";
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs @ { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        poetry2nix = inputs.poetry2nix.lib.mkPoetry2Nix { inherit pkgs; };
      in
        {
          packages = {
            yourPackage = poetry2nix.mkPoetryApplication {
              projectDir = self;

              # set this to true to use premade wheels rather than the source
              preferWheels = true;

              # this enables interactive plotting support with GTK
              overrides = poetry2nix.overrides.withDefaults (final: prev: {
                matplotlib = with pkgs; prev.matplotlib.overridePythonAttrs 

                  {
                    passthru.args.enableGtk3 = true;
                  };
              });
            };
            default = self.packages.${system}.yourPackage;
          };

          # Shell for app dependencies.
          #
          # nix develop
          #
          # Use this shell for developing your app.
          devShells.default = pkgs.mkShell {
            inputsFrom = [self.packages.${system}.yourPackage];
            package = with pkgs; [
              # any development dependencies that you might have in nixpkgs
              ruff
              pyright
              poetry
            ];
          };

          # Shell for poetry.
          #
          # nix develop .#poetry
          #
          # Use this shell for changes to pyproject.toml and poetry.lock.
          devShells.poetry = pkgs.mkShell {
            packages = [pkgs.poetry];
          };
        });
}
