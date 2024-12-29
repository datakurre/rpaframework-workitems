{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:
{
  imports = [
    ./devenv/modules/python.nix
  ];

  languages.python.interpreter = pkgs.python312;
  languages.python.pyprojectOverrides = final: prev: {
    "operaton-tasks" = prev."operaton-tasks".overrideAttrs (old: {
      nativeBuildInputs =
        old.nativeBuildInputs
        ++ final.resolveBuildSystem ({
          "hatchling" = [ ];
        });
    });
  };

  enterShell = ''
    unset PYTHONPATH
    export UV_NO_SYNC=1
    export UV_PYTHON_DOWNLOADS=never
    export REPO_ROOT=$(git rev-parse --show-toplevel)
  '';

  packages = [
    pkgs.entr
    pkgs.gnumake
  ];

  git-hooks.hooks.treefmt = {
    enable = true;
    settings.formatters = [
      pkgs.nixfmt-rfc-style
    ];
  };
}
