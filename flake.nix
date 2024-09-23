{
  description = "A nix flake for the Atoma Node Inference repository.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

    crane.url = "github:ipetkov/crane";

    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.rust-analyzer-src.follows = "";
    };

    flake-utils.url = "github:numtide/flake-utils";

    advisory-db = {
      url = "github:rustsec/advisory-db";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, crane, fenix, flake-utils, advisory-db, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          # Nvidia packages are proprietary, so this must be set.
          # Optionally, we could determine the minimum requirements
          # and allow only those pkgs with unfree licenses.
          config.allowUnfree = true;
        };

        inherit (pkgs) lib;

        craneLib = crane.mkLib pkgs;
        src = ./.;

        # Common arguments can be set here to avoid repeating them later
        commonArgs = {
          inherit src;
          strictDeps = true;

          buildInputs = with pkgs; [
            # Add additional build inputs here
            # TODO: Not all packages listed are available for every system.
            # In the future, it may be best to determine what hardware is
            # officially supported and write conditions for what can be
            # built for which system.
            git gitRepo gnupg autoconf curl
            procps gnumake util-linux m4 gperf unzip
            cudatoolkit linuxPackages.nvidia_x11
            libGLU libGL
            xorg.libXi xorg.libXmu freeglut
            xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
            ncurses5 stdenv.cc binutils
            openssl.dev
            pkg-config
          ] ++ lib.optionals pkgs.stdenv.isDarwin [
            # Additional darwin specific inputs can be set here
            pkgs.libiconv
          ];

          # Additional environment variables can be set directly
          # MY_CUSTOM_VAR = "some value";
          CUDA_PATH="${pkgs.cudatoolkit}";
          LD_LIBRARY_PATH="${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib";
          EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib";
          EXTRA_CCFLAGS="-I/usr/include";
          # In some cases this must be set in order for candle to compile, but it is
          # dependent on your system hardware.
          # Run `nvidia-smi --query-gpu=compute_cap --format=csv`
          # Output should look like: compute_cap 8.9
          # Then export the following env var like so:
          # `export CUDA_COMPUTE_CAP=89`
        };

        craneLibLLvmTools = craneLib.overrideToolchain
          (fenix.packages.${system}.complete.withComponents [
            "cargo"
            "llvm-tools"
            "rustc"
          ]);

        # Build *just* the cargo dependencies (of the entire workspace),
        # so we can reuse all of that work (e.g. via cachix) when running in CI
        # It is *highly* recommended to use something like cargo-hakari to avoid
        # cache misses when building individual top-level-crates
        cargoArtifacts = craneLib.buildDepsOnly commonArgs;

        individualCrateArgs = commonArgs // {
          inherit cargoArtifacts;
          inherit (craneLib.crateNameFromCargoToml { inherit src; }) version;
          # NB: we disable tests since we'll run them all via cargo-nextest
          doCheck = false;
        };

        # fileSetForCrate = crate: lib.fileset.toSource {
        #   root = ./.;
        #   fileset = lib.fileset.unions [
        #     ./Cargo.toml
        #     ./Cargo.lock
        #     ./csrc/cutlass
        #     crate
        #   ];
        # };

        # Build the top-level crates of the workspace as individual derivations.
        # This allows consumers to only depend on (and build) only what they need.
        # Though it is possible to build the entire workspace as a single derivation,
        # so this is left up to you on how to organize things
        csrc = craneLib.buildPackage (individualCrateArgs // {
          pname = "csrc";
          cargoExtraArgs = "-p csrc";
          # src = fileSetForCrate ./csrc;
        });
        models = craneLib.buildPackage (individualCrateArgs // {
          pname = "models";
          cargoExtraArgs = "-p models";
          # src = fileSetForCrate ./models;
        });
      in
      {
        checks = {
          # Build the crates as part of `nix flake check` for convenience
          inherit csrc models;

          # Run clippy (and deny all warnings) on the workspace source,
          # again, reusing the dependency artifacts from above.
          #
          # Note that this is done as a separate derivation so that
          # we can block the CI if there are issues here, but not
          # prevent downstream consumers from building our crate by itself.
          workspace-clippy = craneLib.cargoClippy (commonArgs // {
            inherit cargoArtifacts;
            cargoClippyExtraArgs = "--all-targets -- --deny warnings";
          });

          workspace-doc = craneLib.cargoDoc (commonArgs // {
            inherit cargoArtifacts;
          });

          # Check formatting
          workspace-fmt = craneLib.cargoFmt {
            inherit src;
          };

          workspace-toml-fmt = craneLib.taploFmt {
            src = pkgs.lib.sources.sourceFilesBySuffices src [ ".toml" ];
            # taplo arguments can be further customized below as needed
            # taploExtraArgs = "--config ./taplo.toml";
          };

          # Audit dependencies
          workspace-audit = craneLib.cargoAudit {
            inherit src advisory-db;
          };

          # Audit licenses
          workspace-deny = craneLib.cargoDeny {
            inherit src;
          };

          # Run tests with cargo-nextest
          # Consider setting `doCheck = false` on other crate derivations
          # if you do not want the tests to run twice
          workspace-nextest = craneLib.cargoNextest (commonArgs // {
            inherit cargoArtifacts;
            partitions = 1;
            partitionType = "count";
          });
        };

        packages = {
          inherit csrc models;
        } // lib.optionalAttrs (!pkgs.stdenv.isDarwin) {
          my-workspace-llvm-coverage = craneLibLLvmTools.cargoLlvmCov (commonArgs // {
            inherit cargoArtifacts;
          });
        };

        apps = { };

        devShells.default = craneLib.devShell {
          # Inherit inputs from checks.
          checks = self.checks.${system};

          # Additional dev-shell environment variables can be set directly
          # MY_CUSTOM_DEVELOPMENT_VAR = "something else";
          CUDA_PATH="${pkgs.cudatoolkit}";
          LD_LIBRARY_PATH="${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib";
          EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib";
          EXTRA_CCFLAGS="-I/usr/include";
          # In some cases this must be set in order for candle to compile, but it is
          # dependent on your system hardware.
          # Run `nvidia-smi --query-gpu=compute_cap --format=csv`
          # Output should look like: compute_cap 8.9
          # Then export the following env var like so:
          # `export CUDA_COMPUTE_CAP=89`

          # Extra inputs can be added here; cargo and rustc are provided by default.
          packages = [ ];
        };
      });
}
