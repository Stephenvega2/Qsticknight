{pkgs}: {
  deps = [
    pkgs.xsimd
    pkgs.libxcrypt
    pkgs.pkg-config
    pkgs.mtdev
    pkgs.libcxx
    pkgs.SDL2_ttf
    pkgs.SDL2_mixer
    pkgs.SDL2_image
    pkgs.SDL2
    pkgs.glibcLocales
    pkgs.postgresql
    pkgs.libGLU
    pkgs.libGL
    pkgs.spdlog
    pkgs.nlohmann_json
    pkgs.muparserx
    pkgs.fmt
    pkgs.catch2
    pkgs.gmp
  ];
}
