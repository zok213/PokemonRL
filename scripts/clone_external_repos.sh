#!/usr/bin/env bash
# Clone all 6 foundational open-source repositories referenced in the survey
set -e

echo "[*] Cloning foundational open-source repositories into external/..."
mkdir -p external

repos=(
    "https://github.com/pret/pokered.git external/pokered"
    "https://github.com/PWhiddy/PokemonRedExperiments.git external/PokemonRedExperiments"
    "https://github.com/drubinstein/pokemonred_puffer.git external/pokemonred_puffer"
    "https://github.com/reddheeraj/PokemonRL.git external/PokeRL"
    "https://github.com/UT-Austin-RPL/metamon.git external/metamon"
    "https://github.com/sethkarten/continual-harness.git external/continual-harness"
)

for entry in "${repos[@]}"; do
    url=$(echo $entry | cut -d' ' -f1)
    dir=$(echo $entry | cut -d' ' -f2)
    if [ ! -d "$dir" ]; then
        echo "[+] Cloning $url -> $dir..."
        git clone --depth 1 "$url" "$dir"
    else
        echo "[=] $dir already exists."
    fi
done

echo "[*] All external repositories are up to date!"
