@echo off
REM Clone all 6 foundational open-source repositories referenced in the survey
echo [*] Cloning foundational open-source repositories into external/...

if not exist "external" mkdir external

if not exist "external\pokered" (
    echo [+] Cloning pret/pokered...
    git clone --depth 1 https://github.com/pret/pokered.git external\pokered
) else (
    echo [=] external\pokered already exists.
)

if not exist "external\PokemonRedExperiments" (
    echo [+] Cloning PWhiddy/PokemonRedExperiments...
    git clone --depth 1 https://github.com/PWhiddy/PokemonRedExperiments.git external\PokemonRedExperiments
) else (
    echo [=] external\PokemonRedExperiments already exists.
)

if not exist "external\pokemonred_puffer" (
    echo [+] Cloning drubinstein/pokemonred_puffer...
    git clone --depth 1 https://github.com/drubinstein/pokemonred_puffer.git external\pokemonred_puffer
) else (
    echo [=] external\pokemonred_puffer already exists.
)

if not exist "external\PokeRL" (
    echo [+] Cloning reddheeraj/PokemonRL...
    git clone --depth 1 https://github.com/reddheeraj/PokemonRL.git external\PokeRL
) else (
    echo [=] external\PokeRL already exists.
)

if not exist "external\metamon" (
    echo [+] Cloning UT-Austin-RPL/metamon...
    git clone --depth 1 https://github.com/UT-Austin-RPL/metamon.git external\metamon
) else (
    echo [=] external\metamon already exists.
)

if not exist "external\continual-harness" (
    echo [+] Cloning sethkarten/continual-harness...
    git clone --depth 1 https://github.com/sethkarten/continual-harness.git external\continual-harness
) else (
    echo [=] external\continual-harness already exists.
)

echo [*] All external repositories are up to date!
